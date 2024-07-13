import os
from random import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.pytorch.cli import LightningCLI

from diffusion import DiffusionBridge
from backbones.ncsnpp import NCSNpp
from backbones.discriminator import Discriminator_large
from datasets import DataModule
from utils import compute_metrics, save_image_pair, save_preds, save_eval_images


class BridgeRunner(L.LightningModule):
    def __init__(
        self,
        generator_params,
        discriminator_params,
        diffusion_params,
        lr_g,
        lr_d,
        disc_grad_penalty_freq,
        disc_grad_penalty_weight,
        lambda_rec_loss,
        optim_betas,
        eval_mask,
        eval_subject,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.disc_grad_penalty_freq = disc_grad_penalty_freq
        self.disc_grad_penalty_weight = disc_grad_penalty_weight
        self.lambda_rec_loss = lambda_rec_loss
        self.optim_betas = optim_betas
        self.eval_mask = eval_mask
        self.eval_subject = eval_subject
        self.n_steps = diffusion_params['n_steps']
        self.n_recursions = diffusion_params['n_recursions']

        # Networks
        self.generator = NCSNpp(**generator_params)
        self.discriminator = Discriminator_large(**discriminator_params)

        # Configure diffusion
        self.diffusion = DiffusionBridge(**diffusion_params)

    def training_step(self, batch):
        x0, y, _ = batch
        
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        # Part 1: Train discriminator
        self.toggle_optimizer(optimizer_d)

        # Part 1.a: Train discriminator with real data
        # Sample a time step
        t = torch.randint(1, self.n_steps+1, (x0.shape[0],)).to(x0.device)

        # Sample x_{t-1} and x_t via forward process
        x_tm1 = self.diffusion.q_sample(t - 1, x0, y)
        x_t = self.diffusion.q_sample(t, x0, y)
        x_t.requires_grad = True

        # Perform real data prediction
        disc_out = self.discriminator(x_tm1, x_t, t)
        real_loss = self.adversarial_loss(disc_out, is_real=True)
        disc_real_acc = (disc_out > 0).float().mean()

        # Compute gradient penalty
        if self.global_step % self.disc_grad_penalty_freq == 0:
            grads = torch.autograd.grad(outputs=disc_out.sum(), inputs=x_t, create_graph=True)[0]
            grad_penalty = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = grad_penalty * self.disc_grad_penalty_weight
            real_loss += grad_penalty

        # Part 1.b: Train discriminator with fake data
        # Perform recursive x0 prediction
        x0_r = torch.zeros_like(x_t)
        for _ in range(self.n_recursions):
            x0_r = self.generator(torch.cat((x_t.detach(), y), axis=1), t, x_r=x0_r)
        x0_pred = x0_r

        # Posterior sampling q(x_{t-1} | x_t, y, x0_pred)
        x_tm1_pred = self.diffusion.q_posterior(t, x_t, x0_pred, y)

        # Perform fake data prediction
        disc_out = self.discriminator(x_tm1_pred, x_t, t)
        fake_loss = self.adversarial_loss(disc_out, is_real=False)
        disc_fake_acc = (disc_out < 0).float().mean()

        # Compute discriminator accuracy
        d_acc = (disc_real_acc + disc_fake_acc) / 2

        # Compute total loss
        d_loss = real_loss + fake_loss

        # Perform backprop
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Part 2: Train generator
        self.toggle_optimizer(optimizer_g)

        # Sample a time step
        t = torch.randint(1, self.n_steps+1, (x0.shape[0],)).to(x0.device)

        # Get x_t via forward process
        x_t = self.diffusion.q_sample(t, x0, y)

        # Perform recursive x0 prediction
        x0_r = torch.zeros_like(x_t)
        for _ in range(self.n_recursions):
            x0_r = self.generator(torch.cat((x_t.detach(), y), axis=1), t, x_r=x0_r)
        x0_pred = x0_r

        # Posterior sampling q(x_{t-1} | x_t, y, x0_pred)
        x_tm1_pred = self.diffusion.q_posterior(t, x_t, x0_pred, y)

        # Compute reconstruction loss
        rec_loss = F.l1_loss(x0_pred, x0, reduction="sum")

        # Compute adversarial loss
        adv_loss = self.adversarial_loss(
            self.discriminator(x_tm1_pred, x_t, t), is_real=True)
        
        # Compute total loss and perform backprop
        g_loss = self.lambda_rec_loss*rec_loss + adv_loss

        # Perform backprop
        self.manual_backward(g_loss)

        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Take lr scheduler step
        scheduler_g.step()
        scheduler_d.step()
        
        # Log losses
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/rec", rec_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/adv", adv_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/total", g_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        x0, y, _ = batch

        # Predict x0
        x0_pred = self.diffusion.sample_x0(y, self.generator)

        loss = F.mse_loss(x0_pred, x0)
        metrics = compute_metrics(x0, x0_pred)

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_psnr", metrics["psnr_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics["ssim_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)

        # Log sample images
        if batch_idx == 0 and self.global_rank == 0:
            path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
            save_image_pair(x0, x0_pred, path)

    def on_test_start(self):
        self.test_samples = []
        self.psnrs = []
        self.ssims = []
        self.mask = None
        self.subject_ids = None

        # Load mask for evaluation
        if self.eval_mask:
            self.mask = self.trainer.datamodule.test_dataset._load_data('mask')

        # Load subject ids for evaluation
        if self.eval_subject:
            self.subject_ids = self.trainer.datamodule.test_dataset.subject_ids

    def test_step(self, batch, batch_idx):
        x0, y, slice_idx = batch

        # Predict x0
        x0_pred = self.diffusion.sample_x0(y, self.generator)

        # Gather pred images across all ranks
        all_pred = self.all_gather(x0_pred)
        slice_indices = self.all_gather(slice_idx)
        
        if self.global_rank == 0:
            h, w = x0.shape[-2:]
            self.test_samples.extend(list(zip(
                slice_indices.flatten().tolist(),
                all_pred.reshape(-1, h, w).cpu().numpy())))
        
    def on_test_end(self):
        # Save predicted images
        if self.global_rank == 0:
            # Sort samples by slice index
            self.test_samples.sort(key=lambda x: x[0])
            
            # Extract pred images
            pred = np.array([x[1] for x in self.test_samples])
            slice_indices = np.array([x[0] for x in self.test_samples])

            # Remove repeated slices that can occur in multi-GPU setting
            _, locs = np.unique(slice_indices, return_index=True)
            pred = pred[locs]

            # Get source and target images
            dataset = self.trainer.datamodule.test_dataset
            source = dataset.source
            target = dataset.target

            # Save predictions
            path = os.path.join(self.logger.log_dir, "test_samples", "pred.npy")
            save_preds(pred, path)

            # Compute metrics and save report
            metrics = compute_metrics(
                gt_images=target,
                pred_images=pred,
                mask=self.mask,
                subject_ids=self.subject_ids,
                report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt")
            )

            # Print metrics
            print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"SSIM: {metrics['ssim_mean']:.2f} ± {metrics['ssim_std']:.2f}")

            # Save sample images
            indices = np.random.choice(len(dataset), 10)
            path = os.path.join(self.logger.log_dir, "test_samples")
            save_eval_images(
                source_images=source[indices],
                target_images=target[indices],
                pred_images=pred[indices],
                psnrs=metrics["psnrs"][indices],
                ssims=metrics["ssims"][indices],
                save_path=os.path.join(self.logger.log_dir, "test_samples")
            )

    def adversarial_loss(self, pred, is_real):
        loss = F.softplus(-pred) if is_real else F.softplus(pred)
        return loss.mean()
    
    def configure_optimizers(self):
        optimizer_g = Adam(self.generator.parameters(), lr=self.lr_g, betas=self.optim_betas)
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr_d, betas=self.optim_betas)
        
        # Learning rate schedulers
        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=self.trainer.max_epochs, eta_min=1e-5)
        scheduler_d = CosineAnnealingLR(optimizer_d, T_max=self.trainer.max_epochs, eta_min=1e-5)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


class _LightningCLI(LightningCLI):
    def instantiate_classes(self):
        # Log to checkpoint directory when testing
        if 'test' in self.parser.args and 'CSVLogger' in self.config.test.trainer.logger[0].class_path:
            exp_dir = os.path.dirname(os.path.dirname(self.config.test.ckpt_path))
            logger = self.config.test.trainer.logger[0]
            logger.init_args.save_dir = os.path.dirname(exp_dir)
            logger.init_args.name = os.path.basename(exp_dir)
            logger.init_args.version = "test"

        super().instantiate_classes()


def cli_main():
    cli = _LightningCLI(
        BridgeRunner,
        DataModule,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"}
    )


if __name__ == "__main__":
    cli_main()
