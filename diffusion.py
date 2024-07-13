
import numpy as np
import torch
import matplotlib.pyplot as plt
import lightning as L


class DiffusionBridge(L.LightningModule):
    def __init__(
            self,
            n_steps,
            gamma,
            beta_start,
            beta_end,
            n_recursions,
            consistency_threshold
        ):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.beta_start = beta_start
        self.beta_end = beta_end / n_steps
        self.n_recursions = n_recursions
        self.consistency_threshold = consistency_threshold

        # Define betas
        self.betas = self._get_betas()
        
        # Mean schedule
        s = np.cumsum(self.betas)**0.5
        s_bar = np.flip(np.cumsum(self.betas))**0.5
        mu_x0, mu_y, _ = self.gaussian_product(s, s_bar)

        # Scale gamma for number of diffusion steps
        gamma = gamma * self.betas.sum()
        
        # Noise schedule
        std = gamma * s / (s**2 + s_bar**2)

        # Convert to tensors
        self.register_buffer("s", torch.tensor(s))
        self.register_buffer("mu_x0", torch.tensor(mu_x0))
        self.register_buffer("mu_y", torch.tensor(mu_y))
        self.register_buffer("std", torch.tensor(std))

    def q_sample(self, t, x0, y):
        """ Sample q(x_t | x_0, y) """
        shape = [-1] + [1] * (x0.ndim - 1)

        mu_x0 = self.mu_x0[t].view(shape)
        mu_y = self.mu_y[t].view(shape)
        std = self.std[t].view(shape)

        x_t = mu_x0*x0 + mu_y*y + std*torch.randn_like(x0)
        
        return x_t.detach()

    def q_posterior(self, t, x_t, x0, y):
        """ Sample p(x_{t-1} | x_t, x0, y) """
        shape = [-1] + [1] * (x0.ndim - 1)

        std_t = self.s[t].view(shape)
        std_tm1 = self.s[t-1].view(shape)
        mu_x0_t = self.mu_x0[t].view(shape)
        mu_x0_tm1 = self.mu_x0[t-1].view(shape)
        mu_y_t = self.mu_y[t].view(shape)
        mu_y_tm1 = self.mu_y[t-1].view(shape)

        var_t = std_t**2
        var_tm1 = std_tm1**2
        var_t_tm1 = var_t - var_tm1 * (mu_x0_t / mu_x0_tm1)**2
        v = var_t_tm1 * (var_tm1 / var_t)

        x_tm1_mean = mu_x0_tm1 * x0 + mu_y_tm1 * y + \
            ((var_tm1 - v) / var_t).sqrt() * (x_t - mu_x0_t * x0 - mu_y_t * y)

        x_tm1 = x_tm1_mean + v.sqrt() * torch.randn_like(x_t)

        return x_tm1

    @torch.inference_mode()
    def sample_x0(self, y, generator):
        """ Sample p(x_0 | y) """
        # Set timesteps
        timesteps = torch.arange(self.n_steps, 0, -1, device=y.device)
        timesteps = timesteps.unsqueeze(1).repeat(1, y.shape[0])

        # Sample x_T
        x_t = self.q_sample(timesteps[0], torch.zeros_like(y), y)

        # Predict x0 via recursive reverse process
        for t in timesteps:
            x0_r = torch.zeros_like(x_t)
            for _ in range(self.n_recursions):
                x0_rp1 = generator(torch.cat((x_t, y), axis=1), t, x_r=x0_r)

                # Change in l1-norm
                change = torch.abs(x0_rp1 - x0_r).mean(axis=0).max()
                if change < self.consistency_threshold:
                    break

                x0_r = x0_rp1

            x0_pred = x0_r 
            x_tm1_pred = self.q_posterior(t, x_t, x0_pred, y)
            x_t = x_tm1_pred

        return x0_pred
    
    def _get_betas(self):
        betas_len = self.n_steps + 1
        betas = np.linspace(self.beta_start**0.5, self.beta_end**0.5, betas_len)**2
        
        # Discretization correction
        betas = np.append(0., betas).astype(np.float32)
        
        # Handle odd number of betas
        if betas_len % 2 == 1:
            betas = np.concatenate([
                betas[:betas_len//2],
                [betas[betas_len//2]],
                np.flip(betas[:betas_len//2])
            ])
        
        else:
            betas = np.concatenate([
                betas[:betas_len//2],
                np.flip(betas[:betas_len//2])
            ])

        return betas

    @staticmethod
    def gaussian_product(sigma1, sigma2):
        denom = sigma1**2 + sigma2**2
        mu1 = sigma2**2 / denom
        mu2 = sigma1**2 / denom
        var = (sigma1**2 * sigma2**2) / denom
        return mu1, mu2, var
    
    def vis_scheduler(self):
        plt.figure(figsize=(6, 3))
        plt.plot(self.std**2, label=r'$\sigma_t^2$', color='#3467eb')
        plt.plot(self.mu_x0, label=r'$\mu_{x_0}$', color='#6cd4a2')
        plt.plot(self.mu_y, label=r'$\mu_{y}$', color='#d46c7d')

        plt.legend()
        plt.show()
