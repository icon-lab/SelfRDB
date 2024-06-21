import os
import warnings
import h5py
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def save_image_pair(x0, x0_pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_image = min(4, x0.shape[0])
    fig, axes = plt.subplots(nrows=2, ncols=n_image, figsize=(n_image*2, 4))

    if n_image == 1:
        axes = axes[..., None]

    for i in range(n_image):
        axes[0, i].imshow(x0[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(x0_pred[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout(pad=0.1)
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()


def save_preds(preds, path):
    # preds = np.concatenate(preds).squeeze()
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)

    # Normalize predictions
    preds = ((preds + 1) / 2).clip(0, 1)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, preds)


def to_norm(x):
    x = x/2
    x = x + 0.5
    return x.clip(0, 1)

def norm_01(x):
    return (x - x.min(axis=(-1,-2), keepdims=True))/(x.max(axis=(-1,-2), keepdims=True) - x.min(axis=(-1,-2), keepdims=True))


def mean_norm(x):
    x = np.abs(x)
    return x/x.mean(axis=(-1,-2), keepdims=True)


def apply_mask_and_norm(x, mask, norm_func):
    x = x*mask
    x = norm_func(x)
    return x


def compute_metrics(
    gt_images,
    pred_images, 
    masks=None,
    subjects_info=None,
    norm='mean'
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Compute psnr and ssim
        psnr_values = []
        ssim_values = []

        # Normalize function
        if norm == 'mean':
            norm_func = mean_norm
        elif norm == '01':
            norm_func = norm_01

        # Convert to numpy array
        if torch.is_tensor(gt_images):
            gt_images = gt_images.cpu().numpy().squeeze()
            pred_images = pred_images.cpu().numpy().squeeze()

        # If images bewteen [-1, 1], scale to [0, 1]
        if np.nanmin(gt_images) < -0.1:
            gt_images = (gt_images + 1) / 2
            gt_images = gt_images.clip(0, 1)

        if np.nanmin(pred_images) < -0.1:
            pred_images = (pred_images + 1) / 2
            pred_images = pred_images.clip(0, 1)

        # Apply mask and normalize
        if masks is not None:
            gt_images = apply_mask_and_norm(gt_images, masks, norm_func)
            pred_images = apply_mask_and_norm(pred_images, masks, norm_func)
        else:
            gt_images = norm_func(gt_images)
            pred_images = norm_func(pred_images)

        # Compute psnr and ssim
        for gt, pred in zip(gt_images, pred_images):
            psnr_value = psnr(gt, pred, data_range=gt.max())
            psnr_values.append(psnr_value)

            ssim_value = ssim(gt, pred, data_range=gt.max())*100
            ssim_values.append(ssim_value)

        # Convert list to numpy array
        psnr_values = np.asarray(psnr_values)
        ssim_values = np.asarray(ssim_values)

        res = {
            'psnrs': psnr_values,
            'ssims': ssim_values,
            'psnr_mean': np.nanmean(psnr_values),
            'ssim_mean': np.nanmean(ssim_values),
        }

        return res
