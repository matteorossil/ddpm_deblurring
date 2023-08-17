## Perceptual and Distortion Metrics

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

# return a numpy array of psnr values (batched)
def psnr(sharp, deblurred):

    sharp = sharp.mul(255).add_(0.5).clamp_(0, 255).to("cpu")
    deblurred = deblurred.mul(255).add_(0.5).clamp_(0, 255).to("cpu")

    psnr = PeakSignalNoiseRatio(data_range=255.0)

    return psnr(deblurred, sharp).item()

def ssim(sharp, deblurred):

    sharp = sharp.mul(255).add_(0.5).clamp_(0, 255).to("cpu")
    deblurred = deblurred.mul(255).add_(0.5).clamp_(0, 255).to("cpu")

    ssim = StructuralSimilarityIndexMeasure(data_range=255.0)

    return ssim(deblurred, sharp).item()

from skimage.metrics import peak_signal_noise_ratio
import numpy as np

# this is with skiimage (from Mimo-Unet-Plus)
def psnr_(sharp, deblurred):

    pred_clip = torch.clamp(deblurred, 0, 1)

    pred_numpy = pred_clip.squeeze(0).cpu().numpy()
    label_numpy = sharp.squeeze(0).cpu().numpy()

    psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)

    return psnr

