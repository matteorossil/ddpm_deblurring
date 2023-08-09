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