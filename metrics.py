## Perceptual and Distortion Metrics

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import torch

# return a numpy array of psnr values (batched)
def psnr(sharp, blurred):

    sharp = sharp.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to("cpu", torch.uint8).numpy()
    blurred = blurred.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to("cpu", torch.uint8).numpy()

    psnr = tf.image.psnr(sharp, blurred, max_val=255.0)

    return psnr.numpy()


def ssim(sharp, blurred):

    sharp = sharp.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to("cpu", torch.uint8).numpy()
    blurred = blurred.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to("cpu", torch.uint8).numpy()

    ssim = tf.image.ssim(sharp, blurred, max_val=255.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)

    return ssim.numpy()

    


