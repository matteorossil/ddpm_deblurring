## Perceptual and Distortion Metrics

import tensorflow as tf
import torch

def psnr(sharp, blurred):

    sharp = sharp.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to("cpu", torch.uint8).numpy()
    blurred = blurred.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to("cpu", torch.uint8).numpy()

    psnr = tf.image.psnr(sharp, blurred, max_val=255.0)

    return psnr.numpy()





