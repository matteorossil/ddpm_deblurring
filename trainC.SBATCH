#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=trainC
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

module purge

singularity exec --nv \
	    --overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python /scratch/mr6744/pytorch/ddpm_deblurring/trainer_conditioned.py
		--wandb --port 50 --batch_size 32 --sample_size 32 --d_lr 1e-4 --g_lr 1e-4 --threshold 0.02 --l2_loss 0.
		--ckpt_path 08052023_015846 --ckpt_step 1780000 --dataset_t gopro_small --dataset_v gopro_small
		--num_workers 8 --multiplier 10_000 --sampling_interval 10_000 --random_seed 7 --name conditioned-small"