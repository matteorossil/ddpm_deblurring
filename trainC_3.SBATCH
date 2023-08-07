#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=trainC3
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

module purge

singularity exec --nv \
	    --overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python /scratch/mr6744/pytorch/ddpm_deblurring/trainer_conditioned_3.py
		--wandb --port 51 --batch_size 16 --sample_size 16 --d_lr 1e-4 --g_lr 1e-4 --threshold 0.02 --l2_loss 0.
		--ckpt_path 08032023_032544 --ckpt_step 380000 --dataset_t gopro_small_multi --dataset_v gopro_small_multi
		--num_workers 8 --multiplier 3_200 --sampling_interval 10_000 --random_seed 7 --name conditioned-single-multi"