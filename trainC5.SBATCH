#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=if_gp_w/o_l2
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

module purge

singularity exec --nv \
	    --overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python /scratch/mr6744/pytorch/ddpm_deblurring/trainer_conditioned_pretrained_init.py \
		--port 53 --batch_size 64 --sample_size 32 --d_lr 1e-4 --g_lr 1e-4 --threshold 0.02 --l2_loss 0. \
		--dataset_t gopro --dataset_v gopro --num_workers 8 --multiplier 175 --sampling_interval 46_000 \
		--random_seed 7 --name gopro_w/o_l2_pretrained_init_frozen --wandb --hpc --sample --crop_eval --freeze_init \
		--ckpt_metrics --ckpt_step 621000 --ckpt_path 08242023_053616"

### --port, type=str, default='50'                   --> DDP
### --batch_size, type=int, default=32               --> How many samples in a batch, each GPU takes batch_size/num_gpus
### --sample_size, type=int, default=32              --> How many samples to generate
### --d_lr, type=float, default=1e-4                 --> Learning Rate for Denoiser
### --g_lr, type=float, default=1e-4				 --> Learning rate for Initial Predictor
### --threshold, type=float, default=0.02			 --> Regularizer threshold
### --l2_loss, type=float, default=0.				 --> Hyperparam for L2 loss of Initial Predictor
### --dataset_t, type=str, default="gopro"			 --> Dataset for Training
### --dataset_v, type=str, default="gopro_128"		 --> Dataset for Validation
### --ckpt_step, type=int, default=0				 --> Training step of checkpoint to use
### --ckpt_path, type=str, default=""				 --> Directory  where checkpoints are saved
### --num_workers, type=int, default=8				 --> Workers used in Dataloader
### --multiplier, type=int, default=1				 --> Used to create virtual dataset
### --sampling_interval, type=int, default=10_000	 --> How often to sample
### --random_seed, type=int, default=7				 --> Seed for validation set generation
### --name, type=str, default="conditioned"			 --> WANDB name for run
### --wandb, action="store_true"					 --> Use it to show training on WANDB
### --hpc, action="store_true"						 --> Use it if training on HPC
### --sample, action="store_true"					 --> Use it if you want to sample during training

### --ckpt_metrics, action="store_true"				 --> Use it if you want to use metrics computed during previous checkpoint
### --crop_eval, action="store_true"				 --> Whether to perform crops on the evaluation sets
### --freeze_init, action="store_true"				 --> Whether to perform crops on the evaluation sets

### FORMULA: FLOOR(DATSET_SIZE * MULTIPLIER / BATCH_SIZE) = K * SAMPLING_INTERVAL