python scripts/train/train_diffusion.py hydra/job_logging=none hydra/hydra_logging=none \
                task=$1 \
                exp_name=$2 \
                output_dir=logs/
 