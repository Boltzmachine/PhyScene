save_dir=$1

python scripts/train/train_diffusion.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=physcene \
                output_dir=logs/
 