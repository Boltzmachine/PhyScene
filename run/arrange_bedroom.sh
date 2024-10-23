exp_dir=$1
save_dir=$2

python scripts/eval/arrange_layout.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${exp_dir} \
                task=scene_bedroom_arrange \
                task.evaluator.weight_file=logs/bedroom_arrange/model_04000 \
                evaluation.generate_result_json=true \
                evaluation.jsonname="bedroom.json" \
                evaluation.overlap_type="mesh" \
                evaluation.visual=true \
                evaluation.render2img=true \
                evaluation.save_walkable_map=false \
                evaluation.without_floor=false \
                evaluation.gapartnet=false \
                evaluation.render_save_path=${save_dir} \
                task.evaluator.n_synthesized_scenes=10 \
                task.guidance.scale=0.00 \
                optimizer=collision 

