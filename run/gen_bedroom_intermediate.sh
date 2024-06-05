save_dir=$1

python scripts/eval/gen_layout.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${save_dir} \
                task=scene_bedroom \
                task.network.room_mask_condition=true \
                task.evaluator.weight_file=logs/physcene/model_01800 \
                evaluation.generate_result_json=true \
                evaluation.jsonname="bedroom.json" \
                evaluation.overlap_type="rotated_bbox" \
                evaluation.visual=true \
                evaluation.render2img=true \
                evaluation.save_walkable_map=false \
                evaluation.without_floor=false \
                evaluation.gapartnet=false \
                evaluation.render_save_path="result_render/bedroom_w_guide" \
                # task.evaluator.n_synthesized_scenes=100 
                # optimizer=collision 

