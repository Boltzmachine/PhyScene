import os
import sys
sys.path.insert(0, os.path.dirname(__file__) +"/../")
sys.path.append('/ext/qiuweikang/LayoutPPT/')
from src.layout import Layout
import argparse
import numpy as np
from tqdm import tqdm
import json
import hydra
import torch
# from third_party.DiffuScene.scripts.training_utils import load_config
# from third_party.DiffuScene.scripts.utils import floor_plan_from_scene, export_scene, get_textured_objects_in_scene

# from third_party.scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
# from third_party.scene_synthesis.datasets.threed_front import ThreedFront
# from third_party.scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
# from third_party.scene_synthesis.networks import build_network
# # from third_party.scene_synthesis.utils import get_textured_objects, get_textured_objects_based_on_objfeats, get_textured_object_ids_based_on_objfeats
from utils.utils import get_textured_object_ids_based_on_objfeats
from datasets.base import filter_function, get_dataset_raw_and_encoded
from datasets.threed_future_dataset import ThreedFutureDataset
import functools

from tqdm import tqdm
from renderer.blender import BlenderRenderer

from glob import glob
import re
import omegaconf

scene_id_to_idx = None
raw_dataset = None
objects_dataset = None
classes = None
config = None

def match_for_each_scene(box_params_path):
    save_path = os.path.join(os.path.dirname(box_params_path), 'scene.json')
    # if os.path.exists(save_path):
    #     return
    scene_id = re.search(r"/([^/]*[rR]oom-\d+)/", box_params_path).group(1)
    idx = scene_id_to_idx[scene_id]
    scene = raw_dataset[idx]
    assert scene.scene_id == scene_id
    box_params = np.load(box_params_path, allow_pickle=True).item()

    bbox_params_t = torch.cat([
        box_params["class_labels"],
        box_params["translations"], # 3
        box_params["sizes"], # 3
        box_params["angles"], # 1
        box_params["objfeats_32"]
    ], dim=-1).cpu().numpy()[None]

    models = get_textured_object_ids_based_on_objfeats(
        bbox_params_t, objects_dataset, classes, config
    )
    prediction = {
            "query_id": scene.uid,
            "object_list": models,
        }
    json.dump(prediction, open(save_path, 'w'))


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg):
    cfg.task.dataset = {**cfg.task.dataset,**cfg.dataset}
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR

    global config
    config = cfg

    global scene_id_to_idx
    global raw_dataset

    #train+test
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        cfg.task["dataset"],
        filter_fn=filter_function(
            cfg.task["dataset"],
            split=["train", "val","test"]
        ),
        split=["train", "val","test"]
    )

    global objects_dataset
    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        'data/pickled_data/threed_future_model_bedroom.pkl'
    )

    global classes
    classes = dataset.class_labels
    scene_id_to_idx = {scene.scene_id: idx for idx, scene in enumerate(raw_dataset)}


    # parser = argparse.ArgumentParser()
    # parser.add_argument("dir", type=str)
    # parser.add_argument("--num_workers", type=int, default=1)
    # parser.add_argument("--sample", action='store_true')
    # args = parser.parse_args()
    args = cfg
    renderer = BlenderRenderer()
    
    scene_dirs = os.listdir(args.dir)
    
    all_paths = []
    for scene_dir in scene_dirs:
        step_dirs = os.listdir(os.path.join(args.dir, scene_dir))
        for step_dir in step_dirs:
        # step_dir = 'step0000'
            # if step_dir != 'step0000':
            #     continue
            # for cand_dir in os.listdir(os.path.join(args.dir, scene_dir, step_dir)):
            box_params_path = os.path.join(args.dir, scene_dir, step_dir, 'box_params.npy')
            all_paths.append(box_params_path)


    if args.num_workers > 1:
        import multiprocessing
        with multiprocessing.Pool(args.num_workers) as p:
            list(tqdm(p.imap(match_for_each_scene, all_paths), total=len(all_paths)))
    else:
        for box_params_path in tqdm(all_paths):
            match_for_each_scene(box_params_path)
    
    # for path in tqdm(all_paths):
    #     path = os.path.join(os.path.dirname(path), 'scene.json')
    #     print(path)
    #     try:
    #         renderer.render(json.load(open(path)), do_denormalize=False, file_name=os.path.join(os.path.dirname(path), 'render.png'))
    #     except Exception as e:
    #         print(e)
    #         print('Failed to render', os.path.dirname(box_params_path))
        
if __name__ == "__main__":
    main()
