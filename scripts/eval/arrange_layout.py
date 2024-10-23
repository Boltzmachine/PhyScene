"""Script to perform scene completion and rearrangement."""
import argparse
import logging
import os
import sys
from scipy.ndimage import binary_dilation
import cv2
import math
import multiprocess
import numpy as np
import torch
import json
sys.path.insert(0, os.path.dirname(__file__) +"/../../")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..', '..', '..'))
from src.reward_model import RewardTrainingModule

from utils.utils_preprocess import floor_plan_from_scene, room_outer_box_from_scene

from datasets.base import filter_function, get_dataset_raw_and_encoded
from datasets.threed_front import ThreedFront
from datasets.threed_future_dataset import ThreedFutureDataset

from models.networks import build_network

from simple_3dviz import Scene
import pickle
from datasets.gapartnet_dataset import GAPartNetDataset
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger

import numpy as np
from models.networks import build_network, optimizer_factory, schedule_factory, adjust_learning_rate
from datasets.base import filter_function, get_dataset_raw_and_encoded

import hydra 
from omegaconf import DictConfig, OmegaConf
# from generate_diffusion import show_scene
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from utils.utils import get_textured_objects, get_textured_object_ids_based_on_objfeats
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
# from simple_3dviz.window import show
from utils.utils_preprocess import render as render_top2down
from pyrr import Matrix44
from simple_3dviz.renderables.mesh import Mesh
from kaolin.ops.mesh import check_sign, index_vertices_by_faces,face_normals
import open3d as o3d
import open3d.visualization.gui as gui
from datasets.gapartnet_dataset import GAPartNetDataset
from utils.utils_preprocess import merge_meshes
import trimesh

from utils.overlap import bbox_overlap, calc_overlap_rotate_bbox, calc_wall_overlap, voxel_grid_from_mesh

from scripts.eval.walkable_metric import cal_walkable_metric
from scripts.eval.walkable_map_visual import walkable_map_visual


from gen_layout import render_scene_all, get_objects, map_scene_id, SaveStates


def complete_boxes(img_t, input_boxes, translation_dim=3, size_dim=3, bbox_dim=3):
    img_t_trans = img_t[:, :, 0:translation_dim]
    img_t_angle = img_t[:, :, translation_dim:] 
    
    input_boxes_trans = input_boxes[:, :, 0:translation_dim]
    input_boxes_size  = input_boxes[:, :, translation_dim:translation_dim+size_dim]  
    input_boxes_angle = input_boxes[:, :, translation_dim+size_dim:bbox_dim] 
    input_boxes_other = input_boxes[:, :, bbox_dim:] 
    img_t = torch.cat([ img_t_trans, input_boxes_size, img_t_angle, input_boxes_other ], dim=-1).contiguous()
    return img_t


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg):
    global config
    config = cfg 
    cfg.task.dataset = {**cfg.task.dataset,**cfg.dataset}
    weight_file = cfg.task.evaluator.weight_file
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    if cfg.evaluation.generate_result_json:
        cfg.evaluation.load_result = False
        cfg.evaluation.save_result = True
        # cfg.evaluation.visual = False
    else:
        cfg.evaluation.load_result = True
        cfg.evaluation.save_result = False
        if "nomask" in cfg.evaluation.jsonname or "no_mask" in cfg.evaluation.jsonname or "diffuscene" in cfg.evaluation.jsonname:
            cfg.evaluation.without_floor=True

    #train+test
    split = ['train', 'val', 'test']
    raw_dataset, ground_truth_scenes = get_dataset_raw_and_encoded(
        config.task["dataset"],
        filter_fn=filter_function(
            config.task["dataset"],
            split=split
        ),
        split=split
    )

    #train+test
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config.task["dataset"],
        filter_fn=filter_function(
            config.task["dataset"],
            split=split
        ),
        split=split
    )

    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    #object dataset
    global objects_dataset
    global gapartnet_dataset
    objects_dataset, gapartnet_dataset = get_objects(cfg)

    if not cfg.evaluation.load_result:
        network, _, _ = build_network(
            dataset.feature_size, dataset.n_classes,
            config, weight_file, dataset, cfg, 
            objects_dataset, gapartnet_dataset, device=device
        )
        network.eval()
    else:
        network=None

    classes = np.array(dataset.class_labels)
    print('class labels:', classes, len(classes))


    if cfg.task.generation.save_intermediate:
        process_func = lambda x, keep_empty, input_boxes: dataset.post_process(network.delete_empty_from_network_samples(complete_boxes(x, input_boxes, translation_dim=network.translation_dim, size_dim=network.size_dim, bbox_dim=network.bbox_dim), device=device, keep_empty=keep_empty))
        save_states_func = SaveStates(cfg.task.generation.save_path, process_func)

    synthesized_scenes = []
    floor_plan_mask_list = []
    floor_plan_centroid_list = []
    batch_size = min(cfg.task.test.batch_size, cfg.task.evaluator.n_synthesized_scenes)
    mapping = map_scene_id(raw_dataset)

    idx = 0
    predictions = []
    total = min(cfg.task.evaluator.n_synthesized_scenes, len(dataset))
    total_batch = (total - 1) // batch_size + 1
    for i in range(total_batch):
        print("{} / {}:".format(
            i, total_batch)
        )
        # Get a floor plan
        room_lst = []
        floor_plan_lst = []
        floor_plan_mask_batch_list = []
        floor_plan_centroid_batch_list = []
        scene_idx_lst = []
        scene_id_lst = []
        scene_lst = []
        room_outer_box_lst = []
        tr_floor_lst = []
        input_boxes_lst = []
        for j in range(min(batch_size, total - i*batch_size)):
            scene_idx = i*batch_size+j
            if scene_idx >= len(dataset):
                break
            print(scene_idx)
            # scene_idx = 525  #50 #525  #1610   #3454 #3921
            print(j,scene_idx)
            scene_idx_lst.append(scene_idx)
            samples = dataset[scene_idx]
            # print("scene id ",scene_idx)

            current_scene = raw_dataset[scene_idx]
            scene_id_lst.append(current_scene.scene_id)
            scene_lst.append(current_scene)
        # Get a floor plan
            floor_plan, tr_floor, room_mask = floor_plan_from_scene(
                current_scene, cfg.path_to_floor_plan_textures, no_texture=False
            )
            room_outer_box = room_outer_box_from_scene(current_scene)
            room_outer_box = torch.Tensor(room_outer_box[None,:,:]).to(device)
            
            room_outer_box_lst.append(room_outer_box)

            room_lst.append(room_mask)
            floor_plan_lst.append(floor_plan)
            floor_plan_mask_list.append(current_scene.floor_plan)
            floor_plan_mask_batch_list.append(current_scene.floor_plan)
            floor_plan_centroid_list.append(current_scene.floor_plan_centroid)
            floor_plan_centroid_batch_list.append(current_scene.floor_plan_centroid)
            tr_floor_lst.append(tr_floor)

            if cfg.task.network.room_arrange_condition:
                num_boxes, angle_dim = samples["angles"].shape
                print( 'number of boxes are {:d}'.format(num_boxes) )

                if cfg.task.network.get("objectness_dim", 0) >0:
                    num_objects = int((samples["objectness"]>0).sum())
                else:
                    num_objects = num_boxes - int((samples["class_labels"][:, -1]>0).sum())
                print('number of nonempty objects {}'.format(num_objects))
                num_partial = num_boxes
                render_foldername = "noisy"
            else:
                raise
                num_partial = args.num_partial
                print('number of objects in partial scenes {}'.format(num_partial))
                render_foldername = "partial"
            bbox_params = {
                "class_labels": torch.from_numpy(samples["class_labels"])[None, :num_partial, ...],
                "translations": torch.from_numpy(samples["translations"])[None, :num_partial, ...],
                "sizes": torch.from_numpy(samples["sizes"])[None, :num_partial, ...],
                "angles": torch.from_numpy(samples["angles"])[None, :num_partial, ...],
            }
            input_boxes = torch.cat([
                    bbox_params["translations"],
                    bbox_params["sizes"],
                    bbox_params["angles"],
                    bbox_params["class_labels"],
                ], dim=-1
            )
            if cfg.task.network.get("objectness_dim", 0) >0:
                bbox_params["objectness"] = torch.from_numpy(samples["objectness"])[None, :num_partial, ...]
                input_boxes = torch.cat([ input_boxes, bbox_params["objectness"] ], dim=-1)

            if cfg.task.network.get("objfeat_dim", 0) >0:
                if cfg.task.network["objfeat_dim"] == 32:
                    bbox_params["objfeats"] = torch.from_numpy(samples["objfeats_32"])[None, :num_partial, ...]
                else:
                    bbox_params["objfeats"] = torch.from_numpy(samples["objfeats"])[None, :num_partial, ...]
                input_boxes = torch.cat([ input_boxes, bbox_params["objfeats"] ], dim=-1)
            
            input_boxes_lst.append(input_boxes)


        if len(scene_idx_lst) == 0:
            continue
        room_mask = torch.concat(room_lst)
        room_outer_box = torch.concat(room_outer_box_lst)
        input_boxes = torch.concat(input_boxes_lst)

        bbox_params = network.arrange_scene(
                room_mask=room_mask.to(device),
                input_boxes=input_boxes.to(device),
                room_outer_box=room_outer_box,
                floor_plan=floor_plan_mask_batch_list, 
                floor_plan_centroid=floor_plan_centroid_batch_list,
                batch_size = len(scene_idx_lst),
                num_points=cfg.task["network"]["sample_num_points"],
                point_dim=cfg.task["network"]["point_dim"],
                text=None,
                device=device,
                clip_denoised=cfg.clip_denoised,
                batch_seeds=torch.arange(i, i+1),
                keep_empty = True,
                scene_id_lst = scene_id_lst,
                scene_lst = scene_lst,
                save_states_func=save_states_func if cfg.task.generation.save_intermediate else None,
                # reward_guidance=RewardGuidance(cfg)
        )
        
        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).cpu().numpy()

        render_scene_all(boxes,objects_dataset,dataset,gapartnet_dataset,cfg,room_lst,floor_plan_lst,floor_plan_mask_batch_list,floor_plan_centroid_batch_list,
                        tr_floor_lst,room_outer_box_lst,scene_id_lst,scene_idx_lst, predictions=predictions,raw_dataset=raw_dataset)

        idx += batch_size



if __name__ == "__main__":
    main()