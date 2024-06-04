# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used to train a ATISS."""
import argparse
import logging
import os
import sys
import hydra

import numpy as np
import torch
from torch.utils.data import DataLoader
import subprocess
import json

sys.path.insert(0,sys.path[0]+"/../../")

from utils.utils_preprocess import floor_plan_from_scene, room_outer_box_from_scene
from datasets.base import filter_function, get_dataset_raw_and_encoded, get_encoded_dataset
from models.networks import build_network, optimizer_factory, schedule_factory, adjust_learning_rate
from models.stats_logger import StatsLogger, WandB


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    git_dir = os.path.dirname(os.path.realpath(__file__))
    git_head_hash = "foo"
    try:
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
    except subprocess.CalledProcessError:
        # Keep the current working directory to move back in a bit
        cwd = os.getcwd()
        os.chdir(git_dir)
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
        os.chdir(cwd)
    params["git-commit"] = str(git_head_hash)
    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg):
    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(42)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # Create an experiment directory using the experiment_tag
    if cfg.exp_name is None:
        raise
    else:
        experiment_tag = cfg.exp_name

    experiment_directory = os.path.join(
        cfg.output_dir,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Save the parameters of this run to a file
    save_experiment_params(cfg, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_directory))
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR
    # Parse the config file
    config = {
        "data": {**cfg.task.dataset, **cfg.dataset},
        "training": cfg.task.train,
        "validation": cfg.task.test,
    }
    
    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"])
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],
        translations=train_dataset.bounds["translations"],
        angles=train_dataset.bounds["angles"],
        #add objfeats
        objfeats=train_dataset.bounds["objfeats_32"],
    )
    print("Saved the dataset bounds in {}".format(path_to_bounds))

    validation_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=config["training"]["num_workers"],
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} training scenes with {} object types".format(
        len(train_dataset), train_dataset.n_object_types)
    )
    print("Training set has {} bounds".format(train_dataset.bounds))

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=config["validation"]["num_workers"],
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )
    print("Loaded {} validation scenes with {} object types".format(
        len(validation_dataset), validation_dataset.n_object_types)
    )
    print("Validation set has {} bounds".format(validation_dataset.bounds))

    # Make sure that the train_dataset and the validation_dataset have the same
    # number of object categories
    assert train_dataset.object_types == validation_dataset.object_types

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        train_dataset.feature_size, train_dataset.n_classes,
        cfg, None, device=device
    )
    n_all_params = int(sum([np.prod(p.size()) for p in network.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, network.parameters())]))
    print(f"Number of parameters in {network.__class__.__name__}:  {n_trainable_params} / {n_all_params}")

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], filter(lambda p: p.requires_grad, network.parameters()) ) 
    # optimizer = optimizer_factory(config["training"], network.parameters() )

    # Load the checkpoints if they exist in the experiment directory
    # load_checkpoints(network, optimizer, experiment_directory, args, device)
    # Load the learning rate scheduler 
    lr_scheduler = schedule_factory(config["training"])

    WandB.instance().init(
        config,
        model=network,
        project=cfg.logger.get(
            "project", "autoregressive_transformer"
        ),
        name=experiment_tag,
        watch=False,
        log_frequency=10
    )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "w"
    ))

    epochs = config["training"].get("num_epochs", 200)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    # Do the training
    for i in range(0, epochs):
        # adjust learning rate
        adjust_learning_rate(lr_scheduler, optimizer, i)

        network.train()
        #for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
        for b, sample in enumerate(train_loader):
            # Move everything to device
            for k, v in sample.items():
                if not isinstance(v, list):
                    sample[k] = v.to(device)
            batch_loss = train_on_batch(network, optimizer, sample, cfg)
            StatsLogger.instance().print_progress(i+1, b+1, batch_loss)

        if (i % save_every) == 0:
            save_checkpoints(
                i,
                network,
                optimizer,
                experiment_directory,
            )
        StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    if not isinstance(v, list):
                        sample[k] = v.to(device)
                batch_loss = validate_on_batch(network, sample, config)
                StatsLogger.instance().print_progress(-1, b+1, batch_loss)
            StatsLogger.instance().clear()
            print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main()
