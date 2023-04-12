import math
import json
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import wandb
from tqdm import tqdm
import copy
import torch.nn as nn
import torchvision
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from collections.abc import Iterable
from dotenv import load_dotenv

import dataset
import utils
import evaluation
import training


def rcnn_train_wrapper(train_manuscripts, test_manuscripts, model_savepath=None, results_savepath=None):
    data_path = "./data/images/downsampled"
    annot_json_path = "./data/images/downsampled/annot.json"

    with open("./final_rcnn_config.json", "r") as f:
        config = json.load(f)
    
    load_dotenv()
    utils.no_randomness()

    transform_kwargs = {
        "spatial_augm": True, 
        "color_augmentation": True,
        "crop_scale_params": (0.75, 1),
        "flipping": False,
        "color_jitter_kwargs": {
            "brightness": 0.25,
            "contrast": 0.25,
            "saturation": 0.5,
            "hue": 0.5
        }
    }
    basic_transform = {}

    train_loader, _, _, test_loader, = utils.prepare_data_wrapper(
        data_path, annot_json_path,
        loader_kwargs={ "batch_size": 8, "num_workers": 2 },
        custom_data_augm=transform_kwargs, basic_transformations=basic_transform,
        subsample_eval_datasets=True, oversample_eval_ds=True,
        train_manuscript_names=train_manuscripts,
        valid_manuscript_names=[], test_manuscript_names=test_manuscripts, 
    )

    training.FasterRCNN_Loss_Weight_Modification(
        rpn_class_weights=[1, config['class_annot_weight']],
        detection_class_weights=[1, config['class_annot_weight']]
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    model.to(utils.get_device())
    for params in model.parameters():
        params.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])

    training.CustomScheduler(steps=config["scheduler_steps"])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, training.CustomScheduler.func)

    cycle = training.TrainingCycle(
        wandb=False, delete_old_checkpoints=True, savepath=model_savepath, calculate_coco=False,
    )
    results = cycle.fit(
        model, train_loader, optim, training.training_faster_rcnn_batch, None, max_no_epochs=config["no_epochs"],
        scheduler=lr_scheduler, scheduler_kwargs = {"unit": "epoch"}, 
        validation_loaders={}
    )

    if len(test_manuscripts) > 0:
        eval = evaluation.ComputeClassificationMetrics(verbose=True)
        results = eval.evaluate(model, test_loader)
        print(results)

        if results_savepath is not None:
            with open(results_savepath, "w") as f:
                json.dump(results, f)
        return results


def classifier_train_wrapper(train_manuscripts, test_manuscripts, model_savepath=None, results_savepath=None):
    
    with open(f"./final_classifier_vit_config.json", "r") as f:
        config = json.load(f)

    data_path = "./data/patches"
    annot_json_path = "./data/patches/annot.json"
    mask_path = "./data/patches/distance_masks"

    load_dotenv()
    utils.no_randomness()

    transform_kwargs = {
        "apply_normalization": True,
        "downsample_size": (224, 224),
        "spatial_augm": True, 
        "color_augmentation": True,
        "crop_scale_params": (0.75, 1),
        "flipping": False,
        "color_jitter_kwargs": {
            "brightness": 0.25,
            "contrast": 0.25,
            "saturation": 0.5,
            "hue": 0.5
        }
    }
    basic_transformations = {
        "apply_normalization": True,
        "downsample_size": (224, 224)
    }

    architecture = "vit"
    batch_size = config["batch_size"]

    train_loader, _, _, test_loader = utils.prepare_data_wrapper(
        data_path, annot_json_path, mask_path=mask_path, 
        train_manuscript_names=train_manuscripts,
        valid_manuscript_names=[], test_manuscript_names=test_manuscripts,
        classification_task=True, patches=True,
        custom_data_augm=transform_kwargs, basic_transformations=basic_transformations,
        new_mask=True, out_of_page_mask_value=0,
        subsample_eval_datasets=True, oversample_eval_ds=True, max_oversampling_coefficient=1,
        undersample_no_annotations=True,
        loader_kwargs={ 
            "batch_size": batch_size,
            "num_workers": 2
        }
    )

    if architecture == "resnet":
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = training.extend_resnet_to_another_channel(model)
    elif architecture == "vit":
        model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.conv_proj = torch.nn.Sequential(
            torch.nn.Conv2d(4, 3, kernel_size=1, stride=1, bias=False),
            model.conv_proj
        )
        model.heads.head = torch.nn.Linear(768, 2)
    else:
        raise "Unsupported architecture"

    model.to(utils.get_device())
    for params in model.parameters():
        params.requires_grad = True

    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]
    weight_label1 = config["class_annot_weight"]
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    label_weights = torch.Tensor([1, weight_label1]).to(utils.get_device())
    loss_func = torch.nn.CrossEntropyLoss(weight=label_weights)
    
    training.CustomScheduler(steps=config["scheduler_steps"])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, training.CustomScheduler.func)

    cycle = training.TrainingCycle(
        task="classification", wandb=False, delete_old_checkpoints=True, savepath=model_savepath,
    )
    results = cycle.fit(
        model, train_loader, optim, training.training_classifier, loss_func, max_no_epochs=config["no_epochs"],
        scheduler=lr_scheduler, scheduler_kwargs = {"unit": "epoch"}, 
        validation_loaders={}
    )

    if len(test_manuscripts) > 0:
        eval = evaluation.ClassificationEval(verbose=True)
        results = eval.evaluate(model, test_loader)
        print(results)

        if results_savepath is not None:
            with open(results_savepath, "w") as f:
                json.dump(results, f)
        return results
