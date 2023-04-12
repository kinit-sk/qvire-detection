import matplotlib.pyplot as plt
import json
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
from collections.abc import Iterable
from torchvision.models import ViT_B_16_Weights, ResNet50_Weights
from dotenv import load_dotenv

import dataset
import utils
import evaluation
import training


def eval_detector(test_manuscripts, model_path, savepath=None, qualitative_savepath=None):
    utils.no_randomness()

    data_path = "./data/images/downsampled"
    annot_json_path = "./data/images/downsampled/annot.json"

    _, _, test_idx = utils.divide_dataset(annot_json_path,
        train_manuscript_names=[],
        valid_manuscript_names=[],
        test_manuscript_names=test_manuscripts,
        oversampling_kwargs={
            "oversampling": False,
            "max_oversampling_coefficient": 1,
            "max_eval_oversampling_coefficient": 1,
            "smart_sampling": False
        }
    )

    test_ds = dataset.MyDataset(test_idx, data_path, annot_json_path, transform_kwargs={})
    test_loader = DataLoader(
        test_ds, batch_size=8, num_workers=2, 
        collate_fn=utils.tolist_collate_fn, pin_memory=True
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    model.to(utils.get_device())
    model.load_state_dict(torch.load(model_path))
    model.eval()

    eval = evaluation.ComputeClassificationMetrics(verbose=True, qualitative_savepath=qualitative_savepath, annotation_json_path=annot_json_path)
    results = eval.evaluate(model, test_loader)
    print("DETECTOR RESULTS:", results)

    if savepath is not None:
        with open(savepath, "w") as fp:
            json.dump(results, fp)
    return results


def eval_classifier(test_manuscripts, model_type, model_path, savepath=None, qualitative_savepath=None):
    utils.no_randomness()

    data_path = "./data/patches"
    annot_json_path = "./data/patches/annot.json"
    mask_path = "./data/patches/distance_masks"
        
    _, _, test_idx = utils.divide_dataset(annot_json_path,
        train_manuscript_names=[],
        valid_manuscript_names=[],
        test_manuscript_names=test_manuscripts,
        oversampling_kwargs={
            "oversampling": False,
            "max_oversampling_coefficient": 1,
            "max_eval_oversampling_coefficient": 1,
            "smart_sampling": False
        }
    )
    
    basic_transformations = {
        "apply_normalization": True,
        "downsample_size": (224, 224)
    }
    test_ds = dataset.MyDataset(
        test_idx, data_path, annot_json_path, 
        transform_kwargs=basic_transformations,
        classification_task=True,
        additional_mask_path=mask_path,
        new_distance_mask=True, 
        out_of_page_mask_value=0
    )

    batch_size = 64 if model_type == "resnet" else 32
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=2,
        collate_fn=utils.tolist_collate_fn, pin_memory=True
    )

    if model_type == "resnet":
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = training.extend_resnet_to_another_channel(model)
    elif model_type == "vit":
        model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.conv_proj = torch.nn.Sequential(
            torch.nn.Conv2d(4, 3, kernel_size=1, stride=1, bias=False),
            model.conv_proj
        )
        model.heads.head = torch.nn.Linear(768, 2)
    else:
        raise

    model.to(utils.get_device())
    model.load_state_dict(torch.load(model_path))
    model.eval()

    eval = evaluation.ClassificationEval(verbose=True, qualitative_savepath=qualitative_savepath)
    results = eval.evaluate(model, test_loader)
    print("CLASSIFICATION RESULTS:", results)

    if savepath is not None:
        with open(savepath, "w") as fp:
            json.dump(results, fp) 
    return results

if __name__ == "__main__":
    with open("./cv_manuscripts_division.json") as f:
        test_manuscripts = json.load(f)["fold_3"]["eval"]
    model_path = "DEFINE_PATH_OF_THE_MODEL"

    results_rcnn = eval_detector(test_manuscripts, model_path, savepath=None, qualitative_savepath=None)
    # results_vit = eval_classifier(test_manuscripts, "vit", model_path, savepath=None, qualitative_savepath=None)
