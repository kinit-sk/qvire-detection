from torch.nn import functional as F
import torch
import os
import wandb
from tqdm import tqdm
import copy
import torchvision.models.detection.roi_heads as roi_heads_module
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RegionProposalNetwork
from collections.abc import Iterable

import utils
import evaluation


class FasterRCNN_Eval_Loss:
    rpn_loss = []
    pred_loss = []

    def __init__(self, model):
        self.handles = []
        self.assign_hooks(model)

    def assign_hooks(self, model):
        self.handles += [model.rpn.register_forward_hook(FasterRCNN_Eval_Loss.rpn_forward_hook)]
        self.handles += [model.roi_heads.register_forward_hook(FasterRCNN_Eval_Loss.class_forward_hook)]

    def compute_loss(self):
        accum_loss = 0
        for rpn, pred in zip(FasterRCNN_Eval_Loss.rpn_loss, 
                             FasterRCNN_Eval_Loss.pred_loss):
            all_losses = torch.stack(list({**rpn, **pred}.values()))
            
            accum_loss += all_losses.sum()
        
        accum_loss /= len(FasterRCNN_Eval_Loss.rpn_loss)
        self.cleanup()
        return accum_loss.item()

    def cleanup(self):
        for h in self.handles:
            h.remove()
        
        self.handles = []
        FasterRCNN_Eval_Loss.rpn_loss = []
        FasterRCNN_Eval_Loss.pred_loss = []

    @staticmethod
    def rpn_forward_hook(model, inp, out):
        images, features, targets = inp
        
        features = list(features.values())
        objectness, pred_bbox_deltas = model.head(features)
        anchors = model.anchor_generator(images, features)
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        losses = {}
        labels, matched_gt_boxes = model.assign_targets_to_anchors(anchors, targets)
        regression_targets = model.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = model.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {"loss_objectness": loss_objectness.detach(), "loss_rpn_box_reg": loss_rpn_box_reg.detach()}

        FasterRCNN_Eval_Loss.rpn_loss.append(losses)
        return out

    @staticmethod
    def class_forward_hook(model, inp, out):
        features, proposals, image_shapes, targets = inp
        
        proposals, matched_idxs, labels, regression_targets = model.select_training_samples(proposals, targets)

        box_features = model.box_roi_pool(features, proposals, image_shapes)
        box_features = model.box_head(box_features)
        class_logits, box_regression = model.box_predictor(box_features)

        losses = {}
        loss_classifier, loss_box_reg = roi_heads_module.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        
        FasterRCNN_Eval_Loss.pred_loss.append(losses)
        return out
    

class TrainingCycle:
    def __init__(self, name="model", savepath=None, save_metric=None, verbose_lvl=1, wandb=False,
                 save_model_after_N_evals=1, delete_old_checkpoints=False, calculate_coco=False,
                 qualitative_eval=False, qualitative_savepath=None, wandb_qualitative_eval=True,
                 evaluate_each_N_iterations=10000, task="detection", calculate_eval_loss=False):
        
        self.name = name
        self.savepath = savepath
        self.task = task
    
        self.save_metric = save_metric
        if save_metric is None:
            self.save_metric = "eval_metric"

        self.verbose_lvl = verbose_lvl
        self.wandb = wandb
        self.delete_old_checkpoints = delete_old_checkpoints
        self.save_model_after_N_evals = save_model_after_N_evals
        self.evaluate_each_N_iterations = evaluate_each_N_iterations

        self.calculate_coco = calculate_coco
        self.calculate_eval_loss = calculate_eval_loss
        self.history_metrics = {}
        self.best_model_yet = None
        self.train_loss = 0

        self.qualitative_eval = qualitative_eval
        self.qualitative_savepath = qualitative_savepath
        self.wandb_qualitative_eval = wandb_qualitative_eval
        
        if self.savepath is not None:
            os.makedirs(self.savepath, exist_ok=True)

    def evaluate(self, model, validation_loaders, loss_func=None):
        was_training = model.training
        model.eval()

        if validation_loaders is None or len(validation_loaders.keys()) == 0:
            if "train" not in self.history_metrics.keys():
                self.history_metrics["train"] = []
            self.history_metrics["train"].append({ "train_loss": self.train_loss })

            if was_training:
                model.train()
            return

        if "train_loss" not in self.history_metrics.keys():
            self.history_metrics["train_loss"] = []
        self.history_metrics["train_loss"].append(self.train_loss)

        all_figures = {}
        for i, k in enumerate(list(validation_loaders.keys())):
            all_figures[k] = []
            if self.verbose_lvl > 1:
                print(f"{i+1} | dataset '{k}'...")

            loader = validation_loaders[k]
            if k not in self.history_metrics.keys():
                self.history_metrics[k] = []

            results = {}

            if self.task == "detection":
                if self.calculate_coco:
                    eval = evaluation.CocoEvaluate(loader.dataset.annot_json_path, 
                                    verbose=self.verbose_lvl > 1, epoch_num=len(self.history_metrics[k]),
                                    qualitative_evaluation=self.qualitative_eval,
                                    qualitative_savepath=self.qualitative_savepath)
                    results = eval.evaluate(model, loader)
                    if self.qualitative_eval and self.qualitative_savepath is None:
                        results, all_figures[k] = results
                
                if self.calculate_eval_loss:
                    hook_wrapper = FasterRCNN_Eval_Loss(model)
                
                eval_f1 = evaluation.ComputeClassificationMetrics(
                    count_miss_prediction_to_fn=True,
                    verbose=self.verbose_lvl > 1
                )
                classification_metrics = eval_f1.evaluate(model, loader)
                results.update(classification_metrics)
                
                if self.calculate_eval_loss:
                    loss = hook_wrapper.compute_loss()
                    results[f"eval_loss"] = loss
                self.history_metrics[k].append(results)

            elif self.task == "classification":
                eval = evaluation.ClassificationEval(verbose=self.verbose_lvl > 1)
                results = eval.evaluate(model, loader, loss_func=loss_func)
                self.history_metrics[k].append(results)
            else:
                raise "Unsupported task"

            if k != "train":
                self.history_metrics[k][-1]["eval_metric"] = (
                        self.history_metrics[k][-1]["f1"] -
                        (self.history_metrics[k][-1]["f1"] -
                        self.history_metrics["train"][-1]["f1"])**2
                    )
            if self.verbose_lvl > 0:
                print(evaluation.results_to_string(results, k.upper()))

        if self.wandb:
            logs = {}
            logs["train_loss"] = self.train_loss
            for k in validation_loaders.keys():
                logs[k] = self.history_metrics[k][-1]

                if self.wandb_qualitative_eval and len(all_figures[k]) > 0:
                    for it, f in enumerate(all_figures[k]):
                        logs[f"{k}.im_{it}"] = f

            wandb.log(logs)

        if was_training:
            model.train()
        
    def gradual_save(self, model):
        if self.savepath is None:
            return

        no_evaluations = len(self.history_metrics["train"])
        recent_name = f"{self.name}_LAST_"
        if no_evaluations % self.save_model_after_N_evals != 0:
            return

        if self.delete_old_checkpoints:
            for filename in os.listdir(self.savepath):
                if recent_name in filename:
                    os.remove(os.path.join(self.savepath, filename))
                    break

        if "valid" in list(self.history_metrics.keys()):
            if self.best_model_yet is not None:
                best_metric = self.best_model_yet[self.save_metric]
                last_metric = self.history_metrics["valid"][-1][self.save_metric]

            if self.best_model_yet is None or best_metric < last_metric:
                best_name = f"{self.name}_BEST_"

                for filename in os.listdir(self.savepath):
                    if best_name in filename:
                        os.remove(os.path.join(self.savepath, filename))
                        break

                best_name = f"{self.name}_BEST_{no_evaluations}.bin"
                self.best_model_yet = self.history_metrics["valid"][-1]
                
                torch.save(model.state_dict(), os.path.join(self.savepath, best_name))

        recent_name = f"{self.name}_LAST_{no_evaluations}.bin"
        torch.save(model.state_dict(), os.path.join(
            self.savepath, recent_name))

    def fit(self, model, data_loader, optimizer, batch_step_func, loss_func=None, 
            max_no_epochs=5, max_no_iterations=100_000,
            validation_loaders={}, scheduler=None, 
            scheduler_kwargs={ "unit": "iteration" }):

        was_eval = model.training == False
        model.train()
        iteration_accumulation = 0

        for e in range(max_no_epochs):
            if self.verbose_lvl > 0:
                print(f"EPOCH {e+1}/{max_no_epochs}")
                if self.verbose_lvl > 1:
                    print("Training...")

            iteration_it = 0
            it_for_loss = 0
            self.train_loss = 0

            for batch in tqdm(data_loader, disable=self.verbose_lvl == 0):
                loss = batch_step_func(model, batch, loss_func, optimizer)
                self.make_scheduler_step(scheduler, scheduler_kwargs, "iteration")

                self.train_loss += loss.item()
                iteration_it += 1  
                it_for_loss += 1

                # if evaluate_each_N_iterations == -1, then we dont evaluate results during epoch
                if iteration_it != 0 and iteration_it != len(data_loader) and self.evaluate_each_N_iterations != -1:
                    if iteration_it % self.evaluate_each_N_iterations == 0:
                        self.train_loss /= it_for_loss
                        self.evaluate(model, validation_loaders, loss_func)
                        self.gradual_save(model)

                        self.train_loss = 0
                        it_for_loss = 0

                if iteration_accumulation + iteration_it >= max_no_iterations:
                    return self.history_metrics

            iteration_accumulation += iteration_it
            self.train_loss /= it_for_loss
            data_loader.dataset.reset_epoch()
            
            self.make_scheduler_step(scheduler, scheduler_kwargs, "epoch")
            self.evaluate(model, validation_loaders, loss_func)
            self.gradual_save(model)

        if was_eval:
            model.eval()
        return self.history_metrics

    def make_scheduler_step(self, scheduler, scheduler_kwargs, unit="iteration"):
        current_unit = scheduler_kwargs["unit"] if "unit" in list(
            scheduler_kwargs.keys()) else None
        if scheduler is not None and current_unit == unit:
            kwargs = copy.deepcopy(scheduler_kwargs)
            del kwargs["unit"]

            if isinstance(scheduler, Iterable) == False:
                scheduler = [scheduler]
            for i in range(len(scheduler)):
                scheduler[i].step(**kwargs)


class CustomScheduler:
    steps = []

    def __init__(self, steps=None) -> None:
        if steps is None or len(steps) == 0:
            steps = [
                (3, 1),
                (6, (1, 0.1)),
                (20, 0.1),
                (100, (0.1, 0))
            ]
        CustomScheduler.steps = steps

    @staticmethod
    def func(it):
        for istep, step in enumerate(CustomScheduler.steps):
            if it < step[0]:
                #step - constant
                if isinstance(step[1], Iterable) == False:
                    return step[1]

                #linear slope
                hb, lb = step[1]
                plateau_size = step[0] - CustomScheduler.steps[istep - 1][0]
                it -= CustomScheduler.steps[istep - 1][0]
                return hb - (hb - lb) * (it / plateau_size)
        return 0


#batch_step_functions below
def training_faster_rcnn_batch(model, batch, loss_func, optimizer):
    X, targets = utils.prepare_batch(batch)
    losses = model(X, targets)

    if loss_func is None:
        final_loss = sum(loss for loss in losses.values())
    else:
        final_loss = loss_func(losses)

    optimizer.zero_grad()
    final_loss.backward()
    optimizer.step()
    
    return final_loss


def training_detr_batch(model, batch, loss_func, optimizer):
    encoding = utils.detr_batch_to_device(batch)
    outputs = model(**encoding)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def training_classifier(model, batch, loss_func, optimizer):
    X, y = utils.prepare_classification_batch(batch)

    out = model(X)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def extend_faster_rcnn_to_another_channel(model):
    model.transform.image_mean.append(0)
    model.transform.image_std.append(1)
    model.backbone.body.conv1 = torch.nn.Sequential(
        torch.nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False),
        model.backbone.body.conv1
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    
    return model
    

def extend_resnet_to_another_channel(model):
    model.conv1 = torch.nn.Sequential(
        torch.nn.Conv2d(4, 3, kernel_size=1, stride=1, bias=False),
        model.conv1
    )
    model.fc = torch.nn.Linear(2048, 2)
    
    return model


class FasterRCNN_Loss_Weight_Modification:
    rpn_class_weights = None
    detection_class_weights = None
    def __init__(self, rpn_class_weights=[1,100], detection_class_weights=[1,100], device=utils.get_device()) -> None:
        rpn_class_weights = torch.Tensor(rpn_class_weights).to(device=device)
        detection_class_weights = torch.Tensor(detection_class_weights).to(device=device)

        FasterRCNN_Loss_Weight_Modification.rpn_class_weights = rpn_class_weights
        FasterRCNN_Loss_Weight_Modification.detection_class_weights = detection_class_weights
    
        RegionProposalNetwork.compute_loss = overwrite_rpn_compute_loss
        roi_heads_module.fastrcnn_loss = overwrite_fasterrcnn_loss


def overwrite_rpn_compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = (
        F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        )
        / (sampled_inds.numel())
    )
    
    weight_tensor = torch.zeros(len(sampled_inds), device=utils.get_device())
    
    for idx, w in enumerate(FasterRCNN_Loss_Weight_Modification.rpn_class_weights):
        weight_tensor[labels[sampled_inds] == idx] = w

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds], 
        weight=weight_tensor
    )

    return objectness_loss, box_loss


def overwrite_fasterrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(
        class_logits, labels, 
        weight=FasterRCNN_Loss_Weight_Modification.detection_class_weights,
        reduction="sum"
    ) / len(labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss
