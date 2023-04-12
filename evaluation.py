import shutil
import numpy as np
import torch
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
from collections.abc import Iterable
from torchvision.utils import draw_bounding_boxes

import dataset
import utils


class ComputeClassificationMetrics:
    def __init__(
        self, iou_threshold=0.5, verbose=False, count_miss_prediction_to_fn=True, 
        nms_iou_threshold=0.5, qualitative_savepath=None, annotation_json_path=None
    ):
        self.iou_threshold = iou_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.count_miss_prediction_to_fn = count_miss_prediction_to_fn
        self.verbose = verbose
        self.classes = None

        self.qualitative_savepath = qualitative_savepath
    
        self.manucripts_of_images = None
        if annotation_json_path is not None:
            with open(annotation_json_path) as f:
                annotation_json = json.load(f)

            self.manucripts_of_images = {
                annot["id"]: annot["folder_path"] 
                for annot in annotation_json["images"]
            } 
            self.page_names_of_images = {
                annot["id"]: annot["file_name"] 
                for annot in annotation_json["images"]
            }
        
    def evaluate(self, model, data_loader):
        was_train = model.training
        model.eval()

        self.classes = data_loader.dataset.classes
        data_groups = torch.zeros(len(self.classes), 3).to(utils.get_device())  # TP, FP, FN

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=(self.verbose == False)):
                predictions = BboxConversions.create_bbox_records_batch(
                    model, batch, coco_format=False,
                    detr_extract=data_loader.dataset.detr_feature_extractor
                )
                targets = BboxConversions.get_targets(
                    batch, coco_format=False,
                    detr_extract=data_loader.dataset.detr_feature_extractor
                )

                data_groups = self.accumulate_data_groups(predictions, targets, data_groups, batch=batch)

        precision = data_groups[:, 0] / (data_groups[:, 0] + data_groups[:, 1] + 1e-9)
        recall = data_groups[:, 0] / (data_groups[:, 0] + data_groups[:, 2] + 1e-9)
        f1 = (2*precision*recall) / (precision + recall + 1e-9)

        metrics = {
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1": f1.mean().item()
        }
        if was_train:
            model.train()
        return metrics

    def accumulate_data_groups(self, predictions, ground_truth, data_groups, batch=None):
        if self.qualitative_savepath is not None:
            image_ids = [b["misc"]["image_id"] for b in batch]
            X, _ = utils.prepare_batch(batch)
        
        for image_iter, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            del pred["image_id"]

            for clz_it, clz in enumerate(self.classes):
                clz_pred_indices = pred["labels"] == clz
                clz_gt_indices = gt["labels"] == clz

                filtered_pred = {k: pred[k][clz_pred_indices] 
                                 for k in pred.keys()}
                filtered_gt = {k: gt[k][clz_gt_indices]
                                 for k in gt.keys()}
                
                # applying non maximum suppression on predicted bboxes
                nms_indices = torchvision.ops.nms(filtered_pred["boxes"], filtered_pred["scores"],
                                                  iou_threshold=self.nms_iou_threshold)

                filtered_pred = {
                    k: filtered_pred[k][nms_indices] for k in filtered_pred.keys()
                }
                pred_bboxes = filtered_pred["boxes"]
                gt_bboxes = filtered_gt["boxes"]

                if len(pred_bboxes) == 0:
                    data_groups[clz_it, 2] += len(gt_bboxes)  # increase FN
                    continue

                pairs = []
                for gt_bbox_it, gt_bbox in enumerate(gt_bboxes):
                    pairs += self.get_pairing_suggestions(gt_bbox_it, gt_bbox, pred_bboxes)
                pairs = sorted(pairs, key=lambda x: x["dist"].item())

                tp = 0
                while (len(pairs) != 0):
                    # remove pairs which cant be created anymore
                    poi = pairs[0]
                    pairs = list(filter(
                        lambda p:
                            p["gt_idx"] != poi["gt_idx"] and
                            p["p_idx"] != poi["p_idx"],
                        pairs
                    ))
                    tp += 1
                fp = len(pred_bboxes) - tp

                if self.count_miss_prediction_to_fn:
                    fn = len(gt_bboxes) - tp
                else:
                    fn = max(len(gt_bboxes) - len(pred_bboxes), 0)

                data_groups[clz_it, 0] += tp
                data_groups[clz_it, 1] += fp
                data_groups[clz_it, 2] += fn

                # save image with predictions
                if self.qualitative_savepath is not None and tp + fn + fp > 0:
                    if self.manucripts_of_images is not None:
                        manuscript_name = self.manucripts_of_images[image_ids[image_iter]]

                        if fn > 0:
                            folder_name = f"{manuscript_name}/FN"
                        elif fp > 0:
                            folder_name = f"{manuscript_name}/FP"
                        else:
                            folder_name = f"{manuscript_name}/TP"

                        filename = self.page_names_of_images[image_ids[image_iter]]

                    else:
                        if fn > 0:
                            folder_name = "FN"
                        elif fp > 0:
                            folder_name = "FP"
                        else:
                            folder_name = "TP"

                        filename = f"img_{image_ids[image_iter]}.jpg"

                    os.makedirs(os.path.join(self.qualitative_savepath, folder_name), exist_ok=True)
                    fullpath = os.path.join(self.qualitative_savepath, folder_name, filename)
                    
                    all_boxes = torch.vstack([pred_bboxes, gt_bboxes])
                    colors = ["red"]*len(pred_bboxes) + ["green"]*len(gt_bboxes)

                    image = X[image_iter].cpu() * 255
                    image = image.to(torch.uint8)

                    result = draw_bounding_boxes(image, all_boxes, colors=colors, width=2).numpy()
                    result = np.transpose(result, (1,2,0))

                    plt.figure(figsize=(20,20))
                    plt.imshow(result)
                    plt.savefig(fullpath, bbox_inches="tight")
                    plt.close()
                    
        return data_groups

    def get_pairing_suggestions(self, gt_bbox_it, gt_bbox, pred_bboxes):
        iou = torchvision.ops.box_iou(
            gt_bbox[None].to(utils.get_device()), pred_bboxes)[0]

        relevant_mask = iou >= self.iou_threshold
        if relevant_mask.sum() == 0:
            return []

        relevant_indices = torch.nonzero(relevant_mask)[0]
        return [
            {
                "gt_idx": gt_bbox_it,
                "p_idx": p_idx.item(),
                "dist": 1 - dist
            } for p_idx, dist in zip(relevant_indices, iou)
        ]


class CocoEvaluate:
    def __init__(self, gt_json_path=None, pred_json_path="./temp/temp.json",
                 verbose=False, delete_pred_json=True, all_names=False,
                 qualitative_evaluation=False, qualitative_savepath=None, 
                 number_of_figures=32, epoch_num=None):
        self.ann_type = "bbox"
        self.gt_json_path = gt_json_path
        self.pred_json_path = pred_json_path
        self.delete_pred_json = delete_pred_json
        self.all_names = all_names
        self.verbose = verbose
        
        self.qualitative_evaluation = qualitative_evaluation
        self.qualitative_savepath = qualitative_savepath
        self.number_of_figures = number_of_figures
        
        self.epoch_num = epoch_num
        if self.epoch_num is None:
            self.epoch_num = 0

        if self.qualitative_savepath is not None:
            os.makedirs(self.qualitative_savepath, exist_ok=True)

    @staticmethod
    def  get_metrics_names(all_names=False):
        if all_names == False:
            return [
                "AP@[.5:.05:.95]",
                "AP@50",
                "AP@75",
                "AR_1",
                "AR_10",
                "AR_100",
            ], np.array([0, 1, 2, 6, 7, 8])
        return [
            "AP@[.5:.05:.95]",
            "AP@50",
            "AP@75",
            "AP@[.5:.05:.95]-small_area",
            "AP@[.5:.05:.95]-medium_area",
            "AP@[.5:.05:.95]-large_area",
            "AR_1",
            "AR_10",
            "AR_100",
            "AR_100-small_area",
            "AR_100-medium_area",
            "AR_100-large_area",
        ], np.arange(12)

    def evaluate(self, model, data_loader):
        if self.gt_json_path is None:
            self.gt_json_path = data_loader.dataset.annot_json_path

        # create fake gt json 
        if data_loader.dataset.detection_only:
            os.makedirs("./temp", exist_ok=True)

            with open(self.gt_json_path, "r") as f:
                annotation_json = json.load(f)

            for i in range(len(annotation_json["annotations"])):
                annotation_json["annotations"][i]["category_id"] = 1
            
            fake_path = "./temp/fake_gt.json"
            self.gt_json_path = fake_path
            with open(fake_path, "w") as f:
                json.dump(annotation_json, f)


        prediction_bboxes = BboxConversions.create_bbox_records(
            model, data_loader, coco_format=True, verbose=self.verbose
        )
        metric_names, metric_indices = CocoEvaluate.get_metrics_names(self.all_names)

        state = BboxConversions.create_coco_results_json(prediction_bboxes, savepath=self.pred_json_path)

        if state:
            category_ids = data_loader.dataset.classes
            coco_metrics = self.compute_coco_metrics(data_loader.dataset.image_ids, category_ids)

            results = {}
            for k, idx in zip(metric_names, metric_indices):
                results[k] = coco_metrics[idx]

            if self.qualitative_evaluation:
                all_figures = self.create_bbox_images(data_loader, prediction_bboxes)
                if self.qualitative_savepath is None:
                    return results, all_figures
            return results

        else:
            results = {}
            for k in metric_names:
                results[k] = 0

            if self.qualitative_evaluation and self.qualitative_savepath is None:
                return results, []
            return results

    def create_bbox_images(self, data_loader, coco_prediction_bboxes):
        image_iter = 0
        all_figures = []

        for batch in data_loader:
            for data in batch:
                image_id = data["misc"]["image_id"]
                image = (data["X"] * 255).to(device="cpu", dtype=torch.uint8)
                gt_bboxes = data["boxes"].cpu()

                pred = list(filter(lambda box: box["image_id"] == image_id, 
                            coco_prediction_bboxes))
                if len(pred) > 0:
                    pred = pred[0]

                    nms_indices = torchvision.ops.nms(pred["boxes"], pred["scores"], iou_threshold=0.5)
                    pred = pred["boxes"][nms_indices].cpu()
                    pred[:, 2] += pred[:, 0]
                    pred[:, 3] += pred[:, 1]

                    #if for some reason the bbox is touching the end of the image, lets reduce it by 1px
                    pred[:, 2][pred[:, 2] >= image.shape[2]] -= 1
                    pred[:, 3][pred[:, 3] >= image.shape[1]] -= 1
                else:
                    pred = torch.zeros(0, 4)

                all_boxes = torch.vstack([pred, gt_bboxes])

                if len(all_boxes) > 0:
                    colors = ["red"]*len(pred) + ["green"]*len(gt_bboxes)
                    result = draw_bounding_boxes(image, all_boxes, colors=colors, width=2).numpy()
                    result = np.transpose(result, (1,2,0))
                    figure = plt.imshow(result)
                    
                    gt_mask = dataset.BasicTransform.bboxes_as_image(gt_bboxes, image.shape[-2:], coco_format=False).sum(axis=0)
                    pred_mask = dataset.BasicTransform.bboxes_as_image(pred, image.shape[-2:], coco_format=False).sum(axis=0)
                    
                    union = ((gt_mask > 0) | (pred_mask > 0)).sum()
                    intersect = ((gt_mask > 0) & (pred_mask > 0)).sum()
                    iou = intersect / union

                    title_string = f"Image_id: {image_id} | IoU: {iou:.3f}"
                    plt.xlabel("Truth = green | Predict = red")
                    plt.title(title_string)

                    if self.qualitative_savepath is not None:
                        path = f"{self.qualitative_savepath}/e{self.epoch_num}_{image_iter}.jpg"
                        plt.savefig(path, bbox_inches="tight")
                    else:
                        all_figures.append(figure)
                    plt.close()

                    image_iter += 1
                    if self.number_of_figures == image_iter:
                        return all_figures
        return all_figures

    def compute_coco_metrics(self, image_ids, category_ids):
        utils.HideOutput.hide()
        coco_gt = COCO(self.gt_json_path)
        coco_pred = coco_gt.loadRes(self.pred_json_path)

        coco_eval = COCOeval(coco_gt, coco_pred, self.ann_type)
        coco_eval.params.imgIds = image_ids
        coco_eval.params.catIds = category_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        utils.HideOutput.show()

        if self.delete_pred_json:
            os.remove(self.pred_json_path)
            dirname = os.path.dirname(self.pred_json_path)
            if len(os.listdir(dirname)) == 0 or os.listdir(dirname) == 1 and os.listdir()[0] == "fake_gt.json":
                shutil.rmtree(dirname)

        return coco_eval.stats

    @staticmethod
    def plot_metrics(results, metrics_names, seperate_plot_per_ds=True, figsize=(10, 5)):
        evenly_spaced_interval = np.linspace(0, 1, len(metrics_names))
        colors = [cm.rainbow(x) for x in evenly_spaced_interval]

        if seperate_plot_per_ds:
            for ds_name in results.keys():
                data = results[ds_name]
                plt.figure(figsize=figsize)

                for m_name, col in zip(metrics_names, colors):
                    values = [d[m_name] for d in data]
                    plt.plot(values, color=col, label=m_name)

                plt.legend()
                plt.title(f"COCO Metrics measured on '{ds_name}' dataset")
                plt.show()

        else:
            linestyles = ["-", "--", ".", "-."]
            if len(results) > len(linestyles):
                return
            linestyles = linestyles[:len(results)]

            plt.figure(figsize=figsize)

            for m_name, col in zip(metrics_names, colors):
                for ds_name, line in zip(results.keys(), linestyles):
                    data = results[ds_name]

                    values = [d[m_name] for d in data]
                    label = f"{ds_name.upper()} | {m_name}"
                    plt.plot(values, color=col, label=label, linestyle=line)

            plt.legend()
            plt.title(f"COCO Metrics")
            plt.show()


class BboxConversions:
    @staticmethod
    def pascal_to_coco(X):
        if isinstance(X, Iterable) == False:
            return None
        if type(X) != np.ndarray or not torch.is_tensor(X):
            simple_list = True
            X = torch.Tensor(X)

        input_ndim = X.ndim
        X = X[None] if input_ndim == 1 else X

        X[:, 2] -= X[:, 0]
        X[:, 3] -= X[:, 1]

        X = X[0] if input_ndim == 1 else X
        return X.tolist() if simple_list else X

    @staticmethod
    def coco_to_pascal(X):
        if isinstance(X, Iterable) == False:
            return None
        if type(X) != np.ndarray or not torch.is_tensor(X):
            simple_list = True
            X = torch.Tensor(X)

        input_ndim = X.ndim
        X = X[None] if input_ndim == 1 else X

        X[:, 2] += X[:, 0]
        X[:, 3] += X[:, 1]

        X = X[0] if input_ndim == 1 else X
        return X.tolist() if simple_list else X

    @staticmethod
    def create_bbox_records(model, data_loader, coco_format=True, verbose=False, device=utils.get_device()):
        was_train = model.training
        model.eval()
    
        prediction_bboxes = []    
        with torch.no_grad():
            for batch in tqdm(data_loader, disable=verbose == False):
                prediction_bboxes += BboxConversions.create_bbox_records_batch(
                    model, batch, coco_format=coco_format, device=device,
                    detr_extract=data_loader.dataset.detr_feature_extractor
                )
        if was_train:
            model.train()
        return prediction_bboxes
               
    @staticmethod
    def create_bbox_records_batch(model, batch, coco_format=True, device=utils.get_device(), detr_extract=None):
        if detr_extract is not None:
            return BboxConversions.create_bbox_records_batch_detr(model, batch, detr_extract, coco_format, device)

        X, y = utils.prepare_batch(batch, device=device)
        image_ids = [item["misc"]["image_id"] for item in batch]

        predictions = model(X, y)
        new_pred = []

        for i, p in enumerate(predictions):
            if coco_format and len(p["boxes"]):
                p["boxes"][:, 2:] -= p["boxes"][:, :2]
            
            p["image_id"] = image_ids[i]
            new_pred.append(p)
        return new_pred

    @staticmethod
    def create_bbox_records_batch_detr(model, batch, detr_extract, coco_format=True, device=utils.get_device()):
        encoding = utils.detr_batch_to_device(batch, dev=device)
        outputs = model(**encoding)
        sizes = torch.stack([enc["orig_size"] for enc in encoding["labels"]]).to(utils.get_device())
        results = detr_extract.post_process_object_detection(outputs, target_sizes=sizes, threshold=0.9)

        new_pred = []
        for enc, res in zip(encoding["labels"], results):
            res["boxes"][res["boxes"] < 0] = 0
    
            if coco_format and len(res["boxes"]):
                res["boxes"][:, 2:] -= res["boxes"][:, :2]

            res["image_id"] = enc["image_id"][0].item()
            new_pred.append(res)

        return new_pred

    @staticmethod
    def get_targets(batch, coco_format=False, device=utils.get_device(), detr_extract=None):
        if detr_extract is not None:
            encoding = utils.detr_batch_to_device(batch, dev=device)

            targets = []
            for enc in encoding["labels"]:
                labels = enc["class_labels"]
                orig_h, orig_w = enc["orig_size"]
                boxes = enc["boxes"] # relative COCO format
                
                boxes[:, [0, 2]] *= orig_w 
                boxes[:, [1, 3]] *= orig_h

                if coco_format == False:
                    boxes[:, 2] += boxes[:, 0]
                    boxes[:, 3] += boxes[:, 1]
                
                targets.append({
                    "boxes": boxes,
                    "labels": labels
                })
            return targets
        
        _, targets = utils.prepare_batch(batch, device=device)

        if coco_format:
            for i in range(len(targets)):
                targets[i]["boxes"][:, 2] -= targets[i]["boxes"][:, 0]
                targets[i]["boxes"][:, 3] -= targets[i]["boxes"][:, 1]
        return targets
    
    @staticmethod
    def create_coco_results_json(predictions, savepath="./results.json"):
        results = []

        for pred in predictions:
            if len(pred["boxes"]) > 0:
                pred_list = [
                    {
                        "image_id": int(pred["image_id"]),
                        "category_id": int(c),
                        "bbox": b.tolist(),
                        "score": float(s)
                    } for c, b, s in zip(
                        pred["labels"].to("cpu").numpy(),
                        pred["boxes"].to("cpu").numpy(),
                        pred["scores"].to("cpu").numpy()
                    )
                ]
                results += pred_list

        if len(results) > 0:
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            with open(savepath, "w", encoding="utf-8") as f:
                json.dump(results, f)
            
            return True
        return False


class ClassificationEval:        
    def __init__(self, verbose=False, qualitative_savepath=None) -> None:
        self.verbose = verbose
        self.qualitative_savepath = qualitative_savepath
        self.image_iter = 0

    def evaluate(self, model, loader, loss_func=None):
        was_training = model.training
        model.eval()
        
        data_groups = torch.zeros(4, device=utils.get_device())   #[TP, FN, FP, TN]
        loss_sum = 0

        with torch.no_grad():
            for batch in tqdm(loader, disable=(self.verbose == False)):
                X, y = utils.prepare_classification_batch(batch)
                out = model(X)

                if loss_func is not None:
                    loss_sum += loss_func(out, y)

                pred = out.argmax(dim=1)                
                for i in range(4):
                    data_groups[i] += ((y == (1 - i // 2)) & (pred == (1 - i % 2))).sum()

                if self.qualitative_savepath is not None:
                    self.qualitative_eval(X, y, pred)

        # calc data_groups, avg loss
        tp, fn, fp, tn = data_groups
        acc = (tp + tn) / (data_groups.sum() + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = (2*prec*rec) / (prec + rec + 1e-9)

        results = {
            "accuracy": acc.item(),
            "precision": prec.item(),
            "recall": rec.item(),
            "f1": f1.item()
        }
        if loss_func is not None:
            loss_sum /= len(loader)
            results.update({ "loss_eval": loss_sum.item() })
        
        if was_training:
            model.train()
        return results

    def qualitative_eval(self, X, truth, pred):
        for i in range(len(X)):
            if truth[i] == pred[i]:
                continue

            folder_name = "FN" if pred[i] == 0 else "FP"
            os.makedirs(os.path.join(self.qualitative_savepath, folder_name), exist_ok=True)
            fullpath = os.path.join(self.qualitative_savepath, folder_name, f"image_{self.image_iter}.jpg")

            image = dataset.BasicTransform.inverse_normalize(X[i, :3].cpu())
            image = utils.tensor_to_image(image, tanh=False, flip_channels=False)[0]
            self.image_iter += 1
            
            plt.figure(figsize=(20,20))
            plt.imshow(image)
            plt.savefig(fullpath, bbox_inches="tight")
            plt.close()


def results_to_string(results, dataset_name, epoch_num=None, no_epochs=None):
    if epoch_num is not None and no_epochs is not None:
        string = f"{epoch_num}/{no_epochs} --- {dataset_name} dataset --- "
    else:
        string = f"--- {dataset_name} dataset --- "

    for k in results.keys():
        string += f"{k}={results[k]:.3f} | "
    return string
