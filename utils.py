from sklearn.mixture import GaussianMixture
import torch
import numpy as np
import os
import sys
import json
from collections.abc import Iterable
from tqdm import tqdm 
import cv2 as cv
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision

import dataset

class HideOutput:
    stderr = None
    stdout = None
    
    @staticmethod
    def hide():
        HideOutput.stderr = sys.stderr
        HideOutput.stdout = sys.stdout
        null = open(os.devnull, 'w')
        sys.stdout = sys.stderr = null

    @staticmethod
    def show():
        sys.stderr = HideOutput.stderr
        sys.stdout = HideOutput.stdout


class ReplaceBatchNorms:
    @staticmethod
    def replace_all_batchnorms(model, norm_layer, freeze_norm_layer=False):
        paths_to_batchnorms = []
        ReplaceBatchNorms.find_batchnorm_paths(model, paths_to_batchnorms, [])

        for path in paths_to_batchnorms:
            module = model
            for i in range(len(path)):
                if i < len(path) - 1:
                    try:
                        idx = int(path[i])
                        module = module[idx]
                    except:
                        module = getattr(module, path[i])
                else:    
                    try:
                        idx = int(path[i])
                        module[idx] = ReplaceBatchNorms.replace_batchnorm(module[idx], norm_layer, freeze_norm_layer)
                    except:
                        setattr(module, path[i], ReplaceBatchNorms.replace_batchnorm(
                            getattr(module, path[i]), 
                            norm_layer,
                            freeze_norm_layer
                        ))

    @staticmethod
    def replace_batchnorm(layer, norm_layer, freeze=False):
        nf = layer.num_features

        if norm_layer == torch.nn.BatchNorm2d:
            new_layer = layer
        elif norm_layer == torch.nn.InstanceNorm2d:
            new_layer = torch.nn.InstanceNorm2d(nf)
        elif norm_layer == torch.nn.LayerNorm:
            new_layer = torch.nn.LayerNorm()
        elif norm_layer == torch.nn.GroupNorm:
            num_groups = nf // 16
            new_layer = torch.nn.GroupNorm(num_groups, nf)
        else:
            raise

        if freeze:
            for param in new_layer.parameters():
                param.requires_grad = False
            
        return new_layer

    @staticmethod
    def find_batchnorm_paths(model, paths_to_batchnorms, name_chain):
        for name, module in model.named_children():
            if type(module) == torch.nn.BatchNorm2d:
                paths_to_batchnorms.append(name_chain + [name])
            else:
                ReplaceBatchNorms.find_batchnorm_paths(module, paths_to_batchnorms, name_chain + [name])


class QvireDetector:
    def __init__(self, detector) -> None:
        self.detector = detector

    def __call__(self, image, coco_format=False, return_relative_bboxes=False, device="cpu"):
        """
        This function expects an image whose values are uint8 values and their color channels are ordered as an RGB image. 
        Image shape is expected to be [height, width, channels=3] and should be a numpy array
        
        coco_format == True -> bbox = [x, y, w, h] 
        coco_format == False -> bbox = [x0, y0, x1, y1]
        """
        self.detector = self.detector.to(device)
        if self.detector.training:
            self.detector.eval()

        with torch.no_grad():
            X = self._data_preprocess(image).to(device)
            old_shape = image.shape[:-1]
            new_shape = X.shape[-2:]

            out = self.detector(X)[0]

            bboxes = self._prediction_postprocess(
                out, old_shape, new_shape, 
                coco_format, return_relative_bboxes
            )
            return bboxes.to(dtype=torch.int).tolist()

    def _data_preprocess(self, image):
        downsampled_image = cv.resize(image, (1066, 800), interpolation=cv.INTER_AREA)
        return image_to_tensor(downsampled_image, flip_channels=False)[None]


    def _prediction_postprocess(self, out, old_shape, new_shape, coco_format=False, return_relative_bboxes=False):
        orig_h, orig_w = old_shape
        new_h, new_w = new_shape
        
        nms_indices = torchvision.ops.nms(out["boxes"], out["scores"], iou_threshold=0.5)
        bboxes = out["boxes"][nms_indices].cpu()

        if coco_format:
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / new_w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / new_h
        if return_relative_bboxes:
            return bboxes
        
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * orig_w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * orig_h
        return bboxes


class QvirePatchClassifier:
    def __init__(self, model) -> None:
        self.classifier = model

    def __call__(
        self, image, coco_format=False, return_relative_bboxes=False, device="cpu", 
        loader_kwargs={
            "batch_size": 4,
            "num_workers": 2
        }
    ):
        """
        This function expects an image whose values are uint8 values and their color channels are ordered as an RGB image. 
        Image shape is expected to be [height, width, channels=3] and should be a numpy array
        
        coco_format == True -> bbox = [x, y, w, h] 
        coco_format == False -> bbox = [x0, y0, x1, y1]
        """
        self.classifier = self.classifier.to(device)
        if self.classifier.training:
            self.classifier.eval()

        with torch.no_grad():
            image_patches, patch_coordinates = self._data_preprocess(image)
            loader = DataLoader(image_patches, pin_memory=True, **loader_kwargs)
            
            all_pred = []
            for X in tqdm(loader):
                X = X.to(device)
                pred = self.classifier(X)
                all_pred.append(pred)    
            all_pred = torch.vstack(all_pred)

            bboxes = self._prediction_postprocess(
                all_pred, patch_coordinates, image.shape[:2], 
                coco_format=coco_format, 
                return_relative_bboxes=return_relative_bboxes
            )
            return bboxes

    def _data_preprocess(self, image):
        distance_mask = create_distance_mask_from_image(
            image, apply_bilateral_filter=True, gmm_max_number_of_pixels_processed=100_000,
            gmm_components=4, rgb_percentile_thresholds=(20, 80), bilateral_kernel_size=25
        )
        distance_mask[distance_mask == 256] = 0
        distance_mask = distance_mask.astype(np.uint8)
        image = np.concatenate([image, distance_mask[..., None]], axis=-1)
        
        patches, crop_coords = create_patches_from_image(image, patch_size=(512,512), relative_overlap_size=0.5, filter_patches=True)
        resize_op = transforms.Resize((224,224))
        normalize_op = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        for i in range(len(patches)):
            patches[i] = image_to_tensor(patches[i], flip_channels=False)
            patches[i] = resize_op(patches[i])
        
            img, mask = patches[i][:3], patches[i][3:]
            img = normalize_op(img)
            patches[i] = torch.vstack([img, mask])


        return patches, crop_coords

    def _prediction_postprocess(
        self, predictions, patch_coordinates, orig_shape, 
        coco_format=False, return_relative_bboxes=False
    ):
        labels = predictions.argmax(dim=1)
        labels_1_idx = torch.where(labels == 1)[0]
        scores = predictions[labels_1_idx][:, 1]

        bboxes = torch.Tensor(patch_coordinates).to(predictions.device)
        bboxes = bboxes[labels_1_idx]

        nms_indices = torchvision.ops.nms(bboxes, scores, iou_threshold=0.124)
        bboxes = bboxes[nms_indices]

        if coco_format:
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]

        if return_relative_bboxes:
            h, w = orig_shape
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / w
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / h
            return bboxes.to("cpu", dtype=torch.int).tolist()

        return bboxes.to("cpu", dtype=torch.int).tolist()


def create_patches_from_image(image, patch_size=(512,512), relative_overlap_size=0.5, filter_patches=True):
    image_height, image_width  = image.shape[:2]

    x_offset = patch_size[0] - int(relative_overlap_size*patch_size[0])
    y_offset = patch_size[1] - int(relative_overlap_size*patch_size[1])

    patches = []
    all_crop_coords = []
    for y in range(0, image_height, y_offset):
        for x in range(0, image_width, x_offset):
            crop_coordinates = _contain_crop_within_image(
                x, y, patch_size,
                image_width, image_height
            )
            x0, y0, x1, y1 = crop_coordinates
            patch = image[y0: y1, x0: x1]

            # filter out patches which should not contain quiremarks based on their position to the edges of the page
            if filter_patches:
                patch_mask = patch[..., 3]
                if patch_mask.mean() < 80:
                    continue
            
            patches.append(patch)
            all_crop_coords.append(crop_coordinates)

    return patches, all_crop_coords
    
def _contain_crop_within_image(x, y, patch_size, image_width, image_height):
        new_x = x
        new_y = y

        if x+patch_size[0] > image_width:
            diff = (x+patch_size[0]) - image_width
            new_x = x - diff
        if y+patch_size[1] > image_height:
            diff = (y+patch_size[1]) - image_height
            new_y = y - diff

        return new_x, new_y, new_x+patch_size[0], new_y+patch_size[1]


def create_distance_mask_from_image(
    img, apply_bilateral_filter=True, gmm_max_number_of_pixels_processed=100_000,
    gmm_components=4, rgb_percentile_thresholds=(20, 80), bilateral_kernel_size=50
):
    if apply_bilateral_filter:
        img = cv.bilateralFilter(img, bilateral_kernel_size, 35, 35)
    
    rgb_thresholds = _train_gmm_on_image(
        img, gmm_components, rgb_percentile_thresholds,
        gmm_max_number_of_pixels_processed
    )
    h, w = img.shape[:2]
    avg_size = (h+w) / 2
    
    rgb_masks = []
    for c in range(3):
        channel = img[:, :, c]
        rgb_masks.append(
            (channel >= rgb_thresholds[c][0]) & 
            (channel <= rgb_thresholds[c][1])
        )
    combined_mask = rgb_masks[0] & rgb_masks[1] & rgb_masks[2]

    page_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    page_mask[combined_mask] = 255

    kernel_small_unit = 186.6
    k1 = int(avg_size / kernel_small_unit)
    
    kernel_small = np.ones((k1, k1), np.uint8)
    denoised_page_mask = cv.morphologyEx(page_mask, cv.MORPH_CLOSE, kernel_small)

    kernel = np.ones((101,101), np.uint8) 
    sure_fg = cv.erode(denoised_page_mask, kernel)

    inv_sure_bg = cv.rectangle(
        np.zeros(img.shape[:2], dtype=np.uint8),
        (150, 150), (img.shape[1]-150, img.shape[0]-150),
        (255, 0, 0), -1
    )
    sure_bg = np.bitwise_not(inv_sure_bg)

    mask_for_watershed = np.zeros(img.shape[:2], dtype=np.int32)
    mask_for_watershed[sure_bg == 255] = 1
    mask_for_watershed[sure_fg == 255] = 2
    watershed_segm = cv.watershed(img, mask_for_watershed)

    watershed_segm[watershed_segm < 2] = 0
    watershed_segm[watershed_segm == 2] = 255
    watershed_segm = watershed_segm.astype(dtype=np.uint8)

    proximity_to_edges = _compute_proximity_to_edges(watershed_segm)
    return proximity_to_edges


def _train_gmm_on_image(
    img, gmm_components=4, rgb_percentile_thresholds=(5, 95),
    gmm_max_number_of_pixels_processed=10_000_000
):    
    data = img.reshape(-1, 3)
    train_data = data[:gmm_max_number_of_pixels_processed]
    gmm = GaussianMixture(n_components=gmm_components, covariance_type="tied")
    gmm.fit(train_data)
    
    labels = np.zeros(len(data))
    for i in range(0, len(data), 10_000_000):
        labels[i: i + 10_000_000] = gmm.predict(data[i: i + 10_000_000])

    rgb_thresholds = _compute_rgb_distributions(
        labels, [img], gmm_components, flip_channels=False, 
        distrib_percentile_thresholds=rgb_percentile_thresholds
    )
    return rgb_thresholds


def _compute_proximity_to_edges(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    all_contours = np.vstack([c for c in contours]) 
    hull = cv.convexHull(all_contours)

    mask = cv.drawContours(
        np.zeros(image.shape[:2], np.uint8), [hull], 
        contourIdx=0, color=[255,255,255], thickness=1
    )
    mask = ~mask
    mask_inside = cv.drawContours(
        np.zeros(image.shape[:2], np.uint8), [hull], 
        contourIdx=0, color=[255,255,255], thickness=-1
    )
    distance = cv.distanceTransform(mask, cv.DIST_L2, 3)
    distance = (distance / distance.max())
    distance = (distance - distance.max()) * -1
    
    distance = (distance ** 2) * mask_inside
    distance = cv.GaussianBlur(distance, (51,51), 0)

    distance = (distance / distance.max()) * 255
    distance = np.uint16(distance)
    distance[image == 0] = 256
    
    return distance


def _distance_centre(img):
    h, w = img.shape[:2]

    mask = cv.rectangle(
        np.zeros((h, w), dtype=np.uint8), 
        (w // 2, h // 2), (w // 2 + 1, h // 2 + 1), 
        color=[255,255,255], thickness=1
    )
    mask = ~mask

    distance = cv.distanceTransform(mask, cv.DIST_L2, 3)
    distance = distance / distance.max()
    distance = (distance - 1) * -1
    return distance


def _compute_rgb_distributions(
        labels, all_imgs, gmm_components, flip_channels=True,
        distrib_percentile_thresholds=(5, 95)
    ):
    page_label_idx = _identify_page_label(labels, all_imgs, gmm_components)

    rgb_distributions = [np.array([], dtype=np.uint8) for _ in range(3)]
    offset = 0

    for img in all_imgs:
        num_pixels = img.shape[0] * img.shape[1]
        if flip_channels:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask_image = labels[offset: offset + num_pixels].reshape(img.shape[:2])

        page_pixels = img[mask_image == page_label_idx]
        for i in range(len(rgb_distributions)):
            distrib = page_pixels[:, i]
            rgb_distributions[i] = np.hstack([rgb_distributions[i], distrib])

        offset += num_pixels

    rgb_thresholds = [np.zeros(2, dtype=np.uint8) for _ in range(3)]
    low_thresh, high_thresh = distrib_percentile_thresholds
    for i, distrib in enumerate(rgb_distributions):
        rgb_thresholds[i][0] = np.percentile(distrib, low_thresh)
        rgb_thresholds[i][1] = np.percentile(distrib, high_thresh)

    return rgb_thresholds
     

def _identify_page_label(labels, all_imgs, gmm_components):
    label_ranks = np.zeros(gmm_components)
    offset = 0

    for img in all_imgs:
        num_pixels = img.shape[0] * img.shape[1]
        distance_map = _distance_centre(img)
        mask_image = labels[offset: offset + num_pixels].reshape(img.shape[:2])
        label_values = np.arange(gmm_components)
        
        for il, label in enumerate(label_values):
            label_map = np.copy(mask_image)
            label_map[mask_image != label] = 0
            label_map[mask_image == label] = 1
            label_ranks[il] += (label_map * distance_map).sum()
        offset += num_pixels

    return label_ranks.argmax()


def no_randomness(seed=0):
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

    
def set_requires_grad(models, value):
    if isinstance(models, Iterable) == False:
        models = [models]

    for m in models:
        for param in m.parameters():
            param.requires_grad = value


def divide_dataset_cv(annot_json_path, eval_manuscripts_folds, oversampling_kwargs):
    train_folds, eval_folds = [], []
    for eval_manuscripts in eval_manuscripts_folds:
        train_idx, valid_idx, _ = divide_dataset(
            annot_json_path, valid_manuscript_names=eval_manuscripts,
            test_manuscript_names=[],
            oversampling_kwargs=oversampling_kwargs
        )
        train_folds.append(train_idx)
        eval_folds.append(valid_idx)

    return train_folds, eval_folds


def divide_dataset(
        annot_json_path, train_manuscript_names=None, valid_manuscript_names=None,
        test_manuscript_names=None, dataset_version="first",
        divide_manuscripts_kwargs={
            "train_offset": 0.7,
            "valid_offset": 0.85,
            "sortby": "manuscript_num_images",
            "descending_order": False
        },
        oversampling_kwargs={
            "oversampling": True,
            "max_oversampling_coefficient": 25,
            "max_eval_oversampling_coefficient": 1,
            "smart_sampling": True
        },
        verbose=False,
    ):
    valid_dataset_versions = ["first", "second_33", "third_59"]
    if dataset_version is not None and dataset_version not in valid_dataset_versions:
        raise "invalid dataset version name"
    if dataset_version is None:
        dataset_version = valid_dataset_versions[-1]


    with open(annot_json_path, "r") as f:
        annotation_json = json.load(f)
    
    if valid_manuscript_names is None or test_manuscript_names is None:
        train_manuscript_names, valid_manuscript_names, test_manuscript_names = divide_manuscripts(
            annotation_json, **divide_manuscripts_kwargs
        )
    elif train_manuscript_names is None:
        with open("./dataset_versions.json") as f:
            all_manuscripts = json.load(f)[dataset_version]
    
        train_manuscript_names = list(filter(
            lambda manu: 
                manu not in valid_manuscript_names 
                and manu not in test_manuscript_names, 
            all_manuscripts
        ))

    train_image_ids = []
    valid_image_ids = []
    test_image_ids = []

    for im in annotation_json["images"]:
        if im["folder_path"] in train_manuscript_names:
            train_image_ids.append(im["id"])
        if im["folder_path"] in valid_manuscript_names:
            valid_image_ids.append(im["id"])
        if im["folder_path"] in test_manuscript_names:
            test_image_ids.append(im["id"])

    train_image_ids = np.random.permutation(np.array(train_image_ids))
    if oversampling_kwargs["oversampling"]:
        kw = oversampling_kwargs

        train_image_ids = oversample_annotations(
            annotation_json, train_image_ids,
            max_oversampling_coefficient=kw["max_oversampling_coefficient"],
            smart_sampling=kw["smart_sampling"],
            verbose=verbose
        )

        if kw["max_eval_oversampling_coefficient"] > 1:
            valid_image_ids = oversample_annotations(
                annotation_json, valid_image_ids, 
                max_oversampling_coefficient=kw["max_eval_oversampling_coefficient"],
                smart_sampling=False,
                verbose=verbose,
            )

    train_image_ids = np.random.permutation(np.array(train_image_ids))
    valid_image_ids = np.random.permutation(np.array(valid_image_ids))
    test_image_ids = np.random.permutation(np.array(test_image_ids))
    return train_image_ids, valid_image_ids, test_image_ids


def oversample_annotations(annotation_json, image_ids, max_oversampling_coefficient, smart_sampling=True, verbose=False):
    if len(image_ids) == 0:
        return []
    image_ids = np.array(image_ids)
    annotation_im_ids = np.unique([annot["image_id"] for annot in annotation_json["annotations"]])

    if smart_sampling == False:
        annot_ids = image_ids[np.isin(image_ids, annotation_im_ids)]
        annot_ids = np.repeat(annot_ids, max_oversampling_coefficient)

        no_annot_ids = image_ids[~np.isin(image_ids, annotation_im_ids)]
         
        image_ids = np.random.permutation(np.hstack([no_annot_ids, annot_ids]))
        return image_ids

    images_with_annotations = list(filter(lambda im: im["id"] in annotation_im_ids and im["id"] in image_ids, annotation_json["images"]))
    images_without_annotations = list(filter(lambda im: im["id"] not in annotation_im_ids and im["id"] in image_ids, annotation_json["images"]))

    manuscripts = np.unique([im["folder_path"] for im in images_with_annotations])
    duplications = {
        manu: {
            "num_annotations": 0,
            "duplicates": {},
        } for manu in manuscripts
    }

    for im_meta in images_with_annotations:
        duplications[im_meta["folder_path"]]["duplicates"][im_meta["id"]] = 1
        annots = list(filter(lambda annot: annot["image_id"] == im_meta["id"], annotation_json["annotations"]))
        duplications[im_meta["folder_path"]]["num_annotations"] += len(annots)

    original_annot_len = len(images_with_annotations)
    while (
            len(images_with_annotations) < len(images_without_annotations) 
            and original_annot_len*max_oversampling_coefficient > len(images_with_annotations
        )):
        duplications = _choose_image_to_sample(duplications, annotation_json, images_with_annotations, max_oversampling_coefficient)
        if duplications is None:
            break

    images = np.array(images_with_annotations + images_without_annotations)
    images = np.random.permutation(images)
    image_ids = np.array([im["id"] for im in images])
    return image_ids


def _choose_image_to_sample(duplications, annotation_json, images_with_annotations, max_oversampling_coefficient):
    manuscripts = list(duplications.keys())
    manu_probabilities = np.array([duplications[k]["num_annotations"] for k in duplications.keys()])

    manu_probabilities = 1 / manu_probabilities
    manu_probabilities = np.cumsum(manu_probabilities) / manu_probabilities.sum()

    rng_value = np.random.random()
    for i in range(len(manu_probabilities)):
        if i == 0:
            low = 0
            high = manu_probabilities[i]
        else:
            low = manu_probabilities[i-1]
            high = manu_probabilities[i]
        if (
            rng_value >= low
            and rng_value < high
        ):            
            # choose randomly (uniformly) an image with annotation which will be duplicated
            image_ids = list(duplications[manuscripts[i]]["duplicates"].keys())
            idx = np.random.randint(len(image_ids))
            annots = list(filter(lambda annot: annot["image_id"] == image_ids[idx], annotation_json["annotations"]))

            duplications[manuscripts[i]]["duplicates"][image_ids[idx]] += 1
            duplications[manuscripts[i]]["num_annotations"] += len(annots)

            images_with_annotations.append(
                list(filter(lambda im: im["id"] == image_ids[idx], annotation_json["images"]))[0]
            )

            if duplications[manuscripts[i]]["duplicates"][image_ids[idx]] == max_oversampling_coefficient:
                del duplications[manuscripts[i]]["duplicates"][image_ids[idx]]

                if len(image_ids) == 1:
                    del duplications[manuscripts[i]]

            return duplications
    return None


def divide_manuscripts(annotation_json, train_offset=0.64, valid_offset=0.8, 
                        sortby="manuscript_num_images", descending_order=False):
    if sortby not in ["manuscript_num_images", "manuscript_num_annotations"]:
        raise "Invalid 'sortby' argument"
    
    folder_paths = np.unique([im["folder_path"] for im in annotation_json["images"]])
    num_annotations_per_manuscript = np.zeros(len(folder_paths))
    num_images_per_manuscript = np.zeros(len(folder_paths))

    for i in range(len(folder_paths)):
        images = list(filter(lambda im: im["folder_path"] == folder_paths[i], annotation_json["images"]))
        image_ids = [im["id"] for im in images]
        num_annotations_per_manuscript[i] = len(list(filter(lambda annot: annot["image_id"] in image_ids, annotation_json["annotations"])))
        num_images_per_manuscript[i] = len(images)

    values_of_interest = None
    if sortby == "manuscript_num_images":
        values_of_interest = num_images_per_manuscript
    elif sortby == "manuscript_num_annotations":
        values_of_interest = num_annotations_per_manuscript
    else:
        raise

    indices = np.argsort(values_of_interest)
    if descending_order:
        indices = indices[::-1]

    manuscript_names = folder_paths[indices]
    values_of_interest = values_of_interest[indices]

    offsets = np.cumsum(values_of_interest) / values_of_interest.sum()
    train_manuscript_indices = np.where(offsets < train_offset)
    valid_manuscript_indices = np.where((offsets >= train_offset) & (offsets < valid_offset))
    test_manuscript_indices =  np.where(offsets >= valid_offset)

    train_manuscript_names = manuscript_names[train_manuscript_indices]
    valid_manuscript_names = manuscript_names[valid_manuscript_indices]
    test_manuscript_names = manuscript_names[test_manuscript_indices]

    return train_manuscript_names, valid_manuscript_names, test_manuscript_names


# initial data distribution -> not used
def balanced_subsets(annot_json_path, train_offset=0.64, valid_offset=0.8):
    with open(annot_json_path, "r") as f:
        annotation_json = json.load(f)

    train_ids, valid_ids, test_ids = [], [], []

    image_ids = [im["id"] for im in annotation_json["images"]]
    image_ids_with_bboxes = list(set([annot["image_id"] for annot in annotation_json["annotations"]]))
    image_ids_without_bboxes = list(filter(lambda id: id not in image_ids_with_bboxes, image_ids))

    image_ids_with_bboxes = np.random.permutation(image_ids_with_bboxes).tolist()
    real_train_offset = int(len(image_ids_with_bboxes) * train_offset)
    real_valid_offset = int(len(image_ids_with_bboxes) * valid_offset)

    #distribute images with annotations
    train_ids += image_ids_with_bboxes[:real_train_offset]
    valid_ids += image_ids_with_bboxes[train_offset: real_valid_offset]
    test_ids += image_ids_with_bboxes[real_valid_offset:]

    image_ids_without_bboxes = np.random.permutation(image_ids_without_bboxes).tolist()
    real_train_offset = int(len(image_ids_without_bboxes) * train_offset)
    real_valid_offset = int(len(image_ids_without_bboxes) * valid_offset)

    train_ids += image_ids_without_bboxes[:real_train_offset]
    valid_ids += image_ids_without_bboxes[real_train_offset: real_valid_offset]
    test_ids += image_ids_without_bboxes[real_valid_offset:]

    return np.array(train_ids), np.array(valid_ids), np.array(test_ids)


def split_train_test(dataset_path, test_split=0.2, permutations=None):
    if permutations is None:
        dir_size = len(os.listdir(dataset_path))
        permut = np.random.permutation(dir_size)

        split_index = int((1-test_split)*dir_size)
        train_indices = permut[:split_index]
        test_indices = permut[split_index:]
    else:
        permutations = np.random.permutation(permutations)
        split_index = int((1-test_split)*len(permutations))
        train_indices = permutations[:split_index]
        test_indices = permutations[split_index:]
        
    return train_indices, test_indices


def get_mean_std(loader, channels=3):
    sums = torch.zeros(channels).to(get_device())
    sums_sq = torch.zeros(channels).to(get_device())
    
    N = 0
    for batch in loader:
        X, _ = prepare_batch(batch)
        X = torch.stack(X)

        sums += X.sum(axis=[0,2,3])
        sums_sq += (X**2).sum(axis=[0,2,3])
    
        shape = torch.Tensor(list(X.shape))
        N += shape.prod() / channels
        
    mean = sums / N
    std = ((sums_sq) / N - mean**2)**0.5
    return mean, std


def tolist_collate_fn(batch):
    list_data = []
    for data in batch:
        list_data.append(data)
    return list_data


def data_to_device(X, targets=None, dev=None):
    if dev is None: 
        dev = get_device()

    X = [x.to(dev) for x in X]
    if targets is None:
        return X

    targets = [
        {
            k: t[k].to(dev) 
            for k in t.keys()
        } 
        for t in targets
    ]
    return X, targets


def image_to_tensor(img, dtype=torch.float, flip_channels=True):
    mask = None
    if img.shape[-1] == 4:
        img, mask = img[..., :3], img[..., 3]
        mask = torch.from_numpy(mask).to(dtype) / 255

    if flip_channels:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    arr = np.array(img)
    arr = np.transpose(arr, axes=[2,0,1])
    arr = torch.from_numpy(arr).to(dtype) / 255
    
    if mask is not None:
        return torch.vstack([arr, mask[None]])
    return arr


def tensor_to_image(X, tanh=True, flip_channels=True):
    images = []
    if X.ndim == 3:
        X = X[None]
    for xi in X:
        if tanh:
            img_X = ((xi + 1) / 2) * 255
        else:
            img_X = xi * 255
        img_X = img_X.cpu().numpy().astype("int")
        img_X = np.transpose(img_X, (1,2,0))

        if flip_channels:
            r,g,b = cv.split(img_X)
            img_X = cv.merge([b,g,r])

        img_X = np.uint8(img_X)
        images.append(img_X)
        
    return images


def transform_one_tensor_to_grayscale_tensor(X):
    #transform to image
    img_X = X * 255
    img_X = img_X.cpu().numpy().astype("int")
    img_X = np.transpose(img_X, (1,2,0))
    img_X = np.uint8(img_X)

    #convert to grayscale
    img_X = cv.cvtColor(img_X, cv.COLOR_RGB2GRAY)

    #transform back to tensor    
    arr = torch.from_numpy(img_X).to(torch.float) / 255
    arr = arr.repeat(3, 1, 1)
    return arr


def detr_collate_fn(feature_extractor, batch):
  pixel_values = [item[0] for item in batch]
  labels = [item[1] for item in batch]

  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")

  batch = {}
  batch["pixel_values"] = encoding["pixel_values"]
  batch["pixel_mask"] = encoding["pixel_mask"]
  batch["labels"] = labels
  return batch


def detr_batch_to_device(batch, dev=get_device()):
    batch["pixel_values"] = batch["pixel_values"].to(dev)
    batch["pixel_mask"] = batch["pixel_mask"].to(dev)
    batch["labels"] = [{k: v.to(dev) for k, v in t.items()} for t in batch["labels"]]

    return batch


def prepare_batch(batch, device=get_device()):
    X = [item["X"] for item in batch]
    targets = [{"boxes": item["boxes"], "labels": item["labels"]}
            for item in batch]

    X, targets = data_to_device(X, targets, dev=device)
    return X, targets


def prepare_classification_batch(batch, device=get_device()):
    if type(batch[0]) == list or type(batch[0]) == tuple:
        X = torch.stack([b[0] for b in batch])
        y = torch.hstack([b[1] for b in batch])
    else:
        X, y = batch
        
    X = X.to(device)
    y = y.to(device)
    return X,y 


def optimal_data_augm_transformations_arguments():
    return {
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


def prepare_data_wrapper_cv(
        data_path, annot_json_path, eval_manuscripts_folds=None, mask_path=None, 
        classification_task=False, patches=False,
        loader_kwargs={ "batch_size": 8, "num_workers": 2 },
        custom_data_augm=None, basic_transformations=None,
        subsample_eval_datasets=False, oversample_eval_ds=True
    ):
    
    train_loaders, static_train_loaders, eval_loaders = [], [], []

    if eval_manuscripts_folds is None:
        eval_manuscripts_folds = [
            [
                'Glose_ordinaire_sur_le_livre_de_l_Exode',
                'SAINT-OMER,_Bibliotheque_municipale,_0017_(Reproduction_integrale)',
            ],
            [
                'SAINT-OMER,_Bibliotheque_municipale,_0068_(Reproduction_integrale)',
                'SAINT-OMER,_Bibliotheque_municipale,_0216_(Reproduction_integrale)',
            ],
            [
                'SAINT-OMER,_Bibliotheque_municipale,_0312_(Reproduction_integrale)',
                'SAINT-OMER,_Bibliotheque_municipale,_0705_(Reproduction_integrale)',
            ],
            [
                'SAINT-OMER,_Bibliotheque_municipale,_0715,_vol._1_(Reproduction_integrale)',
                'SAINT-OMER,_Bibliotheque_municipale,_0735_(Reproduction_integrale)',
            ],
            [
                'SAINT-OMER,_Bibliotheque_municipale,_0792_(Reproduction_integrale)',
                'SAINT-OMER,_Bibliotheque_municipale,_0820_(Reproduction_integrale)',
            ],
        ]
        pass

    for eval_manuscripts in eval_manuscripts_folds:
        tr_l, stat_tr_l, val_l, _ = prepare_data_wrapper(
            data_path, annot_json_path, mask_path, 
            valid_manuscript_names=eval_manuscripts, test_manuscript_names=[],
            classification_task=classification_task, patches=patches,
            loader_kwargs=loader_kwargs, custom_data_augm=custom_data_augm,
            basic_transformations=basic_transformations,
            subsample_eval_datasets=subsample_eval_datasets,
            oversample_eval_ds=oversample_eval_ds
        )
        train_loaders.append(tr_l)
        static_train_loaders.append(stat_tr_l)
        eval_loaders.append(val_l)

    return train_loaders, static_train_loaders, eval_loaders


def prepare_data_wrapper(
        data_path, annot_json_path, mask_path=None, 
        train_manuscript_names=None,
        valid_manuscript_names=["SAINT-OMER,_Bibliotheque_municipale,_0715,_vol._1_(Reproduction_integrale)"],
        test_manuscript_names = ["SAINT-OMER,_Bibliotheque_municipale,_0705_(Reproduction_integrale)"],
        classification_task=False, patches=False, max_oversampling_coefficient=None,
        loader_kwargs={ "batch_size": 8, "num_workers": 2 },
        custom_data_augm=None, basic_transformations=None,
        subsample_eval_datasets=False, oversample_eval_ds=True,
        dataset_version="first", new_mask=True, out_of_page_mask_value=0,
        undersample_no_annotations=False
    ):
    no_randomness()

    oversampling_kwargs = {
        "oversampling": True,
        "max_oversampling_coefficient": 25 if max_oversampling_coefficient is None else max_oversampling_coefficient,
        "max_eval_oversampling_coefficient": 10 if oversample_eval_ds else 0,
        "smart_sampling": True
    }
    if patches and classification_task == False:
        oversampling_kwargs = {
            "oversampling": True,
            "max_oversampling_coefficient": 75 if max_oversampling_coefficient is None else max_oversampling_coefficient,
            "max_eval_oversampling_coefficient": 15 if oversample_eval_ds else 0,
            "smart_sampling": False
        }
    elif patches:
        oversampling_kwargs = {
            "oversampling": True,
            "max_oversampling_coefficient": 200 if max_oversampling_coefficient is None else max_oversampling_coefficient,
            "max_eval_oversampling_coefficient": 50 if oversample_eval_ds else 0,
            "smart_sampling": False
        }

    train_image_ids, valid_image_ids, test_image_ids = divide_dataset(
        annot_json_path, 
        train_manuscript_names=train_manuscript_names,
        valid_manuscript_names=valid_manuscript_names,
        test_manuscript_names=test_manuscript_names,
        oversampling_kwargs=oversampling_kwargs,
        dataset_version=dataset_version,
    )    

    train_static_image_ids = train_image_ids
    if subsample_eval_datasets:
        train_static_image_ids = subsample_eval_ds(
            train_static_image_ids, annot_json_path, num_images_per_categ=1000, 
            fetch_unique_annot=True if patches else False
        )
        valid_image_ids = subsample_eval_ds(valid_image_ids, annot_json_path, num_images_per_categ=1000, fetch_unique_annot=False)

    if custom_data_augm is None:
        data_augm_transformations = optimal_data_augm_transformations_arguments()
    else:
        data_augm_transformations = custom_data_augm
    
    if basic_transformations is None:
        basic_transformations = {}

    train_ds = dataset.MyDataset(
        train_image_ids, data_path, annot_json_path,
        transform_kwargs=data_augm_transformations,
        additional_mask_path=mask_path,
        classification_task=classification_task,
        new_distance_mask=new_mask, out_of_page_mask_value=out_of_page_mask_value,
        undersample_no_annotations=undersample_no_annotations
    )
    static_train_ds = dataset.MyDataset(
        train_static_image_ids, data_path, annot_json_path,
        transform_kwargs=basic_transformations,
        additional_mask_path=mask_path,
        classification_task=classification_task,
        new_distance_mask=new_mask, out_of_page_mask_value=out_of_page_mask_value,
        undersample_no_annotations=undersample_no_annotations
    )
    valid_ds = dataset.MyDataset(
        valid_image_ids, data_path, annot_json_path,
        transform_kwargs=basic_transformations,
        additional_mask_path=mask_path,
        classification_task=classification_task,
        new_distance_mask=new_mask, out_of_page_mask_value=out_of_page_mask_value,
        undersample_no_annotations=undersample_no_annotations
    )
    test_ds = dataset.MyDataset(
        test_image_ids, data_path, annot_json_path,
        transform_kwargs=basic_transformations,
        additional_mask_path=mask_path,
        classification_task=classification_task,
        new_distance_mask=new_mask, out_of_page_mask_value=out_of_page_mask_value
    )


    train_loader = DataLoader(
        train_ds, shuffle=True,
        collate_fn=tolist_collate_fn, pin_memory=True,
        **loader_kwargs
    )
    static_train_loader = DataLoader(
        static_train_ds,
        collate_fn=tolist_collate_fn, pin_memory=True,
        **loader_kwargs
    )
    valid_loader = DataLoader(
        valid_ds,
        collate_fn=tolist_collate_fn, pin_memory=True,
        **loader_kwargs
    )    
    test_loader = DataLoader(
        test_ds,
        collate_fn=tolist_collate_fn, pin_memory=True,
        **loader_kwargs
    )    
    return (
        train_loader,
        static_train_loader,
        valid_loader,
        test_loader
    )


def subsample_eval_ds(image_ids, annot_json_path, num_images_per_categ=1000, fetch_unique_annot=True):
    with open(annot_json_path, "r") as f:
        annotation_json = json.load(f)
    
    annotation_im_ids = np.unique([annot["image_id"] for annot in annotation_json["annotations"]])
    image_ids_with_annot = list(filter(lambda im: im in annotation_im_ids, image_ids))
    image_ids_without_annot = list(filter(lambda im: im not in annotation_im_ids, image_ids))

    if fetch_unique_annot:
        image_ids_with_annot = np.unique(image_ids_with_annot).tolist()
    image_ids_without_annot = np.unique(image_ids_without_annot).tolist()

    subsampling_with_annot = np.random.permutation(image_ids_with_annot)[:num_images_per_categ].tolist()
    subsampling_no_annot = np.random.permutation(image_ids_without_annot)[:num_images_per_categ].tolist()
    
    return np.random.permutation(subsampling_with_annot + subsampling_no_annot)
