import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import json
import cv2 as cv
from sklearn.mixture import GaussianMixture
from numpy import ma
from tqdm import tqdm
from collections.abc import Iterable
from PIL import Image

import utils


class BasicTransform:
    CROP_SCALE_PARAMS = (0.75, 1)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, spatial_augm=False, mean=None, std=None, apply_normalization=False, 
                 crop_scale_params=None, downsample_size=None, color_augmentation=False, 
                 color_jitter_kwargs={}, flipping=False):

        self.spatial_augm = spatial_augm
        self.flipping = flipping
        self.crop_scale_params = crop_scale_params

        self.color_augm_pipeline = None
        self.color_augmentation = color_augmentation
        self.color_jitter_kwargs = color_jitter_kwargs

        self.apply_normalization = apply_normalization
        self.downsample_size = downsample_size

        mean = mean if mean is not None else BasicTransform.MEAN
        std = std if std is not None else BasicTransform.STD        
        if self.apply_normalization:
            self.normalization_step = transforms.Normalize(mean, std)

        self.input_pipeline = transforms.Lambda(utils.image_to_tensor)

        if self.color_augmentation:
            if len(color_jitter_kwargs.keys()) > 0:
                self.color_augm_pipeline = transforms.ColorJitter(**color_jitter_kwargs)
            else:
                self.color_augm_pipeline = transforms.ColorJitter(0.15, 0.15, 0.15, 0.025)
        
    def __call__(self, data):
        data["X"] = self.input_pipeline(data["X"])        
        data = self.augm_spatial_function(data)

        if self.color_augmentation:
            mask = data["X"][3:]
            img = self.color_augm_pipeline(data["X"][:3])
            data["X"] = torch.vstack([img, mask])
        if self.apply_normalization:
            mask = data["X"][3:]
            img = self.normalization_step(data["X"][:3])
            data["X"] = torch.vstack([img, mask])

        return data

    def augm_spatial_function(self, data):
        gaussian_kernel = [0, 3]
        crop_scale = BasicTransform.CROP_SCALE_PARAMS
        if self.crop_scale_params is not None: 
            crop_scale = self.crop_scale_params

        random_height = crop_scale[0] + \
            np.random.rand() * (crop_scale[1] - crop_scale[0])
        random_width = crop_scale[0] + \
            np.random.rand() * (crop_scale[1] - crop_scale[0])

        bbox_image = BasicTransform.bboxes_as_image(data["boxes"], data["X"].shape[-2:])
        if len(bbox_image) > 0:
            mask_image = bbox_image.amax(dim=0)
        else:
            mask_image = np.zeros(bbox_image.shape[1:])

        if self.downsample_size is None:
            resize_size = mask_image.shape
        else:
            if isinstance(self.downsample_size, Iterable) == False:
                self.downsample_size = [self.downsample_size, self.downsample_size]
            resize_size = self.downsample_size

        if self.spatial_augm:
            crop_coord = self.compute_crop_coordinates(mask_image, random_height, random_width)
            gaussian_idx = np.random.randint(len(gaussian_kernel))

            rand_kwargs = {
                "crop": {
                    "left": crop_coord[0],
                    "top": crop_coord[1],
                    "width": crop_coord[2],
                    "height": crop_coord[3]
                },
                "resize": {
                    "size": resize_size
                }
            }
            if gaussian_idx != 0:
                rand_kwargs["gaussian_blur"] = {
                    "kernel_size": gaussian_kernel[gaussian_idx]
                }
            if self.flipping:
                hflip_rand = np.random.rand()
                vflip_rand = np.random.rand()

                if hflip_rand > 0.5:
                    rand_kwargs["hflip"] = {}
                if vflip_rand > 0.5: 
                    rand_kwargs["vflip"] = {}
        elif resize_size != mask_image.shape:
            rand_kwargs = {
                "resize": {
                    "size": resize_size
                }
            }
        else:
            return data 

        data["X"] = self._functional_pipeline(data["X"], rand_kwargs)

        if len(bbox_image) > 0:
            invalid_operations = ["gaussian_blur"]
            for op in invalid_operations:
                if op in rand_kwargs.keys():
                    del rand_kwargs[op]

            bbox_image = self._functional_pipeline(bbox_image, rand_kwargs)
            data["boxes"], data["labels"] = BasicTransform.bboxes_from_image(bbox_image, data["labels"])

        return data

    def compute_crop_coordinates(self, mask_image, random_height, random_width):
        pos = np.where(mask_image == 1)
        
        if len(pos[0]) == 0:
            left_margin = mask_image.shape[1] // 2
            top_margin = mask_image.shape[0] // 2
            right_margin = mask_image.shape[1] // 2 + 1
            bottom_margin = mask_image.shape[0] // 2 + 1
        else:
            left_margin = int(np.min(pos[1]))
            top_margin = int(np.min(pos[0]))
            right_margin = mask_image.shape[1] - int(np.max(pos[1]))
            bottom_margin = mask_image.shape[0] - int(np.max(pos[0]))

        left_crop, right_crop = self._helper_crop_coord(
            left_margin, right_margin, random_width, mask_image.shape[1]
        )
        top_crop, bottom_crop = self._helper_crop_coord(
            top_margin, bottom_margin, random_height, mask_image.shape[0]
        )

        x0 = left_crop
        x1 = mask_image.shape[1] - right_crop
        y0 = top_crop
        y1 = mask_image.shape[0] - bottom_crop
        return x0, y0, x1-x0, y1-y0

    def _helper_crop_coord(self, margin1, margin2, random_size, whole_size):
        pixels_to_spare = min(
            int((1-random_size)*whole_size),
            margin1 + margin2
        )
        if pixels_to_spare == 0:
            return 0, 0

        if margin1 <= margin2:
            first_is_1 = True
            first_margin = margin1
        else:
            first_is_1 = False
            first_margin = margin2

        max_random_value = first_margin if first_margin < pixels_to_spare else pixels_to_spare
        offset1 = np.random.randint(
            max_random_value) if max_random_value > 0 else 0
        offset2 = pixels_to_spare - offset1

        if first_is_1:
            return offset1, offset2
        return offset2, offset1

    def _functional_pipeline(self, x, rand_kwargs):
        for key in rand_kwargs.keys():
            func = getattr(transforms.functional, key)
            x = func(x, **rand_kwargs[key])
        return x

    @staticmethod
    def inverse_normalize(X):
        inverse_norm = transforms.Compose([ 
            transforms.Normalize(
                mean = [0, 0, 0],
                std = (1 / np.array(BasicTransform.STD)).tolist()
            ),
            transforms.Normalize(
                mean = (np.array(BasicTransform.MEAN) * (-1)).tolist(),
                std = [1, 1, 1]
            ),
        ])
        return inverse_norm(X)

    @staticmethod
    def bboxes_as_image(bbox, shape, coco_format=True):
        bbox_image = torch.zeros(len(bbox), *shape)

        for i, box in enumerate(bbox):
            if coco_format:
                x, y, w, h = box.int()
                bbox_image[i][y: y+h+1, x: x+w+1] = 1
            else:
                x0, y0, x1, y1 = box.int()
                bbox_image[i][y0: y1, x0: x1] = 1
        return bbox_image

    @staticmethod
    def bboxes_from_image(bbox_image, labels):
        bbox = []
        new_labels = []

        for i, channel, in enumerate(bbox_image):
            if (channel == 1).sum() < 4:
                continue
            
            pos = np.where(channel == 1)
            x0 = np.min(pos[1])
            y0 = np.min(pos[0])
            x1 = np.max(pos[1])
            y1 = np.max(pos[0])

            if x0 == x1 or y0 == y1:
                continue

            bbox.append([x0, y0, x1 - x0, y1 - y0])
            new_labels.append(labels[i])

        bbox = torch.tensor(bbox)
        new_labels = torch.tensor(new_labels, dtype=torch.long)
        return bbox, new_labels


class MyDataset(Dataset):
    def __init__(self, image_ids, data_path, annot_json_path, transform_kwargs={}, classification_task=False,
                 detection_only=True, return_coco_bboxes=False, additional_mask_path=None, undersample_no_annotations=False,
                 detr_feature_extractor=None, new_distance_mask=True, out_of_page_mask_value=0):
        super().__init__()

        self.image_ids = image_ids
        self.data_path = data_path
        self.annot_json_path = annot_json_path
        self.transform_kwargs = transform_kwargs
        self.return_coco_bboxes = return_coco_bboxes
        self.additional_mask_path = additional_mask_path
        self.classification_task = classification_task
        self.detr_feature_extractor = detr_feature_extractor
        self.new_distance_mask = new_distance_mask
        self.out_of_page_mask_value = out_of_page_mask_value
        self.undersample_no_annotations = undersample_no_annotations
    
        self.transform = BasicTransform(**transform_kwargs)

        with open(self.annot_json_path, "r", encoding="utf-8") as f:
            self.annotation_json = json.load(f)

        self.detection_only = detection_only
        if self.detection_only:
            self.classes = [1]
        else:
            self.classes = [cat["id"] for cat in self.annotation_json["categories"]]

        if self.undersample_no_annotations == False:
            if type(image_ids) == int and image_ids == -1:
                self.image_ids = all_image_ids
                self.image_metadata = self.annotation_json["images"]
            else:
                all_image_ids = {im["id"]: idx for idx, im in enumerate(self.annotation_json["images"])}
                indices = np.array([all_image_ids[id] for id in self.image_ids])
                if len(indices):
                    self.image_metadata = np.array(self.annotation_json["images"])[indices].tolist()
                else:
                    self.image_metadata = []
            
            self.all_ds_annot_metadata = list(filter(
                lambda annot: annot["image_id"] in self.image_ids, 
                self.annotation_json["annotations"]
            ))

        else:
            ids_with_annot = np.unique([annot["image_id"] for annot in self.annotation_json["annotations"]])
            
            self.our_ids_with_annot = self.image_ids[np.isin(self.image_ids, ids_with_annot)]
            self.our_ids_without_annot = self.image_ids[~np.isin(self.image_ids, ids_with_annot)]
            self.our_ids_without_annot = np.random.permutation(self.our_ids_without_annot)

            self.reset_counter = 0

            self.reset_epoch()

    def reset_epoch(self):
        if self.undersample_no_annotations:
            x = self.reset_counter
            new_ids_without_annot = self.our_ids_without_annot[
                x*len(self.our_ids_with_annot): (x+1)*len(self.our_ids_with_annot)
            ]

            if len(new_ids_without_annot) == 0:
                self.reset_counter = 0
                x = self.reset_counter

                self.our_ids_without_annot = np.random.permutation(self.our_ids_without_annot)
                new_ids_without_annot = self.our_ids_without_annot[
                    x*len(self.our_ids_with_annot): (x+1)*len(self.our_ids_with_annot)
                ]

            new_epoch_ids = np.hstack([self.our_ids_with_annot, new_ids_without_annot])
            
            all_image_ids = {im["id"]: idx for idx, im in enumerate(self.annotation_json["images"])}
            indices = np.array([all_image_ids[id] for id in new_epoch_ids])
            if len(indices):
                self.image_metadata = np.array(self.annotation_json["images"])[indices].tolist()
            else:
                self.image_metadata = []
            
            self.all_ds_annot_metadata = list(filter(
                lambda annot: annot["image_id"] in new_epoch_ids, 
                self.annotation_json["annotations"]
            ))
            self.reset_counter += 1

    def __getitem__(self, idx):    
        image_metadata = self.image_metadata[idx]

        path_to_image = os.path.join(
            self.data_path, 
            image_metadata["folder_path"], 
            image_metadata["file_name"]
        )

        image = cv.imread(path_to_image)
        bboxes, class_labels = self.get_annotations(image_metadata["id"], image.shape[1], image.shape[0])

        if self.additional_mask_path:
            mask_name = image_metadata["file_name"]                
            
            if self.new_distance_mask:
                mask_name = mask_name[:mask_name.rfind(".")] + ".png"

            path_to_mask = os.path.join(
                self.additional_mask_path, 
                image_metadata["folder_path"], 
                mask_name
            )
            mask = cv.imread(path_to_mask, -1)
            
            if self.new_distance_mask:
                try:
                    mask = np.float32(mask)
                    mask[mask == 256] = self.out_of_page_mask_value
                except:
                    mask = np.zeros(image.shape[:-1], dtype=np.float32)
            image = np.concatenate([image, mask[..., None]], axis=-1)
        
        return_object = {
            "X": image,
            "boxes": bboxes,
            "labels": class_labels,
            "misc": {
                "image_id": image_metadata["id"]
            }
        }

        if self.transform:
            return_object = self.transform(return_object)
        if self.return_coco_bboxes == False and len(return_object["boxes"]) > 0:
            return_object["boxes"][:, 2:] += return_object["boxes"][:, :2]
        if len(return_object["boxes"]) == 0:
            return_object["boxes"] = torch.zeros(0, 4)
            return_object["labels"] = torch.zeros(0, dtype=torch.long)

        if self.classification_task:
            y = torch.Tensor([len(return_object["labels"]) > 0]).to(dtype=torch.long)
            return return_object["X"], y

        if self.detr_feature_extractor is None:
            return return_object

        image = (return_object["X"] * 255).to(dtype=torch.uint8).numpy()
        image = np.transpose(image, (1,2,0))
        image_id = image_metadata["id"] 

        annotations = []
        for box, label in zip(return_object["boxes"], return_object["labels"]):
            annotations.append({
                "bbox": box.tolist(),
                "category_id": label.item(),
                "image_id": image_id,
                "iscrowd": False,
                "area": (box[2] * box[3]).item()
            })
        
        target = {'image_id': image_id, 'annotations': annotations}
        encoding = self.detr_feature_extractor(images=image, annotations=target, return_tensors="pt")
        
        pixel_values = encoding["pixel_values"][0]
        target = encoding["labels"][0]
        return pixel_values, target

    def __len__(self):
        return len(self.image_metadata)

    def get_annotations(self, image_id, width, height):    
        bboxes_objs = list(filter(
            lambda bbox: bbox["image_id"] == image_id,
            self.all_ds_annot_metadata
        ))
        bboxes = torch.tensor([bbox["bbox"] for bbox in bboxes_objs])

        if len(bboxes) > 0:
            mask_a = ((bboxes[:, 0] + bboxes[:, 2]) > width) | ((bboxes[:, 1] + bboxes[:, 3]) > height)
            mask_b = (bboxes[:, 0] < 0) | (bboxes[:, 1] < 0) | (bboxes[:, 2] <= 0) | (bboxes[:, 3] <= 0)
            mask = ~(mask_a | mask_b)

            indices = torch.where(mask)[0]
            bboxes = bboxes[indices]

        if self.detection_only:
            class_labels = torch.ones(len(bboxes), dtype=torch.long)
        else:
            raise "Not implemented yet"
        
        return bboxes, class_labels


def create_distance_masks(
        data_path, annot_json_path, savepath, apply_bilateral_filter=True, gmm_max_number_of_pixels_processed=10_000_000,
        gmm_image_samples=10, gmm_components=4, rgb_percentile_thresholds=(20, 80), bilateral_kernel_size=50, verbose=False
    ):
    
    with open(annot_json_path, "r") as f:
        annotation_json = json.load(f)
    manuscript_names = np.unique([im["folder_path"] for im in annotation_json["images"]]).tolist()

    for i_manum, manuscript in enumerate(manuscript_names):
        if verbose:
            print(f"CREATING MASK FOR MANUSCRIPT: {i_manum+1}/{len(manuscript_names)}")
        manuscript_path = os.path.join(data_path, manuscript)
        
        if os.path.exists(os.path.join(savepath, manuscript)):
            continue
        os.makedirs(os.path.join(savepath, manuscript))

        rgb_thresholds = _train_gmm(
            manuscript_path, apply_bilateral_filter, 
            gmm_image_samples, gmm_components, rgb_percentile_thresholds,
            gmm_max_number_of_pixels_processed
        )

        images_metadata = list(filter(
            lambda im: im["folder_path"] == manuscript,
            annotation_json["images"]
        ))

        for im_meta in tqdm(images_metadata):
            relative_path_no_ext = os.path.join(im_meta["folder_path"], im_meta["file_name"][:im_meta["file_name"].rfind(".")])
            load_fullpath = os.path.join(data_path, relative_path_no_ext + ".jpg")
            save_fullpath = os.path.join(savepath, relative_path_no_ext + ".png")

            img = cv.imread(load_fullpath)
            h, w = img.shape[:2]
            avg_size = (h+w) / 2

            if apply_bilateral_filter:
                img = cv.bilateralFilter(img, bilateral_kernel_size, 35, 35)

            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
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
            kernel_bigger_unit = 103.6
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

            proximity_to_edges = utils._compute_proximity_to_edges(watershed_segm)

            #TO READ uint16 PNG file -> cv2.imread(path, -1)
            cv.imwrite(save_fullpath, proximity_to_edges)



def _train_gmm(manuscript_path, apply_bilateral_filter, gmm_image_samples=10, gmm_components=4, rgb_percentile_thresholds=(5, 95),
                gmm_max_number_of_pixels_processed=10_000_000, bilateral_kernel_size=50):
    num_of_images = len(os.listdir(manuscript_path))
    random_pages = np.random.randint(10, num_of_images - 10, size=gmm_image_samples)

    img_names = [
        f"image_{i}.jpg" for i in random_pages
    ]

    all_imgs = []
    data = []
    for img_name in img_names:
        fullpath = os.path.join(manuscript_path, img_name)    
        img = cv.imread(fullpath)
        h, w = img.shape[:2]

        if apply_bilateral_filter:
            img = cv.bilateralFilter(img, bilateral_kernel_size, 35, 35)

        all_imgs.append(img)
        data.append(img.reshape(-1, 3))

    data = np.vstack(data)
    train_data = data[:gmm_max_number_of_pixels_processed]
    gmm = GaussianMixture(n_components=gmm_components, covariance_type="tied")
    gmm.fit(train_data)
    
    labels = np.zeros(len(data))
    for i in range(0, len(data), 10_000_000):
        labels[i: i + 10_000_000] = gmm.predict(data[i: i + 10_000_000])

    rgb_thresholds = utils._compute_rgb_distributions(labels, all_imgs, gmm_components, rgb_percentile_thresholds)

    return rgb_thresholds


class DivideImagesIntoPatches:
    def __init__(self, data_path, annot_json_path, additional_masks_path=None, patch_size=(1066, 800),
                 savepath="./patches", output_json_name="patches_annot.json", masks_output_folder="distance_masks",
                 relative_overlap_size=0.5, include_partial_bboxes=False, include_patches_with_no_bboxes=True,
                 verbose=True, divide_mask_only=False, existing_json_annot_path=None):

        self.data_path = data_path
        self.annot_json_path = annot_json_path
        self.additional_masks_path = additional_masks_path
        self.savepath = savepath
        self.output_json_name = output_json_name
        self.masks_output_folder = masks_output_folder

        self.patch_size = patch_size
        self.relative_overlap_size = relative_overlap_size
        self.include_partial_bboxes = include_partial_bboxes
        self.include_patches_with_no_bboxes = include_patches_with_no_bboxes
        self.verbose = verbose
        self.divide_mask_only = divide_mask_only

        self.annot_id_iter = 0
        self.crop_id_iter = 0

        if isinstance(patch_size, Iterable) == False:
            self.patch_size = [patch_size, patch_size]

        with open(self.annot_json_path, "r", encoding="utf-8") as f:
            self.annotation_json = json.load(f)

        self.json_data = {
            "images": [],
            "categories": self.annotation_json["categories"],
            "annotations": []
        }
        if existing_json_annot_path is not None:
            with open(existing_json_annot_path) as f:
                self.json_data = json.load(f)   
            self.annot_id_iter = self.json_data["annotations"][-1]["id"] + 1
            self.crop_id_iter = self.json_data["images"][-1]["id"] + 1

    def divide_dataset(self):
        manuscript_names = np.unique([im["folder_path"] for im in self.annotation_json["images"]]).tolist()

        for manu_it, manuscript in enumerate(manuscript_names):
            if self.verbose:
                print(f"MANUSCRIPT: {manu_it+1}/{len(manuscript_names)}")
            if os.path.exists(os.path.join(self.savepath, manuscript)):
                continue

            manuscript_images = list(filter(
                lambda im: im["folder_path"] == manuscript,
                self.annotation_json["images"]
            ))

            for image_metadata in tqdm(manuscript_images, disable=(self.verbose == False)):
                new_image_list, new_annot_list = self.divide_image(image_metadata)

                self.json_data["images"].extend(new_image_list)
                self.json_data["annotations"].extend(new_annot_list)

        self._create_json()

    def divide_image(self, image_metadata):
        json_image_data = []
        json_annot_data = []

        old_img_id = image_metadata["id"]
        folder_name = image_metadata["folder_path"]
        filename = image_metadata["file_name"]
        no_ext_filename = filename[:filename.rfind(".")]

        img_bboxes = None
        image = None
        if self.divide_mask_only == False:
            all_bboxes = self.annotation_json["annotations"]
            img_bboxes = list(filter(lambda bbox: bbox["image_id"] == old_img_id, all_bboxes))

            image_savepath = os.path.join(self.savepath, folder_name)
            os.makedirs(image_savepath, exist_ok=True)

            image = Image.open(os.path.join(self.data_path, folder_name, filename))

        distance_mask = None
        if self.additional_masks_path is not None:
            distance_mask = cv.imread(os.path.join(self.additional_masks_path, folder_name, no_ext_filename + ".png"), -1)
    
            mask_savepath = os.path.join(self.savepath, self.masks_output_folder, folder_name)
            os.makedirs(mask_savepath, exist_ok=True)

        image_width, image_height = image.size if self.divide_mask_only == False else distance_mask.shape[::-1]
        x_offset = (
            self.patch_size[0] -
            int(self.relative_overlap_size*self.patch_size[0])
        )
        y_offset = (
            self.patch_size[1] -
            int(self.relative_overlap_size*self.patch_size[1])
        )

        crop_id = 0
        for y in range(0, image_height, y_offset):
            for x in range(0, image_width, x_offset):
                crop_coordinates = self._contain_crop_within_image(x, y, self.patch_size,
                                                                   image_width, image_height)

                if self.divide_mask_only:
                    if distance_mask is not None:
                        crop_name_mask = f"{no_ext_filename}-crop_{crop_id}.png"

                        x1, y1, x2, y2 = crop_coordinates
                        crop_mask = distance_mask[y1: y2, x1: x2]
                        cv.imwrite(os.path.join(mask_savepath, crop_name_mask), crop_mask)

                        crop_id += 1
                    else:
                        raise
                else:
                    new_annot_data = self.divide_annotation(img_bboxes, crop_coordinates)
                    json_annot_data.extend(new_annot_data)

                    if self.include_patches_with_no_bboxes or len(new_annot_data) > 0:
                        crop_name_img = f"{no_ext_filename}-crop_{crop_id}.jpg"
                        crop_name_mask = f"{no_ext_filename}-crop_{crop_id}.png"
                    
                        json_image_data.append({
                            "file_name": crop_name_img,
                            "folder_path": image_metadata["folder_path"],
                            "id": self.crop_id_iter,
                            "old_image_id": old_img_id,
                            "crop_id": crop_id,
                            "width": crop_coordinates[2] - crop_coordinates[0],
                            "height": crop_coordinates[3] - crop_coordinates[1]
                        })

                        crop = image.crop(box=(crop_coordinates))
                        crop.save(os.path.join(image_savepath, crop_name_img))

                        if distance_mask is not None:
                            x1, y1, x2, y2 = crop_coordinates
                            crop_mask = distance_mask[y1: y2, x1: x2]
                            cv.imwrite(os.path.join(mask_savepath, crop_name_mask), crop_mask)

                
                        self.crop_id_iter += 1
                        crop_id += 1

        return json_image_data, json_annot_data

    def _contain_crop_within_image(self, x, y, patch_size, image_width, image_height):
        new_x = x
        new_y = y

        if x+patch_size[0] > image_width:
            diff = (x+patch_size[0]) - image_width
            new_x = x - diff
        if y+patch_size[1] > image_height:
            diff = (y+patch_size[1]) - image_height
            new_y = y - diff

        return new_x, new_y, new_x+patch_size[0], new_y+patch_size[1]

    def divide_annotation(self, img_bboxes, crop_coordinates):
        annot_data = []

        cx0, cy0, cx1, cy1 = crop_coordinates
        coords = [bbox["bbox"] for bbox in img_bboxes]

        if len(coords) == 0:
            return annot_data

        # change [x,y,w,h] bbox format to [x0,y0,x1,y1] for convience
        coords = np.array(coords)
        coords[:, 2:] += coords[:, :2]

        x0, y0, x1, y1 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]

        valid_dimensions = (x0 != x1) & (y0 != y1)
        point0_within = (x0 >= cx0) & (x0 <= cx1) & (
            y0 >= cy0) & (y0 <= cy1) & valid_dimensions
        point1_within = (x1 >= cx0) & (x1 <= cx1) & (
            y1 >= cy0) & (y1 <= cy1) & valid_dimensions
        both_points_within = point0_within & point1_within

        indices_to_bboxes_to_save = []

        indices = np.where(both_points_within)[0].tolist()
        indices_to_bboxes_to_save += indices
        for idx in indices:
            coords[idx, :2] -= crop_coordinates[:2]
            coords[idx, 2:] -= crop_coordinates[:2]

        point0_within = point0_within ^ both_points_within
        point1_within = point1_within ^ both_points_within

        if self.include_partial_bboxes:
            indices = np.where(point0_within)[0].tolist()
            indices_to_bboxes_to_save += indices

            for idx in indices:
                coords[idx, :2] -= crop_coordinates[:2]
                coords[idx, 2] = np.clip(
                    coords[idx, 2] - cx0, a_min=-np.inf, a_max=(cx1 - cx0))
                coords[idx, 3] = np.clip(
                    coords[idx, 3] - cy0, a_min=-np.inf, a_max=(cy1 - cy0))

            indices = np.where(point1_within)[0].tolist()
            indices_to_bboxes_to_save += indices

            for idx in indices:
                coords[idx, :2] = np.clip(
                    coords[idx, :2] - crop_coordinates[:2], a_min=0, a_max=np.inf)
                coords[idx, 2:] -= crop_coordinates[:2]

        # change [x0,x1,y0,y1] bbox format back to [x,y,w,h] -> COCO standard
        coords[:, 2:] -= coords[:, :2]

        for idx in indices_to_bboxes_to_save:
            bbox = coords[idx].tolist()
            area = int(bbox[2] * bbox[3])

            annot_data.append({
                "bbox": bbox,
                "category_id": img_bboxes[idx]["category_id"],
                "id": self.annot_id_iter,
                "image_id": self.crop_id_iter,
                "old_image_id": img_bboxes[idx]["image_id"],
                "old_id": img_bboxes[idx]["id"],
                "iscrowd": False,
                "area": area
            })
            self.annot_id_iter += 1

        return annot_data

    def _create_json(self):
        with open(os.path.join(self.savepath, self.output_json_name), "w", encoding="utf-8") as f:
            json.dump(self.json_data, f)
