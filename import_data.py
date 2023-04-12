import pandas as pd
import numpy as np
import json
import requests
from unidecode import unidecode
import os
import cv2 as cv
import time
from tqdm import tqdm


def create_dataset(
    excel_path, base_savepath="./data", json_filename="annot.json", sheet_names=["withsignatures"],
    existing_annot_json_path=None
):
    for sheet in sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df = df.loc[df["manifest w annotations"].notna()]
        individual_manuscript_links = df["manifest w annotations"].values

        # acquiring images and annotations for each manuscript
        for link in individual_manuscript_links:
            parse_entire_manuscript(link, base_savepath)

    all_annotations = {
        "images": [],
        "categories": [],
        "annotations": []
    }
    if existing_annot_json_path is not None:
        with open(existing_annot_json_path) as f:
            all_annotations = json.load(f)

    jsons_to_delete = []
    for folder in os.listdir(base_savepath):
        if folder == "downsampled":
            continue
        path = os.path.join(base_savepath, folder, "annot.json")
        if os.path.exists(path):
            jsons_to_delete.append(path)
            with open(path, "r") as f:
                new_annot = json.load(f)
            all_annotations = merge_annotations(all_annotations, new_annot)

    with open(os.path.join(base_savepath, json_filename), "w") as f:
        json.dump(all_annotations, f)

    for p in jsons_to_delete:
        os.remove(p)
    

def merge_annotations(all_annotations, new_annotations):
    num_images = len(all_annotations["images"])
    num_annotations = len(all_annotations["annotations"])
    
    for i in range(len(new_annotations["images"])):
        new_annotations["images"][i]["id"] += num_images
    for i in range(len(new_annotations["annotations"])):
        new_annotations["annotations"][i]["image_id"] += num_images
        new_annotations["annotations"][i]["id"] += num_annotations
        
    id_category_mapping = { c["id"]: -1 for c in new_annotations["categories"]}
    cat_ids = np.array([c["id"] for c in all_annotations["categories"]])
    cat_names = np.array([c["name"] for c in all_annotations["categories"]])

    for categ in new_annotations["categories"]:
        indices = np.where(cat_names == categ["name"])[0] if len(cat_names) > 0 else []
        
        if len(indices) > 0:
            id_category_mapping[categ["id"]] = int(cat_ids[indices[0]])
        else:
            new_id = len(all_annotations["categories"])
            all_annotations["categories"].append({
                "id": len(all_annotations["categories"]),
                "name": categ["name"]
            })
            id_category_mapping[categ["id"]] = new_id

    for i in range(len(new_annotations["annotations"])):
        for j in range(len(new_annotations["annotations"][i]["category_id"])):
            old_id = new_annotations["annotations"][i]["category_id"][j]
            new_id = id_category_mapping[old_id]
            new_annotations["annotations"][i]["category_id"][j] = new_id

    all_annotations["images"] += new_annotations["images"]
    all_annotations["annotations"] += new_annotations["annotations"]
    return all_annotations
        

def parse_entire_manuscript(link, base_savepath="./data", parse_annotations=True, debugging=False):
    try:
        response = requests.get(link)
    except:
        return
    if response.status_code != 200:
        time.sleep(5)
        response = requests.get(link)
        if response.status_code != 200:
            print("ERROR")
            raise
    
    metadata = response.json()
    manuscript_name = unidecode(metadata["label"])
    for char in [" ", "/", "\\", ":", "?", "\"", "'", "<", ">", "|"]:
        manuscript_name = manuscript_name.replace(char, "_")

    manuscript_folder_path = os.path.join(base_savepath, manuscript_name)
    manuscript_annotation_path = os.path.join(manuscript_folder_path, "annot.json")

    print(f"Acquiring data from manuscript: {manuscript_name}")

    if os.path.exists(manuscript_folder_path):
        return
    
    os.makedirs(manuscript_folder_path, exist_ok=True)
    annotation_json = {
        "images": [],
        "categories": [],
        "annotations": [],
    }

    annotation_iter = 0
    image_iter = 0
    images_metadata = response.json()["sequences"][0]["canvases"]
    for image_metadata in tqdm(images_metadata, total=len(images_metadata)):
        filename = f"image_{image_iter}.jpg"
        fullpath = os.path.join(manuscript_folder_path, filename)
            
        new_bboxes, new_categories, (height, width) = parse_one_image(image_metadata, fullpath, parse_annotations=parse_annotations)

        if height != -1 and width != -1:
            annotation_json["images"].append({
                "id": image_iter,
                "file_name": filename,
                "folder_path": manuscript_name,
                "width": int(width),
                "height": int(height)
            })
            
            #process categories
            new_cat_ids = []
            for categ_per_bbox in new_categories:
                new_cat_ids.append([])
                for categ in categ_per_bbox:
                    cat_ids = np.array([cat["id"] for cat in annotation_json["categories"]])
                    cat_names = np.array([cat["name"] for cat in annotation_json["categories"]])

                    indices = np.where(cat_names == categ)[0] if len(cat_names) > 0 else []
                    if len(indices) == 0:
                        id_to_add = int(cat_ids[-1] + 1 if len(cat_ids) > 0 else 0)
                        annotation_json["categories"].append({
                            "id": id_to_add,
                            "name": categ
                        })
                    else:
                        id_to_add = int(cat_ids[indices[0]])

                    new_cat_ids[-1].append(id_to_add)

            #process annotations
            for i_bbox, bbox in enumerate(new_bboxes): 
                annotation_json["annotations"].append({
                    "bbox": bbox,
                    "id": annotation_iter + i_bbox,
                    "category_id": new_cat_ids[i_bbox],
                    "image_id": image_iter,
                    "iscrowd": False,
                    "area": bbox[2] * bbox[3]
                })
            annotation_iter += len(new_bboxes)
            image_iter += 1

            if debugging and image_iter == 10:
                break

    with open(manuscript_annotation_path, "w") as f:
        json.dump(annotation_json, f)


def parse_one_image(image_metadata, savepath, parse_annotations=True):
    inner_metadata = image_metadata["images"][0]["resource"]
    fake_height, fake_width = inner_metadata["height"], inner_metadata["width"]
    img_link = inner_metadata["@id"]

    try:
        response = requests.get(img_link)
    except:
        print("ERROR")
        raise
    if response.status_code != 200:
        time.sleep(5)
        response = requests.get(img_link)
        if response.status_code != 200:
            print("Invalid image link")
            return [], [], (-1, -1)

    with open(savepath, "wb") as f:
        f.write(response.content)

    img = cv.imread(savepath)
    real_height, real_width = img.shape[0], img.shape[1]
    height_multiplier = real_height / fake_height
    width_multiplier = real_width / fake_width

    if parse_annotations:
        annot_link = image_metadata["otherContent"][0]["@id"]
        bboxes, labels = get_image_annotations(annot_link, height_multiplier, width_multiplier)
    else:
        bboxes, labels = [], []

    return bboxes, labels, (real_height, real_width)


def get_image_annotations(link, height_multiplier, width_multiplier):
    try:
        response = requests.get(link)
    except:
        print("ERROR")
        raise
    if response.status_code != 200:
        time.sleep(5)
        response = requests.get(link)
        if response.status_code != 200:
            print("Invalid image annotations link")
            return [], []

    metadata = response.json()

    all_bbox_coordinates = []
    labels = []

    for annot_metadata in metadata["resources"]:
        bbox_labels = []
        for labels_metadata in annot_metadata["resource"]:
            bbox_labels.append(labels_metadata["http://dev.llgc.org.uk/sas/full_text"])
        labels.append(bbox_labels)

        bbox_string = annot_metadata["on"][0]["selector"]["default"]["value"]
        coords = bbox_string.split(",")
        coords[0] = coords[0].split("=")[1]

        x = round(int(coords[0]) * width_multiplier)
        y = round(int(coords[1]) * height_multiplier)
        w = round(int(coords[2]) * width_multiplier)
        h = round(int(coords[3]) * height_multiplier)
        all_bbox_coordinates.append([x,y,w,h])

    return all_bbox_coordinates, labels


def downsample_entire_dataset(base_savepath="./data", downsampled_folder="downsampled", 
                              json_filename="annot.json", new_height=800, verbose=False):
    os.makedirs(os.path.join(base_savepath, downsampled_folder), exist_ok=True)

    with open(os.path.join(base_savepath, json_filename), "r") as f:
        annotation_json = json.load(f)

    for it, image_metadata in tqdm(enumerate(annotation_json["images"]), total=len(annotation_json["images"]), disable=verbose==False):
        path = os.path.join(
            base_savepath, 
            image_metadata["folder_path"], 
            image_metadata["file_name"]
        )
        image = cv.imread(path)
        height, width = image.shape[0], image.shape[1]

        downsample_scale = new_height / height
        new_width = int(width * downsample_scale)

        downsampled_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
        new_folder_to_create = os.path.join(
            base_savepath, 
            downsampled_folder,
            image_metadata["folder_path"]
        )
        os.makedirs(new_folder_to_create, exist_ok=True)
        new_path = os.path.join(new_folder_to_create, image_metadata["file_name"])
        cv.imwrite(new_path, downsampled_image)
        
        annotation_json["images"][it]["height"] = downsampled_image.shape[0]
        annotation_json["images"][it]["width"] = downsampled_image.shape[1]

        annot_image_ids = np.array([annot["image_id"] for annot in annotation_json["annotations"]])
        indices = np.where(annot_image_ids == image_metadata["id"])[0]

        for idx in indices:
            annot = annotation_json["annotations"][idx]

            bbox = annot["bbox"]
            x = int(bbox[0] * downsample_scale)
            y = int(bbox[1] * downsample_scale)
            w = int(bbox[2] * downsample_scale)
            h = int(bbox[3] * downsample_scale)
            bbox = [x,y,w,h]

            annotation_json["annotations"][idx]["bbox"] = [x,y,w,h]
            annotation_json["annotations"][idx]["area"] = w*h

    new_json_path = os.path.join(base_savepath, downsampled_folder, json_filename)
    with open(new_json_path, "w") as f:
        json.dump(annotation_json, f)
