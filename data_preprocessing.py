import import_data
import dataset

PREPARE_DETECTOR_DATA = True

if __name__ == "__main__":
    data_path = "./data/images"
    annot_json_path = "./data/images/annot.json"
    mask_path = "./data/distance_masks"
    patches_path = "./data/patches"
    
    # prepare data for detector -> detecting quiremarks on strongly downsampled images
    if PREPARE_DETECTOR_DATA:
        import_data.downsample_entire_dataset(
            base_savepath=data_path, downsampled_folder="downsampled", 
            json_filename="annot.json", new_height=800, verbose=True
        )

    # prepare data for classifier -> patchwise classification
    else:
        dataset.create_distance_masks(data_path, annot_json_path, savepath=mask_path, verbose=True)

        dataset.DivideImagesIntoPatches(
            data_path, annot_json_path, additional_masks_path=mask_path, patch_size=(512, 512),
            savepath=patches_path, output_json_name="annot.json", masks_output_folder="distance_masks",
            relative_overlap_size=0.25
        )
