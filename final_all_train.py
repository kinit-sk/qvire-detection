import json
import model_training

TRAIN_DETECTOR = True

if __name__ == "__main__":
    model_savepath = None

    with open("./dataset_versions.json") as f:
        all_manuscripts = json.load(f)["third_59"]

    if TRAIN_DETECTOR:
        results = model_training.rcnn_train_wrapper(
            train_manuscripts=all_manuscripts,
            test_manuscripts=[],
            model_savepath=model_savepath
        )
    else:
        results = model_training.classifier_train_wrapper(
            train_manuscripts=all_manuscripts,
            test_manuscripts=[],
            model_savepath=model_savepath
        )
