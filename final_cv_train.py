import json
import model_training

CV_TRAIN_DETECTOR = True

if __name__ == "__main__":
    model_savepath = None
    results_savepath = None

    with open("./cv_manuscripts_division.json") as f:
        manus = json.load(f)
    
    all_results = []
    for i in range(5):
        train_manuscripts = manus[f"fold_{i}"]["train"]
        test_manuscripts = manus[f"fold_{i}"]["eval"]

        if CV_TRAIN_DETECTOR:
            results = model_training.rcnn_train_wrapper(
                train_manuscripts,
                test_manuscripts,
                model_savepath=model_savepath,
                results_savepath=results_savepath
            )
        else:
            results = model_training.classifier_train_wrapper(
                train_manuscripts,
                test_manuscripts,
                model_savepath=model_savepath,
                results_savepath=results_savepath
            )
        all_results.append(results)
