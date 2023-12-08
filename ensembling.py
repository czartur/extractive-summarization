import json
import argparse
from pathlib import Path
from typing import Optional

def read_precictions(prediction_folder : str) -> list:
    # list of predictions for all models
    all_predictions = []

    for item in Path(prediction_folder).iterdir():
        if not item.suffix == ".json": continue  
        prediction = json.load(open(item, "r"))
        print(f"Read prediction for {item.stem}")
        all_predictions.append(prediction) 

    return all_predictions

def majority_vote(all_predictions : list) -> dict:
    # majority vote dict for all dialogs
    test_labels = {}

    for key in all_predictions[0].keys():
        pred_for_key = [preds[key] for preds in all_predictions]
        
        # calculate mean and cap to 1 where val >= 0.5
        major_preds = [1 if sum(x) / len(all_predictions) >= 0.5 else 0 for x in zip(*pred_for_key)]
        
        # assign to test_labels
        test_labels[key] = major_preds

    return test_labels


def main(prediction_folder : str, labels_path : str):

    all_predictions = read_precictions(prediction_folder)
    test_labels = majority_vote(all_predictions)

    json.dump(test_labels, open(labels_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Majority vote over prediction labels")
    
    parser.add_argument("--prediction_folder", type=str, default="prediction", help="Path to the prediction folder")
    parser.add_argument("--labels_path", type=str, default="test_labels.json", help="Path to labels")

    args = parser.parse_args()

    main(args.prediction_folder, args.labels_path)