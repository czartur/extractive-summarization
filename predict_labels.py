import src.custom_utils as cu
import torch
import json
from transformers import AutoModel, BertTokenizerFast
import argparse
from typing import Optional

device = "cuda" if torch.cuda.is_available() else "cpu"

# base model for embedding
base_model_name = 'bert-base-uncased'
base_model = AutoModel.from_pretrained(base_model_name)
tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
max_seq_len = 80

# format test input data
def predict_labels(test_sentences : dict, test_speakers : dict, model : AutoModel, device : str = device) -> dict:
    model.to(device)
    test_data = {}
    for id in test_sentences:
        test_data[id] = cu.format_input(test_sentences[id], test_speakers[id], tokenizer, max_seq_len, device)

    model.eval()
    test_labels = {}

    for id in test_sentences.keys():
        out = model(**test_data[id])
        pred = out.argmax(dim=1)
        test_labels[id] = pred.cpu().detach().tolist()

    return test_labels


def main(test_folder_path : str, model_path : str, labels_path : str):
    # load model 
    model = torch.load(model_path)
    
    # load test data
    test_sentences, test_speakers, _  = cu.read_data_by_ID(test_folder_path, combine = False)

    try:
        test_labels = predict_labels(test_sentences, test_speakers, model)
    except RuntimeError as e:
        if device == "cuda" and "CUDA out of memory" in str(e):
            print("Insufficient memory on gpu, predictions will be calculated using cpu")
            test_labels = predict_labels(test_sentences, test_speakers, model, device="cpu")
        else: raise e

    json.dump(test_labels , open("test_labels.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test_labels.json for submission")
    
    parser.add_argument("--test_folder_path", type=str, default="test", help="Path to the test folder")
    parser.add_argument("--model_path", type=str, default="best_model.pt", help="Path to the model")
    parser.add_argument("--labels_path", type=str, default="test_labels.json", help="Path to labels")

    args = parser.parse_args()

    main(args.test_folder_path, args.model_path, args.labels_path)