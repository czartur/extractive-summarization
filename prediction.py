import src.Utils as cu
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
def predict_labels(test_sentences : dict, test_speakers : dict, in_degrees : dict, out_degrees : dict, model : AutoModel, device : str = device) -> dict:
    model.to(device)
    test_data = {}
    for id in test_sentences:
        test_data[id] = cu.format_input(test_sentences[id], test_speakers[id], in_degrees[id], out_degrees[id], tokenizer, max_seq_len, device)

    model.eval()
    test_labels = {}

    for id in test_sentences.keys():
        out = model(test_data[id])
        pred = out.argmax(dim=1)
        test_labels[id] = pred.cpu().detach().tolist()
        print(id)
    return test_labels


def main(test_folder_path : str, model_path : str, labels_path : str):
    # load model 
    model = torch.load(model_path)
    
    # load test data
    test_sentences, test_speakers, test_edges = cu.read_data_by_ID(test_folder_path, combine = False)
    dialog_ids = test_sentences.keys()

    # capture in and out degrees
    in_degrees = {}
    out_degrees = {}
    for id in dialog_ids:
        n_nodes = len(test_sentences[id])
        in_degree = [0]*n_nodes
        out_degree = [0]*n_nodes

        for edge in test_edges[id]:
            out_degree[edge[0]] += 1
            in_degree[edge[2]] += 1

        in_degrees[id] = in_degree
        out_degrees[id] = out_degree

    try:
        test_labels = predict_labels(test_sentences, test_speakers, in_degrees, out_degrees, model)
    except RuntimeError as e:
        if device == "cuda" and "CUDA out of memory" in str(e):
            print("Insufficient memory on gpu, predictions will be calculated using cpu")
            test_labels = predict_labels(test_sentences, test_speakers, in_degrees, out_degrees, model, device="cpu")
        else: raise e

    json.dump(test_labels , open(labels_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test_labels.json for submission")

    parser.add_argument("--test_folder", type=str, default="test", help="Path to the test folder")
    parser.add_argument("--model", type=str, default="best_model.pt", help="Path to the model")
    parser.add_argument("--test_labels", type=str, default="test_labels.json", help="Path to test labels")

    args = parser.parse_args()

    main(args.test_folder, args.model, args.test_labels)