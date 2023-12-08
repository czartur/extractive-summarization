import torch 
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import F1Score
from transformers import AutoModel, BertTokenizerFast

import argparse

import src.Utils as cu
from src.Model import MLP_FT
from src.Dataset import data_loader
from src.Trainer import train_model 
from src.Tuner import HyperTuner

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

## define base mode (embedder)
base_model_name = 'bert-base-uncased'
base_model = AutoModel.from_pretrained(base_model_name)
tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
max_seq_len = 80

def read_dataset(training_folder : str, training_labels_path : str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # read training data
    df = cu.read_data_to_dataframe("training", "training_labels.json")

    df['sentences'] = df['speakers'] + ": " + df['sentences']
    train, valid = train_test_split(df, test_size=0.2, random_state=69, stratify=df.labels)

    print(f"Train: {len(train)}\nValid: {len(valid)}")
    return train, valid

def train_from_params(model_params : dict, training_params : dict, train : pd.DataFrame, valid : pd.DataFrame, model_path : str):
    # model
    model = MLP_FT(base_model, model_params)
    class_weights = compute_class_weight('balanced', classes=np.unique(train['labels'].to_numpy()), y=train['labels'].to_numpy())
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float()) 
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'], weight_decay=training_params['weight_decay'])
    metric = F1Score(task='binary', num_classes=2).to(device)

    # data loaders
    train_loader, valid_loader = data_loader(training_params['batch_size'], train, valid, tokenizer, max_seq_len)

    # train model
    trained_weights, _ = train_model(model, criterion, optimizer, metric, training_params, train_loader, valid_loader)
    
    # reload weights
    model.load_state_dict(trained_weights)
    torch.save(model, model_path)

def find_best_parameters(n_trials : int, train : pd.DataFrame, valid : pd.DataFrame):
    # !mkdir -p tuner/parameters # save model and training parameters
    # !mkdir -p tuner/results # save results

    hpt = HyperTuner(base_model, train, valid, tokenizer, max_seq_len)
    hpt.optimize(n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test_labels.json for submission")
    
    parser.add_argument("--hyperparameter_search", type=int, default="0", help="Number of trials in the hyperparameter search")
    parser.add_argument("--training_folder", type=str, default="training", help="Path to the training folder")
    parser.add_argument("--training_labels", type=str, default="training_labels.json", help="Path to training labelss files")
    parser.add_argument("--model", type=str, default="trained_model.pt", help="Path to save the model")
    parser.add_argument("--training_parameters", type=str, default="default_training_parameters.json", help="Path to the training parameters")
    parser.add_argument("--model_parameters", type=str, default="default_model_parameters.json", help="Path to the model parameters")

    args = parser.parse_args()

    train, valid = read_dataset(args.training_folder, args.training_labels)
    if args.hyperparameter_search > 0:
        find_best_parameters(args.hyperparameter_search, train, valid)
    else:
        training_params = json.load(open(args.training_parameters, "r"))
        model_params = json.load(open(args.model_parameters, "r"))
        train_from_params(model_params, training_params, train, valid, args.model)