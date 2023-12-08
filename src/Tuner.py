import optuna
import torch
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import F1Score

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

from src.Model import MLP_FT
from src.Dataset import data_loader
from src.Trainer import train_model

class HyperTuner:
    def __init__(self, base_model, train, valid, tokenizer, max_seq_len):
        self.base_model = base_model
        self.train = train
        self.valid = valid
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def objective(self, trial):
        model_params = {
            # "input_size" : self.base_model.config.hidden_size + 4 + 1 + 2,
            "input_size" : self.base_model.config.hidden_size + 1 + 2,
            "output_size" : 2,
            "n_layers" : trial.suggest_int("n_layers", 2, 3), 
            "n_p" : trial.suggest_float("n_p", 0.2, 0.7),
        }
        for i in range(model_params["n_layers"]):
            model_params[f"n_{i}_size"] = trial.suggest_int(f"n_{i}_size", 200, 800)

        training_params = {
            "batch_size" : 100,
            "lr" : trial.suggest_float("lr", 1e-5, 1e-4),
            "weight_decay" : trial.suggest_float("weight_decay", 1e-5, 1e-4),
            "n_epochs" : 5,
            "eval_at" : 50,
            "max_patience" : 10,
        }

        # model
        model = MLP_FT(self.base_model, model_params)
        class_weights = compute_class_weight('balanced', classes=np.unique(self.train['labels'].to_numpy()), y=self.train['labels'].to_numpy())
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float()) 
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'], weight_decay=training_params['weight_decay'])
        metric = F1Score(task='binary', num_classes=2).to(device)

        # data loaders
        train_loader, valid_loader = data_loader(training_params['batch_size'], self.train, self.valid, self.tokenizer, self.max_seq_len)
        
        _, results = train_model(model, criterion, optimizer, metric, training_params, train_loader, valid_loader)

        # save results
        json.dump(training_params, open(f"tuner/parameters/training_{trial.number}.json", "w"))
        json.dump(model_params, open(f"tuner/parameters/model_{trial.number}.json", "w"))
        json.dump(results, open(f"tuner/results/results_{trial.number}.json", "w"))

        return results["score"]
    
    def optimize(self, n_trials):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)