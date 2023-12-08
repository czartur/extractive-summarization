import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, filename='logs/training.log', encoding='utf-8', filemode='a')

def train_model(model, criterion, optimizer, metric, params, train_loader, valid_loader, early_stopping = True):
    n_epochs = params['n_epochs']
    eval_at = params['eval_at']
    max_patience = params['max_patience']

    hst_train_loss = [] 
    hst_valid_loss = []
    hst_f1_score = []

    best_valid_loss = float("inf")
    best_f1_score = 0
    patience = max_patience
    best_weights = None
    
    it = 0
    # itera nas epochs
    for epoch in range(n_epochs):
        if patience == 0: break
        
        # itera nos train batches
        for samples in tqdm(train_loader):
            if patience == 0: break
            it += 1

            # train step
            # model.train()
            out = model(samples)
            optimizer.zero_grad()
            loss = criterion(out, samples['label'])
            loss.backward()
            optimizer.step()
            
            train_loss = loss.cpu().detach().numpy() / 1

            if it % eval_at != 0: continue

            # model.eval()

            valid_loss = 0
            f1_score = 0
            
            # itera nos valid batches
            for samples in valid_loader:
                with torch.no_grad():
                    out = model(samples)
                    loss = criterion(out, samples['label'])
                    valid_loss += loss.cpu().detach().numpy() / len(valid_loader)
                    f1_score += metric(samples['label'], out.argmax(dim=1)).cpu().detach().numpy() / len(valid_loader)
            
            # early stopping
            if early_stopping:
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_weights = model.state_dict()
                    patience = max_patience
                else:
                    patience -= 1 
                # if valid_loss < best_valid_loss:
                #     best_valid_loss = valid_loss
                #     best_weights = model.state_dict()
                #     patience = max_patience
                # else:
                #     patience -= 1 
            
            hst_train_loss.append(train_loss)
            hst_valid_loss.append(valid_loss)
            hst_f1_score.append(f1_score)

            logging.info('Iter: {} | Train Loss: {} | Val Loss: {} | F1-score: {}'.format(it, train_loss, valid_loss, f1_score))

    # objective function criterion
    combined = sorted(zip(hst_valid_loss, hst_f1_score), key=lambda x : x[0])
    _, scores = zip(*combined)
    qtd = 3
    final_score = sum(scores[:qtd]) / qtd

    results = {
        "score" : final_score,
        "params" : params,
        "valid_loss" : hst_valid_loss,
        "train_loss" : hst_train_loss,
        "f1_score" : hst_f1_score, 
    }
    
    return best_weights, results