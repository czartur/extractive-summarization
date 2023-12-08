
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

class WordFeatureDataset(Dataset):
    def __init__(self, data : pd.DataFrame, tokenizer : AutoTokenizer, max_seq_len : int):
        # gather data
        sentences = data['sentences'].to_list()
        speakers = data['speakers'].to_list()
        labels = data['labels'].to_list()
        in_degrees = data['in_degrees'].to_list()
        out_degrees = data['out_degrees'].to_list()

        # token parameters
        params = {
            'max_length' : max_seq_len,
            'padding' : True,
            'truncation' : True,
            'add_special_tokens' : True,
            'return_token_type_ids' : False
        }
        tokens = tokenizer.batch_encode_plus(sentences, **params)
        
        # hot encoder for speakers
        switcher = {
            "PM" : [1,0,0,0],
            "ME" : [0,1,0,0],
            "UI" : [0,0,1,0],
            "ID" : [0,0,0,1]
        }

        self.sequences = torch.tensor(tokens['input_ids']).to(device)
        self.attention_masks = torch.tensor(tokens['attention_mask']).to(device)
        self.speakers = torch.Tensor([switcher[el] for el in speakers]).to(device)
        self.lengths = torch.Tensor([[len(sentence.split())] for sentence in sentences]).to(device)
        self.in_degrees = torch.Tensor([[deg] for deg in in_degrees]).to(device)
        self.out_degrees = torch.Tensor([[deg/6] for deg in out_degrees]).to(device)
        self.labels = torch.tensor(labels).to(device)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        attention_mask = self.attention_masks[idx]
        speaker = self.speakers[idx]
        in_degree = self.in_degrees[idx]
        out_degree = self.out_degrees[idx]
        length = self.lengths[idx]
        label = self.labels[idx]

        sample = {
            'sequence': sequence,
            'attention_mask': attention_mask,
            'speaker': speaker,
            'in_degree' : in_degree,
            'out_degree' : out_degree,
            'length': length,
            'label': label
        }
        return sample
    

def data_loader(batch_size : int, train : pd.DataFrame, valid : pd.DataFrame, tokenizer : AutoTokenizer, max_seq_len : int = 80) -> tuple[DataLoader, DataLoader]:
    # create custom datasets
    train_dataset = WordFeatureDataset(train, tokenizer, max_seq_len)
    valid_dataset = WordFeatureDataset(valid, tokenizer, max_seq_len)

    # create dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, generator=torch.Generator(device=device))
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, generator=torch.Generator(device=device))

    return train_loader, valid_loader