import json
import torch
from pathlib import Path
from typing import Union, Optional
"""
Concatenates all sentences and *labels in $folder_path to lists

"""
def read_data(dialogs_folder : str, labels_file : Optional[str]) -> tuple[list, list, Optional[list]]:

    dialogs_path = Path(dialogs_folder)
    labels_path = Path(labels_file)
    
    if not dialogs_path.is_dir():
        raise Exception(f"Error: {dialogs_folder} is not a folder")
    if not labels_path.is_file():
        raise Exception(f"Error: {labels_file} is not a file")
    
    sentences = []
    speakers = []
    labels = []
    
    labels_data = json.load(open(labels_path, "r"))
    for item in Path(dialogs_path).iterdir():
        if not item.is_file(): continue 
        if not item.suffix == ".json": continue
        
        # load data
        dialog = json.load(open(item, "r"))
        sentences += [exchange["text"] for exchange in dialog]
        speakers += [exchange["speaker"] for exchange in dialog]
        
        if labels_path:
            labels += labels_data[item.stem]
    
    if labels_path:
        return sentences, speakers, labels

    return sentences, speakers

"""
Concatenates sentences of each dialog in $folder_path 

"""
def read_data_by_ID(folder_path : str, combine : bool = True) -> tuple[dict, dict, dict]:
    speakers = dict()
    dialogs = dict()
    edges = dict()
    for item in Path(folder_path).iterdir():
        if not item.is_file(): continue 
        if not item.suffix == ".json": continue

        dialog_id = item.stem

        with open(item.with_suffix(".json"), "r") as json_file:
            nodes_data = json.load(json_file)
        with open(item.with_suffix(".txt"), "r") as txt_file:
            edges_data = txt_file.readlines() 
        
        # nodes
        dialog = []
        speaker = [] 
        for sentence in nodes_data:
            speaker.append(sentence["speaker"])
            if combine:
                dialog.append(sentence["speaker"] + ": " + sentence["text"])
            else:
                dialog.append(sentence["text"]) 
        dialogs[dialog_id] = dialog
        speakers[dialog_id] = speaker

        
        # edges
        connections = [] 
        for connection in edges_data:
            id_from, attribute, id_to = connection.split()
            connections.append([int(id_from), attribute, int(id_to)])
        edges[dialog_id] = connections
        
    return dialogs, speakers, edges

def format_input(sentences, speakers, tokenizer, max_seq_len, device):
    # hot encoder
    switcher = {
        "PM" : [1,0,0,0],
        "ME" : [0,1,0,0],
        "UI" : [0,0,1,0],
        "ID" : [0,0,0,1]
    }
    # params
    params = {
        'max_length' : max_seq_len,
        'padding' : True,
        'truncation' : True,
        'return_token_type_ids' : False
    }
    # tokenization
    tokens = tokenizer.batch_encode_plus(sentences, **params)

    res = {
        "seq" : torch.tensor(tokens['input_ids']).to(device),
        "mask" : torch.tensor(tokens['attention_mask']).to(device),
        "speakers" : torch.Tensor([switcher[el] for el in speakers]).to(device),
        "lengths" : torch.Tensor([[len(sentence.split())] for sentence in sentences]).to(device),
    }

    return res