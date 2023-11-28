import json
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

def gather_dataset(folder_path : str, combine : bool = True) -> tuple[dict, dict, dict]:
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

def tt_split(dialogs, labels, test_size=0.2, random_state=42):
    train_sentences = []
    val_sentences = []
    train_labels = [] 
    val_labels = []
    for dialog_id in dialogs:
        # train test split inside the dialog
        d_sentences = dialogs[dialog_id]
        d_labels = labels[dialog_id]
        d_train_sentences, d_val_sentences, d_train_labels, d_val_labels = train_test_split(d_sentences, d_labels, test_size=test_size, random_state=random_state)
        
        # aggregate split
        train_sentences += d_train_sentences
        val_sentences += d_val_sentences
        train_labels += d_train_labels
        val_labels += d_val_labels
    
    return train_sentences, val_sentences, train_labels, val_labels

def hotencode(X : list) -> list:
    switcher = {
        "PM" : [1,0,0,0],
        "ME" : [0,1,0,0],
        "UI" : [0,0,1,0],
        "ID" : [0,0,0,1]
    }
    return [switcher[el] for el in X]

# dataset = gather_dataset("training")
# with open("test.json", "w") as json_file:
    # json.dump(dataset, json_file, indent=2)

