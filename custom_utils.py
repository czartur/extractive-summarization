import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def gather_dataset(folder_path : str) -> dict:
    nodes = dict()
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
        for sentence in nodes_data:
            dialog.append(sentence["speaker"] + ": " + sentence["text"])
        nodes[dialog_id] = dialog 
        
        # edges
        connections = [] 
        for connection in edges_data:
            id_from, id_to = connection.split()[0], connection.split()[2]
            connections.append([int(id_from), int(id_to)])
        edges[dialog_id] = connections
        
    return nodes, edges

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


# dataset = gather_dataset("training")
# with open("test.json", "w") as json_file:
    # json.dump(dataset, json_file, indent=2)

