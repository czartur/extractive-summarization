# École Polytechnique  ~ INF554 Data Challenge 

## Overview
This repository contains the code and resources for a deep learning project focused on text classification using fine-tuned language models. The goal of the project is to classify utterances of role-playing dialogs as relevant or irrelevant.

## Table of Contents 📑
- [Installation](#installation) 📥
- [Usage](#usage)🚦
- [Dataset](#dataset)📂
- [Model Architecture](#model-architecture)🧠
- [Training](#training)🏋️‍♂️
- [Prediction](#prediction)📊

## Installation 📥
1. Ensure you have Python 3.x (with pip) before running the project

2. Clone the repository:
   ```bash
   git clone https://github.com/czartur/extractive-summarization.git
   cd extractive-summarization

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

## Usage🚦
To use the project, follow these steps:

1. Prepare your dataset or use the provided example dataset.
2. Configure the model architecture and hyperparameters.
3. Train the model using the provided scripts.
4. Evaluate the model on test data. 

## Dataset 📂
1. The data for the features should be organized into a folder containing multiple JSON and TXT files encoding information of each dialogue in a graph like dataset:

- The JSON files (nodes)
```json
[
    {
        "speaker": "PM",
        "text": "Okay, well",
        "index": 0,
    },
    {
        "speaker": "ME",
        "text": "Hello there!",
        "index": 1,
    },
    ...
]
```

* speaker: The speaker of the dialogu, who should be a project manager (PM), a marketing expert (ME), a user interface designer (UI) or an industrial designer(ID).
* text: The text spoken by the speaker.
* index: The index of the sentence in the dialog

- The TXT files (edges)
```txt
0 Continuation 1
0 Continuation 2
...
```
* first column: "from" index
* second column: attribute
* third columns: "to" index

2. The data for the labels should be a single JSON file containing a list of classes (0 or 1) for each dialogue id:

```json
"IS1003d": [
        0,
        1,
        0,
        0,
        1,
        ...
]
```

# Model Architecture 🧠

The final architecture of our model consists of two primary components: a BERT layer and a MLP layer. The data flow involves three key steps:

* The initial step utilizes BERT as the foundational element of our model. This layer expects sequences of tokens and yields an embedding vector for each sequence. 
% Sequences are generated from the utterances, with each word mapped to a unique identifier using a dedicated tokenizer for the model.
* Crafted features are combined with the embedding vector of a classification token, obtained from the output of the last BERT layer.
* Subsequently, this concatenation serves as input to the MLP layer. Multiple configurations are explored for the MLP, including variations in the number of layers, layer size, and dropout rates.

Our chosen optimizer is Adam, and we employ binary cross-entropy as the loss function. Notably, the loss function is supplied with weights derived from the frequency of each class in the training dataset, considering a significant majority of non-important sentences. It is worth emphasizing that both the BERT layer and the MLP layer contribute to the backward step, as we opted not to freeze BERT's parameters.

# Training 🏋️‍♂️

To train our model we provide a script that can be run by specifying a ```training folder``` containing training features as explained in the dataset section and a ```training labels``` file. The user can also provide custom files for training and model parameters following the example already present in the repo:

```bash
    python3 training.py --training_folder <folder_path> --training_labels <labels_path>
```

By default, this script will generate a ```trained_model.pt``` file that can be used for prediction later. If prefered, the output model path can also be specified in the script under the tag ```--model```.

Additionally, we also provide a way to find the best set of hyperparameters by using optuna library and specifying the number of trials:

```bash
    python3 training.py --hyperparameter_search=<n_trials>
```

The results and parameters found are saved inside the folder tuner.

# Prediction 📊

From a trained model we can predict labels for a test dataset. The test folder and the model file to be used has to be specified:

```bash
    python3 prediction.py --test_folder <test_folder_path> --model <model_path>
```

Alternatively, we can use a set of predicted labels to generate, using a majority vote strategy, a final prediction. The set of predicted test labels must be inside the same folder which is passed as an argument to the script:

```bash
    python3 ensembling.py --prediction_folder <prediction_folder_path>
```

By default, these two script will generate a ```test_labels.json``` but this path can also be specified in the scripts under the tag --test_labels.
