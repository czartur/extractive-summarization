# École Polytechnique  ~ INF554 Data Challenge 

## Overview
This repository contains the code and resources for a deep learning project focused on text classification using fine-tuned language models. The goal of the project is to classify utterances of role-playing dialogs as relevant or irrelevant.

## Table of Contents 📑
- [Installation](#installation) 📥
- [Usage](#usage)🚦
- [Dataset](#dataset)📂
- [Model Architecture](#model-architecture)🧠
- [Training](#training)🏋️‍♂️
- [Evaluation](#evaluation)📊
- [Results](#results)📈

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
1. The data for the features should be organized into a folder containing multiple JSON files, one for each dialogue:

```json
[
    {
        "speaker": "PM",
        "text": "Okay, well",
    },
    {
        "speaker": "ME",
        "text": "Hello there!",
    },
    ...
]
```

* speaker: The speaker of the dialogu, who should be a project manager (PM), a marketing expert (ME), a user interface designer (UI) or an industrial designer(ID).
* text: The text spoken by the speaker.

2. The data for labels should be a single JSON file containing a list of labels (0 or 1) for each dialogue id:

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

# Training 🏋️‍♂️

# Evaluation 📊

# Results📈
