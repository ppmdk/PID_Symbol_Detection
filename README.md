# PID_Symbol_Detection

## Installation and Setup

### 1. Create and Activate Virtual Environment

#### Windows
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate
```

#### Linux/MacOS
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/Scripts/activate
```

## Usage Guide

### Stage 1: Generic Symbol Detection (Class-Agnostic)

#### Data Preparation
```bash
python src/run_pipeline.py stage1 --prepare_data
```

#### Model Training
```bash
python src/run_pipeline.py stage1 --train_model
```

#### Complete Stage 1 Pipeline (Data Prep + Training)
```bash
python src/run_pipeline.py stage1
```

### Stage 2: Symbol Classification (Few-Shot)

#### Data Preparation
```bash
python src/run_pipeline.py stage2 --prepare_data
```

#### Model Training
```bash
python src/run_pipeline.py stage2 --train_model
```

#### Complete Stage 2 Pipeline (Data Prep + Training)
```bash
python src/run_pipeline.py stage2
```

### Inference

#### Stage 1 Inference
```bash
python src/run_pipeline.py stage1_inference
```

#### Stage 2 Inference (Label Transfer)
```bash
python src/run_pipeline.py stage2_inference
```

### Evaluation

#### Stage 1 Evaluation
```bash
python src/run_inference.py evaluation --evaluate_stage1
```

#### Stage 2 Evaluation
```bash
python src/run_inference.py evaluation --evaluate_stage2
```

#### Complete Evaluation (Both Stages)
```bash
python src/run_inference.py evaluation
```

### Data Organization

#### Directory Structure
```
data/
├── raw/
│   ├── images/          # Place your P&ID images here
│   └── labels_class_aware/  # Place corresponding ground truth labels here
```

#### Sample Dataset
The repository includes a sample dataset from the [Dataset-P&ID](https://drive.google.com/drive/u/1/folders/1gMm_YKBZtXB3qUKUpI-LF1HE_MgzwfeR) collection, containing 5 P&ID images and their corresponding labels. This dataset is used for demonstration purposes and originates from the research paper available [here](https://arxiv.org/pdf/2109.03794).

#### Adding Custom Data
To use your own P&ID images:
1. Place your images in `data/raw/images/`
2. Place corresponding ground truth labels in `data/raw/labels_class_aware/`
3. Ensure labels follow the same format as the sample dataset


### Proposed Framework vs Conventional Framework

<img src="./media/workflow.svg" >

### Benefits of Proposed Framework
Class-Agnostic Object Detection & One-shot Label Transfer is found to be more:
1. Generalizable to different underlying P&ID drawing styles
2. Robust to class-imbalance
compared to equivalent class-aware counterparts.

### Simplified Visual Walkthrough of Proposed Framework 

#### 1. Data preprocessing

This step breaks down large P&ID sheets into overlapping patches. 

<img src="./media/overlapping_patches.png" width="800">

Plus, class-aware labels are transformed into class-agnostic to prepare for training a Yolo object detection model.

#### 2. Train Yolo (Stage-1)

Trains a 'Generic' symbol detector

<img src="./media/train_yolo.svg" width="400">

#### 3. Inferencing with SAHI (Stage-1)

For large P&IDs infer on smaller patches and combine the results (implemented via <a href="https://github.com/obss/sahi"> SAHI </a>).

<img src="./media/sahi_sample.gif" width="250">

#### 4. Label Transfer (Stage-2)
Train a model using one labeled image per symbol class (e.g. P&ID legend). The model can be a Siamese Network/ Prototypical (Zero-shot) Network or a Traditional classifier trained on augmented images.

<img src="./media/label_transfer.png" width="400">


If you use this package in your work, please cite it as:
```
@article{GUPTA2024105260,
title = {Semi-supervised symbol detection for piping and instrumentation drawings},
journal = {Automation in Construction},
volume = {159},
pages = {105260},
year = {2024},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2023.105260},
url = {https://www.sciencedirect.com/science/article/pii/S0926580523005204},
author = {Mohit Gupta and Chialing Wei and Thomas Czerniawski},
}
```
