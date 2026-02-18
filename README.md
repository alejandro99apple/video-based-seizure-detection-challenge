# XGBoost Landmark Classification

A machine learning project that uses XGBoost for landmark classification from segment data. The project processes data in `.npy` format and trains a classification model.

## Description

This project includes:
- **Model Training**: Script to train an XGBoost classifier with landmark data
- **Prediction**: Main script to make predictions on new data
- **Preprocessing**: Utility functions to clean and augment data
- **Dockerization**: Full support to run the project in Docker containers

## Requirements

- Python 3.x
- Dependencies are listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <REPOSITORY_URL>
cd challenge_alejandro_diaz_montes_de_oca
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python train_model.py
```

This script will process the training data and generate model artifacts in the `model_artifacts/` folder.

### Run predictions

```bash
python main.py <input_dir> <output_dir>
```

Or using environment variables (for Docker):
```bash
export INPUT=<input_path>
export OUTPUT=<output_path>
python main.py
```

## Project Structure

```
├── main.py                 # Main prediction script
├── train_model.py          # Model training script
├── utils.py                # Utility functions
├── requirements.txt        # Project dependencies
├── Dockerfile              # Docker configuration
├── README.md               # This file
├── .gitignore              # Git ignore file
├── data/                   # Training and testing data (.npy files)
├── model_artifacts/        # Trained model artifacts
└── output/                 # Model results and outputs
```

## Docker

To run the project in Docker:

```bash
docker build -t xgboost-classifier .
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output xgboost-classifier
```
