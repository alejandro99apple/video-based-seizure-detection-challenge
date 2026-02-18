import sys
import os
import glob
import json
import numpy as np
import pandas as pd
import utils
import re

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from xgboost import XGBClassifier


def explore_train_data(train_data_dir='../data'):
    if train_data_dir is None:
        return
    
    _npy_paths = glob.glob(train_data_dir + '/*.npy')
    _train_data_df = pd.read_csv(os.path.join(train_data_dir, 'train_data.csv'))
    print('npy paths', len(_npy_paths), 'labels', _train_data_df.shape, _train_data_df['label'].sum())


def process_train_data(train_data_dir='../train_data', max_elem=None):
    _train_data_df = pd.read_csv(os.path.join(train_data_dir, 'train_data.csv'))
    _means = []

    UTILITY_THRESHOLD = 0.80     # use up to this value
    instances = []

    for i in range(len(_train_data_df)):
        if max_elem is not None and i >= max_elem:
            break

        _name = _train_data_df.iloc[i]['segment_name']
        _label = _train_data_df.iloc[i]['label']
        _lmk_arr = np.load(os.path.join(train_data_dir, _name))
        _ID = int(re.search(r'child_(\d+)_', _name).group(1))

        print(f"{i} - Name:{_name}  Label:{_label}   Shape:{_lmk_arr.shape}  NaN Values:{np.sum(np.isnan(_lmk_arr))}")

        # Replace np.nan values
        _lmk_arr = utils.preprocess_and_augment(_lmk_arr, UTILITY_THRESHOLD)

        instance_vector = utils.process_data(_lmk_arr, _ID, _label)

        instances.append(instance_vector)

    trainable_dataset = pd.DataFrame(instances)

    return trainable_dataset




def train_model(train_data:pd.DataFrame):
    
    X = train_data.drop(columns=["class", "ID"]).values.astype('float32')
    y = (train_data["class"] == 1).astype('float32').values
    groups = train_data["ID"].values  # Group by subject ID

    print("\nStarting model training...")
    print(f"Total samples: {len(X)}")
    print(f"Positive samples: {y.sum()}, Negative samples: {(1-y).sum()}")
    print(f"Number of unique subjects: {len(np.unique(groups))}\n")

    model = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        max_depth=7,
        min_child_weight=5,
        reg_lambda=10.0,
        scale_pos_weight=2.4
    )

    model.fit(X, y)

    model_dir = '/model_artifacts'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgb_model.json')
    model.get_booster().save_model(model_path)

    feature_columns = [c for c in train_data.columns if c not in ['class', 'ID']]
    metadata = {
        'feature_columns': feature_columns,
        'utility_threshold': 0.80,
        'decision_threshold': 0.5,
        'positive_label': 1,
        'negative_label': 0,
        'n_samples': int(len(X)),
        'n_positive': int(y.sum()),
        'n_negative': int((1 - y).sum()),
        'n_subjects': int(len(np.unique(groups))),
        'model_format': 'xgboost-json'
    }
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Model saved at: {model_path}")
    print(f"Metadata saved at: {metadata_path}")

    return model


if __name__ == '__main__':
    
    explore_train_data("data")
    trainable_dataset = process_train_data("data")
    train_model(trainable_dataset)
