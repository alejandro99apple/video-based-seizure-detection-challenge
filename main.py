import sys
import os
import glob
import json
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import utils


if __name__ == '__main__':
    base_data_dir = '/data'
    base_output_dir = '/output'
    input_rel = os.environ.get('INPUT')
    output_rel = os.environ.get('OUTPUT')

    # Determine if using ENV vars or command line args
    using_env_vars = (input_rel is not None) or (output_rel is not None)
    
    if using_env_vars:
        # Environment variables mode
        test_data_dir = os.path.join(base_data_dir, input_rel) if input_rel else base_data_dir
        output_path = os.path.join(base_output_dir, output_rel) if output_rel else os.path.join(base_output_dir, 'test_data.csv')
        output_dir = os.path.dirname(output_path) if output_rel else base_output_dir
    else:
        # Command line arguments mode
        if len(sys.argv) < 3:
            raise SystemExit(
                'Usage: python main.py <input_dir> <output_dir>\n'
                'Or set env vars INPUT and OUTPUT for Docker execution.'
            )
        test_data_dir = sys.argv[1]
        output_dir = sys.argv[2]
        output_path = os.path.join(output_dir, 'test_data.csv')

    os.makedirs(output_dir, exist_ok=True)
    print(sys.argv)
    print('Input dir:', test_data_dir)
    print('Output file:', output_path)

    model_dir = os.path.join(os.path.dirname(__file__), 'model_artifacts')
    model_path = os.path.join(model_dir, 'xgb_model.json')
    metadata_path = os.path.join(model_dir, 'metadata.json')

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    booster = xgb.Booster()
    booster.load_model(model_path)

    feature_columns = metadata['feature_columns']
    utility_threshold = float(metadata.get('utility_threshold', 0.8))
    decision_threshold = float(metadata.get('decision_threshold', 0.5))

    file_names = []
    feature_rows = []

    paths = glob.glob(os.path.join(test_data_dir, '*.npy'))
    print('The found .npy landmarks files:', len(paths))
    print("Processing files...")

    for p in paths:
        name = os.path.basename(p)
        lmk_arr = np.load(p)
        match = re.search(r'child_(\d+)_', name)
        subject_id = int(match.group(1)) if match else -1

        lmk_arr = utils.preprocess_and_augment(lmk_arr, utility_threshold)
        instance_vector = utils.process_data(lmk_arr, subject_id, 0)

        file_names.append(name)
        feature_rows.append(instance_vector)

    features_df = pd.DataFrame(feature_rows)
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feature_columns].values.astype('float32')
    dmatrix = xgb.DMatrix(X, feature_names=feature_columns)
    probs = booster.predict(dmatrix)
    labels = (probs >= decision_threshold).astype(int)

    output_df = pd.DataFrame({'file_name': file_names, 'label': labels})
    output_df.to_csv(output_path, sep=',', index=False)
    print('Inference done, saving csv file:', output_path, output_df.shape)
