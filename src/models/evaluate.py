# load train dataset and val_idx
# filter by val_idx
# after look at https://github.com/iterative/example-get-started/blob/main/src/evaluate.py


import logging
import os
import json

import src.config as cfg

import click
import pandas as pd
from src.utils import save_as_pickle

from catboost import CatBoostClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('input_model_filepath', type=click.Path(exists=True))
@click.argument('input_validx_filepath', type=click.Path(exists=True))

def main(input_data_filepath, input_target_filepath, input_model_filepath, input_validx_filepath):
  
    logger = logging.getLogger(__name__)
    logger.info('making validation metrics')

    train_data = pd.read_pickle(input_data_filepath)
    train_target = pd.read_pickle(input_target_filepath)

    val_indxes = pd.read_pickle(input_validx_filepath)['indexes'].values
    
    val_data = train_data.loc[val_indxes]
    val_target = train_target.loc[val_indxes]



    trained_model = CatBoostClassifier().load_model(input_model_filepath)

    y_pred = trained_model.predict(val_data)

    precision_per_class = precision_score(val_target, y_pred, average=None).tolist()
    precision_weighted = precision_score(val_target,y_pred, average='weighted')

    recall_per_class = recall_score(val_target, y_pred, average=None).tolist()
    recall_weighted = recall_score(val_target, y_pred, average='weighted')


    metrics = {

        'f1': f1_score(val_target, y_pred, average = 'weighted'),
        'acc': accuracy_score(val_target, y_pred),
        'precision_per_class':  precision_per_class,
        'precision': precision_score(val_target, y_pred, average='weighted'),
        'recall_per_class': recall_per_class,
        'recall': recall_weighted
    }

    with open("reports/figures/metrics.json", "w") as outfile:
        json.dump(metrics, outfile)

main()
