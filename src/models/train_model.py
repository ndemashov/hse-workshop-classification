# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.utils import save_as_pickle
import pandas as pd
import catboost as cb
import src.config as cfg
import os

@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path())
@click.argument('output_validx_filepath', type=click.Path())
def main(input_data_filepath, input_target_filepath, output_model_filepath, output_validx_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_data = pd.read_pickle(input_data_filepath)
    train_target = pd.read_pickle(input_target_filepath)

    train_idx, val_idx = train_test_split(
        train_data.index, test_size=0.2, random_state=42)


    clf = cb.CatBoostClassifier(
        loss_function='MultiLogloss',
        iterations=1000,
        silent=True,
        depth=6,
        l2_leaf_reg=2.0,
        learning_rate=0.01,
        early_stopping_rounds=100,
        random_seed=77,
        cat_features= cfg.CAT_COLS
    )
    clf.fit(train_data.loc[train_idx], train_target.loc[train_idx])
    clf.save_model(os.path.join(output_model_filepath, "catboost.cbm"))
    pd.DataFrame({'indexes':val_idx.values}).to_csv(output_validx_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
