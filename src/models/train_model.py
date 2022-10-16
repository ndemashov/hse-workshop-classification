# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
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


    RANDOM_STATE = 77
    N_SPLITS = 5
    N_RANDOM_SEEDS = 7

    train_idx, val_idx = train_test_split(
        train_data.index, 
        test_size=0.2, 
        random_state=RANDOM_STATE
    )

    

    skf = StratifiedKFold(
        n_splits= N_SPLITS, 
        shuffle=True, 
        random_state=RANDOM_STATE
    )

    lgr = LogisticRegressionCV(
        Cs=50,
        solver='lbfgs',
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    real_pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    real_cols_pipe = make_pipeline(SimpleImputer(), StandardScaler(), PowerTransformer())
    preprocess_pipe = ColumnTransformer(n_jobs=-1, transformers=[
        ('real_cols', real_pipe, cfg.REAL_COLS),
        ('cat_cols', cat_pipe, cfg.CAT_COLS),
        ('ohe_cols', 'passthrough', cfg.OHE_COLS)
    ])

    model_one_target = Pipeline([
        ('preprocess_original_features', preprocess_pipe),
        ('model', lgr)
    ])

    logit = MultiOutputClassifier(model_one_target, n_jobs=-1)

    for train_idx, test_idx in skf.split(train_data, train_target.sum(axis=1)):
        X_train, X_test = train_data.iloc[train_idx], train_data.iloc[test_idx]
        y_train, y_test = train_target.iloc[train_idx], train_target.iloc[test_idx]   
        logit.fit(X_train, y_train)
          
    logit.save_model(os.path.join(output_model_filepath, "logreg.cbm"))
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
