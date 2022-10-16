import sys
import src.config as cfg
import gc

import pandas as pd
import numpy as np
import re
from tqdm.notebook import tqdm
from metrics import *
from helper import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.impute import *
from sklearn.multioutput import *
from sklearn.base import clone
from sklearn.svm import *
from sklearn.model_selection import *

def logreg():

    # загрузка данных
    train = pd.read_pickle(cfg.PREPARED_TRAIN_DATA_PATH)
    test = pd.read_pickle(cfg.PREPARED_TEST_DATA_PATH)
    val = pd.read_pickle(cfg.PREPARED_VAL_DATA_PATH)

    # отделение целевых данных 
    X_train, Y_train = train.drop(cfg.TARGETS, axis=1), train[cfg.TARGETS]

    pred_proba_oof = pd.DataFrame(data=np.zeros(shape=(len(train), len(cfg.TARGETS))), index=train.index, columns=cfg.TARGETS)
    pred_proba_test = pd.DataFrame(data=np.zeros(shape=(len(test), len(cfg.TARGETS))), index=test.index, columns=cfg.TARGETS)
    metrics = {}

    EXPERIMENT_FAMILY_NAME = 'logreg'
    EXPERIMENT_NAME = 'baseline'
    RANDOM_STATE = 77
    N_SPLITS = 5
    N_RANDOM_SEEDS = 7


    scoring = get_weird_single_col_pred_proba_score()

    base_model = LogisticRegressionCV(
        Cs=50,
        solver='lbfgs',
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    real_cols_pipe = make_pipeline(SimpleImputer(), StandardScaler(), PowerTransformer())
    preprocess_pipe = ColumnTransformer(n_jobs=-1, transformers=[
        ('real_cols', real_cols_pipe, cfg.REAL_COLS),
        ('cat_cols', OneHotEncoder(handle_unknown='ignore', dtype=np.int8), cfg.CAT_UNORDERED_COLS),
        ('ordinal_cols', clone(real_cols_pipe), cfg.CAT_ORDERED_COLS),
        ('binary_cols', SimpleImputer(strategy='constant', fill_value=0), cfg.BINARY_COLS),
        ('real_poly', make_pipeline(clone(real_cols_pipe), PolynomialFeatures(degree=3, interaction_only=False)), cfg.REAL_COLS),
        ('binary_poly', make_pipeline(clone(real_cols_pipe), PolynomialFeatures(degree=3, interaction_only=True)), cfg.BINARY_COLS)
    ])

    model_one_target = Pipeline([
        ('preprocess_original_features', preprocess_pipe),
        ('model', base_model)
    ])
    model = MultiOutputClassifier(model_one_target, n_jobs=-1)


    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    fold = 0
    for train_idx, val_idx in tqdm(cv.split(X_train, Y_train), total=N_SPLITS):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_idx], Y_train.iloc[train_idx])
            
        model_name = f'{EXPERIMENT_NAME}_fold_{fold}.pkl'
        model_path = os.path.join(cfg.MODELS_PATH, EXPERIMENT_FAMILY_NAME, EXPERIMENT_NAME)
        check_path(model_path)
        save_model(fold_model, os.path.join(model_path, model_name))
        
        pred_proba_oof.iloc[val_idx, :] += squeeze_pred_proba(fold_model.predict_proba(X_train.iloc[val_idx]))
        pred_proba_test.iloc[:, :] += squeeze_pred_proba(fold_model.predict_proba(test))
        gc.collect()

        fold += 1

    pred_proba_test /= N_SPLITS

    tresholds = get_tresholds(train[cfg.TARGETS], pred_proba_oof)
    sample_submission = pd.read_csv(cfg.SAMPLE_SUBMISSION_PATH).set_index('ID')
    submission = make_prediction(pred_proba_test, tresholds, sample_submission)

    submission_path = os.path.join(cfg.SUBMISSION_PATH, EXPERIMENT_FAMILY_NAME)
    check_path(submission_path)
    submission.to_csv(os.path.join(submission_path, f'{EXPERIMENT_NAME}.csv'))

    pred_proba_oof_path = os.path.join(cfg.OOF_PRED_PATH, EXPERIMENT_FAMILY_NAME)
    check_path(pred_proba_oof_path)
    pred_proba_oof.to_pickle(os.path.join(pred_proba_oof_path, f'{EXPERIMENT_NAME}.pkl'))

    pred_proba_test_path = os.path.join(cfg.TEST_PRED_PATH, EXPERIMENT_FAMILY_NAME)
    check_path(pred_proba_test_path)
    pred_proba_test.to_pickle(os.path.join(pred_proba_test_path, f'{EXPERIMENT_NAME}.pkl'))

    