import numpy as np
import pandas as pd

from sklearn import model_selection, metrics
import lightgbm as lgb
import matplotlib.pyplot as plt

from pathlib import Path

import os
import pandas as pd

SEED = 42





def oof_adversarial_analysis(X_train_total,y_train_total,panel_original):

    params_2 = {'objective': 'binary',
              'max_depth': 2,
              'boosting': 'gbdt',
                'verbose':-1,
              'metric': 'auc'}
    params_2 =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
          'max_depth': 2,
        'num_leaves': 4,
        'learning_rate': 0.02,
        'verbose': -1,
        'lambda_l1': 1,
        'scale_pos_weight': 8,  #for unbalanced labels
        "seed": 42

    } 
    cv_scores = []

    
    num_rounds = 1000
    early_stopping_rounds = 100
    verbose_eval = 50

    #features = list(adversarial_result['feature'])

    oof_pred = np.zeros((len(X_train_total), ))
    
    oof_pred_original = np.zeros((len(panel_original), ))

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=SEED)
    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_importance_df = pd.DataFrame()

    #for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_total, y_train_total)):
    for fold, (train_idx, val_idx) in enumerate(kf.split(y_train_total)):

        x_train, x_val = X_train_total.iloc[train_idx],X_train_total.iloc[val_idx]
        y_train, y_val = y_train_total.iloc[train_idx],y_train_total.iloc[val_idx]

        train_set = lgb.Dataset(x_train, label=y_train)
        test_set  = lgb.Dataset(x_val,   label=y_val)

        model = lgb.train(params_2, train_set,
                                          num_rounds,
                    valid_sets=[train_set, test_set],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval
                )
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train_total.columns
        fold_importance_df["importance"] = model.feature_importance(importance_type="gain")
        fold_importance_df["fold"] = fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof_pred[val_idx] = model.predict(x_val).reshape(oof_pred[val_idx].shape)
        
        oof_pred_original += model.predict(panel_original[X_train_total.columns])
        
        cv_scores.append(model.best_score["valid_1"]['auc'])
        print("\n")

    adversarial_validation_auc = np.mean(cv_scores)
    print(f"Mean Adversarial AUC: {adversarial_validation_auc:.4f}")
    
    oof_pred_original = oof_pred_original/5.0
    return oof_pred,feature_importance_df,adversarial_validation_auc,oof_pred_original



def adversarial_training(panel,columns_group,params,num_rounds = 1000,
early_stopping_rounds = 100,
verbose_eval = 50,N_SPLITS=5):

    auc_variables = []

    for column in columns_group:
        #try:
        columns_for_model = [column]

        print(10*'*',column,10*'*')
        feature_importance_df = pd.DataFrame()

        kf = model_selection.KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        cv_scores = []
        models = []

        for fold_idx, (dev_idx, val_idx) in enumerate(kf.split(panel)):
            print(f"Fold: {fold_idx+1}")
            X_dev, y_dev = panel.loc[dev_idx, columns_for_model], panel.loc[dev_idx, "is_test"].values
            X_val, y_val = panel.loc[val_idx, columns_for_model], panel.loc[val_idx, "is_test"].values

            dev_dataset = lgb.Dataset(X_dev, y_dev)
            val_dataset = lgb.Dataset(X_val, y_val)

            clf = lgb.train(
                params,
                dev_dataset,
                num_rounds,
                valid_sets=[dev_dataset, val_dataset],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval
            )

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = X_dev.columns
            fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
            fold_importance_df["fold"] = fold_idx + 1

            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


            cv_scores.append(clf.best_score["valid_1"]['auc'])
            models.append(clf)
            print("\n")

        adversarial_validation_auc = np.mean(cv_scores)
        print(column,f"Mean Adversarial AUC: {adversarial_validation_auc:.4f}")

        auc_variables.append([column,adversarial_validation_auc])

    return auc_variables,feature_importance_df