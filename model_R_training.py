"""
Model Training and Evaluation for R Target

This module implements the training, evaluation, and analysis of a LightGBM regression model
for predicting R values. It includes cross-validation, performance metrics calculation,
visualization of actual vs predicted values, and SHAP value analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm
import shap
from joblib import dump, load
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm.callback import early_stopping

def train_and_evaluate_model_R(X_analize, target_R, n_splits=5, n_repeats=20, random_state=123):
    """
    Trains and evaluates a LightGBM model for R target using repeated k-fold cross-validation.
    
    Args:
        X_analize: DataFrame with features
        target_R: Series with target values
        n_splits: Number of folds for cross-validation
        n_repeats: Number of repetitions for cross-validation
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with performance metrics
    """
    X = X_analize
    Y = target_R
    
    train_R2_metric_results = []
    train_mse_metric_results = []
    train_mae_metric_results = []
    test_R2_metric_results = []
    test_mse_metric_results = []
    test_mae_metric_results = []
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    for train_indices, test_indices in rkf.split(X):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
        
        model = lightgbm.LGBMRegressor(
            n_jobs=-1,
            objective='regression',
            num_leaves=112,
            learning_rate=0.1,
            feature_fraction=0.9855475553463364,
            bagging_fraction=0.8069570540135589,
            min_data_in_leaf=5,
            metric='rmse',
            custom_eval_metric_name=None,
            lambda_l1=0.0026686145190637704,
            lambda_l2=8.730406606367973e-06,
            bagging_freq=7,
            extra_trees=False,
            num_boost_round=1000,
            feature_pre_filter=False,
            explain_level=0,
            verbose=-1,
        )
        
        stop_callback = early_stopping(stopping_rounds=1000)
        lgb_model = model.fit(
            X_train, Y_train,
            eval_set=[(X_test, Y_test)],
            callbacks=[stop_callback],
        )
        
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_R2_metric_results.append(r2_score(Y_train, train_preds))
        train_mse_metric_results.append(mean_squared_error(Y_train, train_preds))
        train_mae_metric_results.append(mean_absolute_error(Y_train, train_preds))
        test_R2_metric_results.append(r2_score(Y_test, test_preds))
        test_mse_metric_results.append(mean_squared_error(Y_test, test_preds))
        test_mae_metric_results.append(mean_absolute_error(Y_test, test_preds))
    
    print('Train')
    print('Train R-square:', np.mean(train_R2_metric_results))
    print('Mean Absolute Error:', np.mean(train_mae_metric_results))
    print('Mean Squared Error:', (np.mean(train_mse_metric_results)))
    print('Root Mean Squared Error:', (np.mean(train_mse_metric_results)**(1/2)))
    
    print('Validation')
    print('Cross-validation result (R-square):', np.mean(test_R2_metric_results))
    print('Cross-validation result (MAE):', np.mean(test_mae_metric_results))
    print('Cross-validation result (MSE):', (np.mean(test_mse_metric_results)))
    print('Cross-validation result (RMSE):', (np.mean(test_mse_metric_results)**(1/2)))
    
    return {
        'train_r2': np.mean(train_R2_metric_results),
        'train_mae': np.mean(train_mae_metric_results),
        'train_mse': np.mean(train_mse_metric_results),
        'train_rmse': np.mean(train_mse_metric_results)**(1/2),
        'test_r2': np.mean(test_R2_metric_results),
        'test_mae': np.mean(test_mae_metric_results),
        'test_mse': np.mean(test_mse_metric_results),
        'test_rmse': np.mean(test_mse_metric_results)**(1/2),
    }

def plot_actual_vs_predicted_R(X_analize, target_R, random_state=123):
    """
    Creates a scatter plot of actual vs predicted values for R target.
    
    Args:
        X_analize: DataFrame with features
        target_R: Series with target values
        random_state: Random seed for reproducibility
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_analize, target_R, test_size=0.2, random_state=random_state
    )
    
    model = lightgbm.LGBMRegressor(
        n_jobs=-1,
        objective='regression',
        num_leaves=112,
        learning_rate=0.1,
        feature_fraction=0.9855475553463364,
        bagging_fraction=0.8069570540135589,
        min_data_in_leaf=5,
        metric='rmse',
        custom_eval_metric_name=None,
        lambda_l1=0.0026686145190637704,
        lambda_l2=8.730406606367973e-06,
        bagging_freq=7,
        extra_trees=False,
        num_boost_round=1000,
        feature_pre_filter=False,
        explain_level=0,
        verbose=-1,
    )
    
    stop_callback = early_stopping(stopping_rounds=1000)
    model.fit(
        X_train, Y_train,
        eval_set=[(X_test, Y_test)],
        callbacks=[stop_callback],
    )
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set(font_scale=2)
    
    f, ax = plt.subplots(figsize=(13, 10), facecolor='white')
    ax.set_facecolor('white')
    ax.grid(False)
    
    plt.scatter(Y_train, train_preds, color='#DD7059', s=70, label='train data')
    plt.scatter(Y_test, test_preds, color='#569FC9', s=70, label='validation')
    plt.plot(Y_test, Y_test, color='#444444', linewidth=3)
    
    plt.xlabel('actual data')
    plt.ylabel('predicted data')
    plt.legend()
    plt.xlim(-1, 105)
    plt.ylim(-1, 105)
    
    plt.show()

def train_and_save_final_model_R(X_analize, target_R, model_filename='lightgbm_model_final2.joblib'):
    """
    Trains a final model for R target on all data and saves it to a file.
    
    Args:
        X_analize: DataFrame with features
        target_R: Series with target values
        model_filename: Filename to save the model
    """
    final_model = lightgbm.LGBMRegressor(
        n_jobs=-1,
        objective='regression',
        num_leaves=112,
        learning_rate=0.1,
        feature_fraction=0.9855475553463364,
        bagging_fraction=0.8069570540135589,
        min_data_in_leaf=5,
        metric='rmse',
        custom_eval_metric_name=None,
        lambda_l1=0.0026686145190637704,
        lambda_l2=8.730406606367973e-06,
        bagging_freq=7,
        extra_trees=False,
        num_boost_round=1000,
        feature_pre_filter=False,
        explain_level=0,
        verbose=-1,
    )
    
    final_model.fit(X_analize, target_R)
    
    dump(final_model, model_filename)
    print(f"Финальная модель для R, обученная на всех данных, сохранена как '{model_filename}'")
    
    return final_model

def analyze_shap_values_R(X_analize, target_R, n_splits=5, n_repeats=20, random_state=42):
    """
    Analyzes feature importance for R target using SHAP values across multiple cross-validation splits.
    
    Args:
        X_analize: DataFrame with features
        target_R: Series with target values
        n_splits: Number of folds for cross-validation
        n_repeats: Number of repetitions for cross-validation
        random_state: Random seed for reproducibility
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    shap_values_all = []
    X_test_all = []
    
    for train_index, test_index in rkf.split(X_analize):
        X_train, X_test = X_analize.iloc[train_index], X_analize.iloc[test_index]
        y_train, y_test = target_R.iloc[train_index], target_R.iloc[test_index]
        
        model = lightgbm.LGBMRegressor(
            n_jobs=-1,
            objective='regression',
            num_leaves=112,
            learning_rate=0.1,
            feature_fraction=0.9855475553463364,
            bagging_fraction=0.8069570540135589,
            min_data_in_leaf=5,
            metric='rmse',
            custom_eval_metric_name=None,
            lambda_l1=0.0026686145190637704,
            lambda_l2=8.730406606367973e-06,
            bagging_freq=7,
            extra_trees=False,
            num_boost_round=1000,
            feature_pre_filter=False,
            explain_level=0,
            verbose=-1,
        )
        
        model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        
        shap_values_all.append(shap_values.values)
        X_test_all.append(X_test)
    
    shap_values_combined = np.vstack(shap_values_all)
    X_test_combined = pd.concat(X_test_all, axis=0)
    
    shap_values_explanation = shap.Explanation(
        values=shap_values_combined,
        base_values=np.zeros(shap_values_combined.shape[0]),
        data=X_test_combined.values,
        feature_names=X_analize.columns.tolist()
    )
    
    print("\nSHAP values for R")
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    feature_mask = np.ones(shap_values_explanation.values.shape[1], dtype=bool)
    feature_mask[:1] = False  # Исключаем первый признак
    
    filtered_shap_values = shap_values_explanation.values[:, feature_mask]
    filtered_data = X_test_combined.iloc[:, feature_mask]
    filtered_feature_names = X_test_combined.columns[feature_mask].tolist()
    
    filtered_explanation = shap.Explanation(
        values=filtered_shap_values,
        base_values=np.zeros(filtered_shap_values.shape[0]),
        data=filtered_data.values,
        feature_names=filtered_feature_names
    )
    
    plt.figure(figsize=(4, 10))
    shap.summary_plot(
        filtered_explanation,
        filtered_data,
        plot_type="bar",
        show=False,
        color_bar_label='Feature value',
        alpha=1,
        plot_size=1
    )
    
    return filtered_explanation, filtered_data

if __name__ == "__main__":
    metrics_R = train_and_evaluate_model_R(X_analize, target_R)
    plot_actual_vs_predicted_R(X_analize, target_R)
    final_model_R = train_and_save_final_model_R(X_analize, target_R)
    shap_explanation_R, X_test_filtered_R = analyze_shap_values_R(X_analize, target_R)
