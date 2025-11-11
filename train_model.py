import pandas as pd
import joblib
import logging
import utils 
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor 
from sklearn.metrics import mean_absolute_error, r2_score
import shap 

# Set up logging for train script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_FILENAME = "final_ev_consumption_model_xgb.pkl" 
N_JOBS = -1 
N_ITER_SEARCH = 20 

def create_pipeline(features_config):
    """Creates the full ColumnTransformer and ML Pipeline."""
    
    base_num_features = features_config['numerical_features'] 
    cat_features = features_config['categorical_features']
    engineered_feature = features_config['new_engineered_features'] 
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    engineered_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, base_num_features),
            ('cat', categorical_transformer, cat_features),
            ('eng', engineered_transformer, engineered_feature) 
        ],
        remainder='drop',
        n_jobs=N_JOBS
    )

    model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=N_JOBS)
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    return full_pipeline

def train_and_evaluate(df, features_config, full_pipeline):
    """Trains the model with hyperparameter tuning, evaluates, and calculates SHAP values."""
    
    all_input_cols = (
        features_config['numerical_features'] + 
        features_config['categorical_features'] +
        features_config['new_engineered_features'] 
    )
    Y_col = features_config['target_column']
    
    X = df[all_input_cols]
    Y = df[Y_col]
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    logging.info("Starting Randomized Hyperparameter Tuning (XGBoost)...")
    
    param_distributions = {
        'regressor__n_estimators': [100, 250, 500, 1000],
        'regressor__max_depth': [3, 5, 7, 9],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__subsample': [0.6, 0.8, 1.0],
        'regressor__colsample_bytree': [0.6, 0.8, 1.0],
    }
    
    grid_search = RandomizedSearchCV(
        full_pipeline, 
        param_distributions, 
        n_iter=N_ITER_SEARCH, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        verbose=1,
        random_state=42,
        n_jobs=N_JOBS
    )
    
    grid_search.fit(X_train, Y_train) 
    
    best_pipeline = grid_search.best_estimator_
    
    logging.info(f"Best Parameters Found: {grid_search.best_params_}")
    
    # --- Evaluation ---
    Y_pred = best_pipeline.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    logging.info(f"Mean Absolute Error (MAE): {mae:.4f} KWh")
    logging.info(f"R-squared (R2): {r2:.4f}")
    
    # --- 1. Compute SHAP Values for Global Importance ---
    logging.info("Calculating SHAP Global Feature Importance...")
    X_processed = best_pipeline.named_steps['preprocessor'].transform(X_test)
    
    explainer = shap.TreeExplainer(best_pipeline.named_steps['regressor'])
    shap_values = explainer.shap_values(X_processed)
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Get processed feature names and sanitize them
    feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    sanitized_names = utils.sanitize_feature_names(feature_names)
    
    # --- 2. Save SHAP Importance to JSON ---
    shap_df = pd.DataFrame({'Feature': sanitized_names, 'SHAP_Importance': mean_abs_shap})
    shap_df = shap_df.sort_values(by='SHAP_Importance', ascending=False)
    shap_df.to_json('model_shap_importance.json', orient='records', indent=4)
    logging.info("Global SHAP importance saved to model_shap_importance.json.")

    return best_pipeline, mae

def main():
    """Main execution function."""
    try:
        df, features_config = utils.load_data()
    except Exception as e:
        logging.error(f"Terminating due to fatal data loading error: {e}")
        return
        
    try:
        df_engineered = utils.feature_engineer(df)
    except KeyError as e:
        logging.error(f"Feature engineering failed on base column: {e}")
        return
    
    full_pipeline = create_pipeline(features_config)
    
    final_pipeline, mae = train_and_evaluate(df_engineered, features_config, full_pipeline)
    
    joblib.dump(final_pipeline, MODEL_FILENAME)
    logging.info(f"SUCCESS: Final XGBoost model pipeline saved as {MODEL_FILENAME}")
    
    joblib.dump(mae, 'model_mae_consumption.pkl') 
    logging.info("SUCCESS: Model MAE saved.")

if __name__ == "__main__":
    main()