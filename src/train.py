import pandas as pd
import numpy as np
import joblib # For saving/loading models
import mlflow
import mlflow.sklearn # Ensure MLflow's sklearn integration is available

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Import the data processing function
from src.data_processing import process_data

# Set MLflow tracking URI (if not set via environment variable)
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Credit_Risk_Model_Training")


def train_and_log_model(model_name: str, model_instance, X_train, y_train, X_test, y_test, params=None):
    """
    Trains a model, evaluates it, and logs the results to MLflow.
    
    :param model_name: Name of the model (e.g., "Logistic Regression", "Random Forest").
    :param model_instance: The sklearn model instance to train (can be a GridSearchCV/RandomizedSearchCV object).
    :param X_train: Training features.
    :param y_train: Training target.
    :param X_test: Testing features.
    :param y_test: Testing target.
    :param params: Dictionary of parameters to log specifically for the model (e.g., best params from tuning).
    :return: A tuple of (trained_model, roc_auc_score, run_id).
    """
    with mlflow.start_run(run_name=f"{model_name}_Training") as run:
        run_id = run.info.run_id
        print(f"\n--- Starting MLflow Run for {model_name} (Run ID: {run_id}) ---")

        mlflow.log_param("model_type", model_name)

        if params:
            mlflow.log_params(params)
            print(f"Logged custom parameters: {params}")

        print(f"Training {model_name}...")
        model_instance.fit(X_train, y_train)
        print(f"{model_name} training complete.")

        if hasattr(model_instance, 'best_estimator_'):
            trained_model = model_instance.best_estimator_
            mlflow.log_params(model_instance.best_params_)
            print(f"Logged best parameters from tuning: {model_instance.best_params_}")
            print(f"Best score from tuning: {model_instance.best_score_:.4f}")
        else:
            trained_model = model_instance

        print(f"Evaluating {model_name}...")
        y_pred = trained_model.predict(X_test)
        
        # Ensure y_proba is calculated only if the model supports it and there are two classes in y_test
        y_proba = None
        if hasattr(trained_model, 'predict_proba') and len(np.unique(y_test)) > 1:
            y_proba = trained_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        roc_auc = 0.0 # Default if not calculable
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except ValueError as e:
                print(f"Warning: Could not calculate ROC AUC score: {e}")
                roc_auc = 0.0 # Set to 0 if ROC AUC is undefined

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc_score": roc_auc
        }
        mlflow.log_metrics(metrics)
        print(f"Logged metrics: {metrics}")

        print("\n--- Evaluation Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

        # mlflow.sklearn.log_model(trained_model, "model")
        mlflow.sklearn.log_model(trained_model, artifact_path="model", registered_model_name="CreditRiskProxyModel")
        print(f"Logged {model_name} model.")

        return trained_model, roc_auc, run_id


def train_models_and_track(df_raw: pd.DataFrame, target_column: str = 'is_high_risk'):
    """
    Orchestrates the data processing, model training, hyperparameter tuning,
    and MLflow tracking for multiple models.
    Identifies and registers the best model.
    """
    print("\n--- Preparing Data for Model Training ---")
    processed_df = process_data(df_raw.copy())
    
    if processed_df is None:
        print("Data processing failed. Aborting model training.")
        return

    X = processed_df.drop(columns=[target_column])
    y = processed_df[target_column]

    # Check if there's enough data for splitting and stratification
    # Ensure at least 2 samples for each class for train_test_split with stratify
    if len(np.unique(y)) < 2 or (y.value_counts().min() < 2): # Check min count for any class
        print(f"Error: Not enough samples for both classes to perform stratified train-test split or cross-validation. "
              f"Current target distribution: {y.value_counts()}. "
              "Please ensure at least 2 samples for each class in your data.")
        return # Abort if data is insufficient
    
    # Adjusted test_size for better handling of dummy data with stratification
    # For a small dataset, a larger test_size (e.g., 0.4 or 0.5) might be needed
    # to ensure both classes are present in both train and test sets.
    # With the new larger dummy data, 0.2 should be fine.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


    best_roc_auc = -1
    best_model_info = None # (model_instance, run_id, model_type)

    # --- Model 1: Logistic Regression ---
    print("\n--- Training Logistic Regression Model ---")
    lr_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear']
    }
    # Adjusted cv for very small datasets. For real data, use higher cv (e.g., 5 or 10)
    # n_jobs=-1 uses all available CPU cores.
    lr_grid_search = GridSearchCV(
        LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        lr_params, cv=2, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    lr_model, lr_roc_auc, lr_run_id = train_and_log_model(
        "Logistic Regression", lr_grid_search, X_train, y_train, X_test, y_test
    )
    if lr_roc_auc > best_roc_auc:
        best_roc_auc = lr_roc_auc
        best_model_info = (lr_model, lr_run_id, "Logistic Regression")

    # --- Model 2: Random Forest Classifier ---
    print("\n--- Training Random Forest Classifier Model ---")
    rf_params = {
        'n_estimators': sp_randint(10, 100),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': sp_randint(3, 10),
        'min_samples_split': sp_randint(2, 10),
        'min_samples_leaf': sp_randint(1, 5),
        'criterion': ['gini', 'entropy']
    }
    # n_iter: Number of parameter settings that are sampled.
    rf_random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        rf_params, n_iter=10, cv=2, scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42
    )
    rf_model, rf_roc_auc, rf_run_id = train_and_log_model(
        "Random Forest", rf_random_search, X_train, y_train, X_test, y_test
    )
    if rf_roc_auc > best_roc_auc:
        best_roc_auc = rf_roc_auc
        best_model_info = (rf_model, rf_run_id, "Random Forest")

    # --- Register the Best Model ---
    if best_model_info:
        model_to_register, best_run_id, model_type = best_model_info
        print(f"\nBest model identified: {model_type} with ROC AUC: {best_roc_auc:.4f}")
        
        model_uri = f"runs:/{best_run_id}/model"
        print(f"Registering best model from run {best_run_id} to MLflow Model Registry...")
        
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="CreditRiskProxyModel",
            tags={"model_type": model_type, "target_proxy": "is_high_risk"}
        )
        print(f"Model registered as: {registered_model.name} version {registered_model.version}")
        print("To view MLflow UI, run 'mlflow ui' in your terminal from the project root.")


# Example usage (for testing within the script or in a notebook)
if __name__ == '__main__':
    # Enhanced dummy DataFrame for better testing of train_test_split and cross-validation
    # Aim for more customers and a more balanced distribution of high_risk
    num_transactions = 200 # Increased total transactions
    num_customers = 100 # Increased unique customers

    # Generate transaction data
    transaction_data = {
        'TransactionId': [f'T{i}' for i in range(1, num_transactions + 1)],
        'BatchId': [f'B{(i-1)//5 + 1}' for i in range(1, num_transactions + 1)],
        'AccountId': [f'A{(i-1)//2 + 1}' for i in range(1, num_transactions + 1)],
        'SubscriptionId': [f'S{(i-1)//2 + 1}' for i in range(1, num_transactions + 1)],
        'CustomerId': [f'C{np.random.randint(1, num_customers + 1)}' for _ in range(num_transactions)], # Randomly assign customers
        'CurrencyCode': np.random.choice(['USD', 'EUR', 'GBP'], num_transactions).tolist(),
        'CountryCode': np.random.randint(1, 6, num_transactions).tolist(),
        'ProviderId': np.random.choice(['P1', 'P2', 'P3'], num_transactions).tolist(),
        'ProductId': np.random.choice(['ProdA', 'ProdB', 'ProdC', 'ProdD'], num_transactions).tolist(),
        'ProductCategory': np.random.choice(['CatX', 'CatY', 'CatZ'], num_transactions).tolist(),
        'ChannelId': np.random.choice(['Ch1', 'Ch2'], num_transactions).tolist(),
        'Amount': np.random.uniform(10, 5000, num_transactions).tolist(),
        'Value': np.random.uniform(10, 5000, num_transactions).tolist(), 
        'TransactionStartTime': [
            (pd.to_datetime('2022-01-01') + pd.to_timedelta(np.random.randint(0, 365*2), unit='D') + 
             pd.to_timedelta(np.random.randint(0, 24), unit='H')).strftime('%Y-%m-%d %H:%M:%S') for _ in range(num_transactions)
        ],
        'PricingStrategy': np.random.randint(1, 4, num_transactions).tolist(),
        'FraudResult': np.random.choice([0, 1], num_transactions, p=[0.9, 0.1]).tolist() # Lower fraud rate to simulate real data
    }
    dummy_df = pd.DataFrame(transaction_data)

    # --- Manually create some "high-risk" customers for RFM to ensure stratification works ---
    # These customers will have older, fewer, smaller transactions
    high_risk_customer_ids = [f'HR_C{i}' for i in range(1, 11)] # 10 high-risk customers
    
    high_risk_transactions = []
    for i, cid in enumerate(high_risk_customer_ids):
        for j in range(np.random.randint(1, 3)): # 1 or 2 transactions per high-risk customer
            transaction_time = (pd.to_datetime('2021-01-01') + pd.to_timedelta(np.random.randint(0, 180), unit='D') + 
                                pd.to_timedelta(np.random.randint(0, 24), unit='H')).strftime('%Y-%m-%d %H:%M:%S')
            high_risk_transactions.append({
                'TransactionId': f'HR_T{i}_{j}',
                'BatchId': f'HR_B{i}',
                'AccountId': f'HR_A{i}',
                'SubscriptionId': f'HR_S{i}',
                'CustomerId': cid,
                'CurrencyCode': 'USD',
                'CountryCode': 1,
                'ProviderId': 'P1',
                'ProductId': 'ProdA',
                'ProductCategory': 'CatX',
                'ChannelId': 'Ch1',
                'Amount': np.random.uniform(10, 100), # Low monetary
                'Value': np.random.uniform(10, 100),
                'TransactionStartTime': transaction_time, # Old recency
                'PricingStrategy': 1,
                'FraudResult': 0 # Not necessarily fraudulent, just disengaged
            })
    
    dummy_df_high_risk = pd.DataFrame(high_risk_transactions)
    dummy_df = pd.concat([dummy_df, dummy_df_high_risk], ignore_index=True)

    print("--- Running data processing with enhanced dummy data (with proxy target) ---")
    train_models_and_track(dummy_df.copy())
