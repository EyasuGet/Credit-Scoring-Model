import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc
import joblib
import tempfile
import shutil

from src.data_processing import process_data

from src.api.pydantic_models import PredictionRequest, PredictionResponse, TransactionInput

# --- Set MLflow Tracking URI explicitly and early for the API ---
project_root = "/home/eyuleo/Documents/kifya/Credit-Scoring-Model"
mlruns_path = os.path.join(project_root, "mlruns")

mlflow.set_tracking_uri(f"file://{mlruns_path}")
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_path}" 
print(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")


# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk (high-risk vs. low-risk) based on customer transaction data.",
    version="1.0.0"
)

MLFLOW_MODEL_NAME = "CreditRiskProxyModel"
MLFLOW_RUN_ID_FOR_MODEL = "6fd7c6e772984b50992c4c245fdd2406"
MLFLOW_ARTIFACT_PATH = "" 

try:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(MLFLOW_RUN_ID_FOR_MODEL)
    experiment_id = run.info.experiment_id
except Exception as e:
    print(f"Could not retrieve experiment ID for run {MLFLOW_RUN_ID_FOR_MODEL}: {e}")
    print("Falling back to a more direct path construction, assuming default MLflow structure.")

    experiment_dirs = [d for d in os.listdir(mlruns_path) if os.path.isdir(os.path.join(mlruns_path, d)) and d.isdigit()]
    if experiment_dirs:

        experiment_id = experiment_dirs[0]
        print(f"Heuristically determined experiment ID: {experiment_id}")
    else:
        raise RuntimeError("Could not determine MLflow experiment ID. Ensure mlruns structure is standard.")


# Construct the full absolute path to model.pkl
# This path must be valid *inside the Docker container*
# which means it must match the mounted host path.
# absolute_model_pkl_path = os.path.join(
#     mlruns_path,
#     experiment_id,
#     MLFLOW_RUN_ID_FOR_MODEL,
#     "artifacts",
#     MLFLOW_ARTIFACT_PATH, # This is now ""
#     "model.pkl" # The actual pickled model file name
# )

# model_version_str = f"Direct PKL Load from {MLFLOW_RUN_ID_FOR_MODEL}"
absolute_model_pkl_path = "/home/eyuleo/Documents/kifya/Credit-Scoring-Model/mlruns/820322135837684092/models/m-3d81bc9bd96a4d2b85b32ad107fdeded/artifacts/model.pkl"
model_version_str = "Direct PKL Load from registered model m-3d81bc9bd96a4d2b85b32ad107fdeded"


# Global variable to hold the loaded model
model = None


@app.on_event("startup")
async def load_model():
    """
    Load the MLflow model by directly loading the model.pkl file.
    This bypasses MLflow's artifact resolution mechanism for problematic environments.
    """
    global model
    print(f"Loading model: {model_version_str} from {absolute_model_pkl_path}...")
    try:
        if not os.path.exists(absolute_model_pkl_path):
            raise FileNotFoundError(f"Model PKL file not found at: {absolute_model_pkl_path}")
            
        # Load the model directly using joblib
        model = joblib.load(absolute_model_pkl_path)
        
        print(f"Model from {absolute_model_pkl_path} loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {absolute_model_pkl_path}: {e}")
        raise RuntimeError(f"Failed to load MLflow model: {e}")

@app.get("/health", summary="Health Check", response_model=dict)
async def health_check():
    """
    Health check endpoint to verify if the API is running and the model is loaded.
    """
    if model is not None:
        return {"status": "ok", "model_loaded": True, "model_version": model_version_str}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

@app.post("/predict", summary="Predict Credit Risk", response_model=PredictionResponse)
async def predict_credit_risk(request: PredictionRequest):
    """
    Receives a list of raw transaction data for a customer, processes it,
    and returns the predicted credit risk probability and label.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    raw_transactions_data = [t.dict() for t in request.transactions]
    df_raw_input = pd.DataFrame(raw_transactions_data)

    if 'CustomerId' not in df_raw_input.columns or df_raw_input['CustomerId'].nunique() != 1:
        raise HTTPException(
            status_code=400,
            detail="Prediction request must contain transactions for exactly one CustomerId."
        )
    customer_id = df_raw_input['CustomerId'].iloc[0]

    try:
        processed_df_for_prediction = process_data(df_raw_input.copy())
        
        if 'is_high_risk' in processed_df_for_prediction.columns:
            X_predict = processed_df_for_prediction.drop(columns=['is_high_risk'])
        else:
            raise HTTPException(status_code=500, detail="Processed data missing 'is_high_risk' column.")

        if X_predict.shape[0] != 1:
            raise HTTPException(
                status_code=500,
                detail="Internal processing error: Expected single customer row after aggregation."
            )
        
        high_risk_probability = model.predict_proba(X_predict)[0][1]
        is_high_risk_label = int(model.predict(X_predict)[0])

    except Exception as e:
        print(f"Error during data processing or prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal processing error: {e}")

    return PredictionResponse(
        customer_id=customer_id,
        is_high_risk_probability=float(high_risk_probability),
        is_high_risk_label=is_high_risk_label,
        model_version=model_version_str
    )