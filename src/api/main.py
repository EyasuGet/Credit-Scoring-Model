
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc
import joblib
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
MLFLOW_RUN_ID_FOR_MODEL = "743f17d1fbc24b79817859e655f9e291"
MLFLOW_ARTIFACT_PATH = "model" 

# Construct the direct artifact URI
model_uri_direct = f"runs:/{MLFLOW_RUN_ID_FOR_MODEL}/artifacts/{MLFLOW_ARTIFACT_PATH}"
model_version_str = f"Direct Load from Run {MLFLOW_RUN_ID_FOR_MODEL}" # For logging purposes

model = None


@app.on_event("startup")
async def load_model():
    """
    Load the MLflow model from the registry using a direct artifact URI.
    """
    global model
    print(f"Loading model: {model_version_str} from MLflow Model Registry...")
    try:
        model = mlflow.pyfunc.load_model(model_uri=model_uri_direct)
        print(f"Model from {model_uri_direct} loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {model_uri_direct}: {e}")
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