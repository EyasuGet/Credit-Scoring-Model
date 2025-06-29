# Credit-Scoring-Model

This repository houses a credit risk modeling project, focusing on the development of a robust and interpretable credit scoring system.

---

## Project Structure

The project adheres to a standardized structure for clarity, maintainability, and ease of collaboration:

credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                      # Raw and processed data (added to .gitignore)
│   ├── raw/                   # Raw data goes here
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── init.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

---
