from pydantic import BaseModel, Field
from typing import List, Optional

class TransactionInput(BaseModel):
    # Customer and transaction identifiers (will be used for aggregation)
    TransactionId: str = Field(..., description="Unique identifier for the transaction")
    BatchId: str = Field(..., description="Batch identifier for the transaction")
    AccountId: str = Field(..., description="Account identifier associated with the transaction")
    SubscriptionId: str = Field(..., description="Subscription identifier for the transaction")
    CustomerId: str = Field(..., description="Unique identifier for the customer")

    # Transaction details
    CurrencyCode: str = Field(..., description="Currency code of the transaction (e.g., USD, EUR)")
    CountryCode: int = Field(..., description="Country code of the transaction")
    ProviderId: str = Field(..., description="Provider identifier")
    ProductId: str = Field(..., description="Product identifier")
    ProductCategory: str = Field(..., description="Category of the product")
    ChannelId: str = Field(..., description="Channel through which the transaction occurred")
    Amount: float = Field(..., description="Amount of the transaction")
    Value: float = Field(..., description="Value of the transaction (should be identical to Amount)")
    TransactionStartTime: str = Field(..., description="Timestamp of the transaction (YYYY-MM-DD HH:MM:SS)")
    PricingStrategy: int = Field(..., description="Pricing strategy used for the transaction")
    FraudResult: int = Field(..., description="Original fraud result (0 for no fraud, 1 for fraud). Note: This will be dropped during processing.")

    class Config:
        schema_extra = {
            "example": {
                "TransactionId": "T123456",
                "BatchId": "B789",
                "AccountId": "A101",
                "SubscriptionId": "S202",
                "CustomerId": "C303",
                "CurrencyCode": "USD",
                "CountryCode": 254,
                "ProviderId": "P1",
                "ProductId": "ProdA",
                "ProductCategory": "Gaming",
                "ChannelId": "Web",
                "Amount": 150.75,
                "Value": 150.75,
                "TransactionStartTime": "2023-11-20 14:30:00",
                "PricingStrategy": 2,
                "FraudResult": 0
            }
        }

# Define a Pydantic model for the request body, which will be a list of transactions.
class PredictionRequest(BaseModel):
    transactions: List[TransactionInput] = Field(..., description="A list of transaction records for which to predict credit risk.")

    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "TransactionId": "T123456",
                        "BatchId": "B789",
                        "AccountId": "A101",
                        "SubscriptionId": "S202",
                        "CustomerId": "C303",
                        "CurrencyCode": "USD",
                        "CountryCode": 254,
                        "ProviderId": "P1",
                        "ProductId": "ProdA",
                        "ProductCategory": "Gaming",
                        "ChannelId": "Web",
                        "Amount": 150.75,
                        "Value": 150.75,
                        "TransactionStartTime": "2023-11-20 14:30:00",
                        "PricingStrategy": 2,
                        "FraudResult": 0
                    },
                    {
                        "TransactionId": "T123457",
                        "BatchId": "B789",
                        "AccountId": "A101",
                        "SubscriptionId": "S202",
                        "CustomerId": "C303",
                        "CurrencyCode": "EUR",
                        "CountryCode": 33,
                        "ProviderId": "P2",
                        "ProductId": "ProdB",
                        "ProductCategory": "E-commerce",
                        "ChannelId": "Mobile",
                        "Amount": 500.00,
                        "Value": 500.00,
                        "TransactionStartTime": "2023-11-20 15:00:00",
                        "PricingStrategy": 1,
                        "FraudResult": 0
                    }
                ]
            }
        }


