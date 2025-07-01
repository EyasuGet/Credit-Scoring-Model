import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime


class RFMCalculator:
    """
    Calculates Recency, Frequency, and Monetary (RFM) metrics for customers
    from raw transaction data.
    """
    def __init__(self, customer_id_col='CustomerId', transaction_time_col='TransactionStartTime', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.transaction_time_col = transaction_time_col
        self.amount_col = amount_col

    def calculate(self, df_raw: pd.DataFrame, snapshot_date: datetime.datetime = None) -> pd.DataFrame:
        """
        Calculates RFM metrics.
        :param df_raw: The raw transactions DataFrame.
        :param snapshot_date: The date to calculate Recency against. If None, uses one day after max transaction date.
        :return: DataFrame with CustomerId and RFM values.
        """
        df_rfm = df_raw.copy()

        df_rfm[self.transaction_time_col] = pd.to_datetime(df_rfm[self.transaction_time_col], errors='coerce')

        # Drop rows 
        df_rfm = df_rfm.dropna(subset=[self.transaction_time_col])

        if df_rfm.empty:
            raise ValueError("DataFrame is empty after dropping invalid transaction times. Cannot calculate RFM.")

        if snapshot_date is None:
            snapshot_date = df_rfm[self.transaction_time_col].max() + datetime.timedelta(days=1)
        
        print(f"Using snapshot date for RFM calculation: {snapshot_date}")

        # Group by customer and calculate RFM components
        rfm_table = df_rfm.groupby(self.customer_id_col).agg(
            Recency=(self.transaction_time_col, lambda date: (snapshot_date - date.max()).days),
            Frequency=(self.customer_id_col, 'count'),
            Monetary=(self.amount_col, 'sum')
        ).reset_index()

        rfm_table['Recency'] = rfm_table['Recency'].apply(lambda x: max(x, 1))

        return rfm_table


class CustomerSegmenter:
    """
    Segments customers using K-Means clustering on RFM metrics and assigns a 'high-risk' label.
    """
    def __init__(self, n_clusters=3, random_state=42, customer_id_col='CustomerId'): # Added customer_id_col here
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.scaler = None
        self.cluster_centroids = None
        self.high_risk_cluster_label = None
        self.customer_id_col = customer_id_col # Stored customer_id_col

    def segment(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs K-Means clustering on RFM data and identifies the high-risk cluster.
        :param rfm_df: DataFrame with CustomerId, Recency, Frequency, Monetary columns.
        :return: DataFrame with CustomerId and 'is_high_risk' binary label.
        """
        rfm_data = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()

        # 1. Handle skewness: Apply log transformation (add 1 to handle zero values)

        rfm_data['Recency_log'] = np.log1p(rfm_data['Recency'])
        rfm_data['Frequency_log'] = np.log1p(rfm_data['Frequency'])
        rfm_data['Monetary_log'] = np.log1p(rfm_data['Monetary'])
        
        rfm_processed = rfm_data[['Recency_log', 'Frequency_log', 'Monetary_log']]

        # 2. Scale RFM features
        self.scaler = StandardScaler()
        rfm_scaled = self.scaler.fit_transform(rfm_processed)
        rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_processed.columns, index=rfm_processed.index)

        # 3. K-Means clustering
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10) 
        rfm_df['Cluster'] = self.kmeans_model.fit_predict(rfm_scaled_df)

        # 4. Analyze cluster centroids to define 'high-risk'
        scaled_centroids = self.kmeans_model.cluster_centers_
        self.cluster_centroids = pd.DataFrame(self.scaler.inverse_transform(scaled_centroids), 
                                                columns=rfm_processed.columns)
      
        cluster_scores = scaled_centroids[:, 0] - scaled_centroids[:, 1] - scaled_centroids[:, 2] # Recency_log - Frequency_log - Monetary_log
        self.high_risk_cluster_label = np.argmax(cluster_scores)

        print("\nK-Means Cluster Centroids (Inverse Transformed to Original Scale):")
        print(self.cluster_centroids)
        print(f"\nIdentified High-Risk Cluster Label: {self.high_risk_cluster_label}")


        # 5. Create 'is_high_risk' binary column
        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == self.high_risk_cluster_label).astype(int)

        # Return CustomerId and the new target variable
        return rfm_df[[self.customer_id_col, 'is_high_risk']]


def get_proxy_target(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to orchestrate the proxy target variable engineering.
    Calculates RFM, clusters customers, and assigns the 'is_high_risk' label.
    :param df_raw: The raw transactions DataFrame.
    :return: DataFrame with CustomerId and 'is_high_risk' binary label.
    """
    print("\n--- Task 4: Proxy Target Variable Engineering ---")
    
    # Calculate RFM Metrics
    rfm_calculator = RFMCalculator()
    rfm_df = rfm_calculator.calculate(df_raw)
    print("\nRFM Calculation Complete. Sample RFM Data:")
    print(rfm_df.head())

    customer_segmenter = CustomerSegmenter(n_clusters=3, random_state=42, customer_id_col=rfm_calculator.customer_id_col)
    high_risk_df = customer_segmenter.segment(rfm_df)
    
    print("\nHigh-Risk Label Assignment Complete. Sample High-Risk Data:")
    print(high_risk_df.head())
    print("\nHigh-Risk Cluster Distribution:")
    print(high_risk_df['is_high_risk'].value_counts())

    return high_risk_df

if __name__ == '__main__':
    # Create a dummy DataFrame mimicking your raw data structure
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'],
        'BatchId': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12'],
        'AccountId': ['A1', 'A1', 'A2', 'A2', 'A3', 'A3', 'A4', 'A4', 'A5', 'A5', 'A1', 'A2'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S5', 'S5', 'S1', 'S2'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C5', 'C1', 'C2'],
        'CurrencyCode': ['USD', 'USD', 'EUR', 'USD', 'EUR', 'GBP', 'USD', 'EUR', 'USD', 'GBP', 'USD', 'EUR'],
        'CountryCode': [1, 1, 2, 1, 2, 3, 1, 2, 1, 3, 1, 2],
        'ProviderId': ['P1', 'P1', 'P2', 'P1', 'P2', 'P3', 'P1', 'P2', 'P1', 'P3', 'P1', 'P2'],
        'ProductId': ['ProdA', 'ProdA', 'ProdB', 'ProdA', 'ProdB', 'ProdC', 'ProdA', 'ProdB', 'ProdA', 'ProdC', 'ProdA', 'ProdB'],
        'ProductCategory': ['CatX', 'CatX', 'CatY', 'CatX', 'CatY', 'CatZ', 'CatX', 'CatY', 'CatX', 'CatZ', 'CatX', 'CatY'],
        'ChannelId': ['Ch1', 'Ch1', 'Ch2', 'Ch1', 'Ch2', 'Ch3', 'Ch1', 'Ch2', 'Ch1', 'Ch3', 'Ch1', 'Ch2'],
        'Amount': [100.5, 200.0, 50.0, 15000.0, 75.0, 300.0, 120.0, 60.0, 180.0, 350.0, 110.0, 80.0],
        'Value': [100.5, 200.0, 50.0, 15000.0, 75.0, 300.0, 120.0, 60.0, 180.0, 350.0, 110.0, 80.0],
        'TransactionStartTime': [
            '2023-01-01 10:00:00', '2023-01-05 11:30:00', # C1: Recent, Avg F, High M
            '2022-03-10 14:00:00', '2022-03-15 09:00:00', # C2: Old, Low F, Low M (potential high risk)
            '2023-05-01 16:00:00', '2023-05-05 10:00:00', # C3: Recent, Avg F, Avg M
            '2023-06-01 12:00:00', '2023-06-02 13:00:00', # C4: Very Recent, High F, Avg M
            '2022-01-01 17:00:00', '2022-01-05 18:00:00', # C5: Very Old, Low F, Low M (potential high risk)
            '2023-01-06 09:00:00', # C1 additional, makes C1 higher F and M
            '2022-03-16 14:00:00'  # C2 additional, makes C2 higher F
        ],
        'PricingStrategy': [1, 1, 2, 1, 2, 3, 1, 2, 1, 3, 1, 2],
        'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    dummy_df = pd.DataFrame(data)

    print("--- Running Proxy Target Engineering with dummy data for testing ---")
    high_risk_customers_df = get_proxy_target(dummy_df.copy())
    if high_risk_customers_df is not None:
        print("\n--- Final High-Risk Customers Data (first 5 rows) ---")
        print(high_risk_customers_df.head())
        print("\n--- High-Risk Customers Info ---")
        high_risk_customers_df.info()
        print("\n--- High-Risk Customers Describe ---")
        print(high_risk_customers_df.describe())
