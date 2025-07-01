
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime 
try:
    from xverse import WOE
    print("Using WOE transformer from xverse.")
    XVERSE_WOE_AVAILABLE = True
except ImportError:
    print("xverse not found. A custom WOE transformer will be used instead.")
    XVERSE_WOE_AVAILABLE = False


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features (hour, day, month, year) from a datetime column.
    Drops the original datetime column after extraction.
    """
    def __init__(self, date_column='TransactionStartTime'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.date_column in X_copy.columns:

            X_copy[self.date_column] = pd.to_datetime(X_copy[self.date_column], errors='coerce')

            # Extract features. Use .dt accessor, and handle NaT if any
            X_copy['TransactionHour'] = X_copy[self.date_column].dt.hour
            X_copy['TransactionDay'] = X_copy[self.date_column].dt.day
            X_copy['TransactionMonth'] = X_copy[self.date_column].dt.month
            X_copy['TransactionYear'] = X_copy[self.date_column].dt.year

            # Drop the original datetime column
            X_copy = X_copy.drop(columns=[self.date_column])
        return X_copy


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops specified columns from the DataFrame.
    """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers in specified numerical columns using a percentile method.
    Values below lower_percentile are set to lower_bound, and values above
    upper_percentile are set to upper_bound.
    """
    def __init__(self, columns=None, upper_percentile=0.99, lower_percentile=0.01):
        self.columns = columns
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        self.upper_bounds = {}
        self.lower_bounds = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()

        for col in self.columns:
            if col in X.columns:
                # Calculate bounds only on non-null values
                self.upper_bounds[col] = X[col].quantile(self.upper_percentile)
                self.lower_bounds[col] = X[col].quantile(self.lower_percentile)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns and col in self.upper_bounds and col in self.lower_bounds:
                # Apply capping
                X_copy[col] = np.where(X_copy[col] > self.upper_bounds[col], self.upper_bounds[col], X_copy[col])
                X_copy[col] = np.where(X_copy[col] < self.lower_bounds[col], self.lower_bounds[col], X_copy[col])
        return X_copy


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction data to a customer-level DataFrame.
    Calculates aggregate numerical features.
    The fraud target (if original) is NOT determined here as we're using a proxy.
    Also merges static categorical customer-related features.
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'): # Removed fraud_col as it's now proxy
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Calculate aggregate numerical features for each customer
        customer_df = X.groupby(self.customer_id_col).agg(
            total_transaction_amount=(self.amount_col, 'sum'),
            average_transaction_amount=(self.amount_col, 'mean'),
            transaction_count=(self.amount_col, 'count'),
            std_transaction_amount=(self.amount_col, 'std')
        ).reset_index()

        # Handle potential NaNs in std_transaction_amount if a customer only has one transaction
        customer_df['std_transaction_amount'] = customer_df['std_transaction_amount'].fillna(0)

        # Merge relevant original customer-level features that are static per customer
        static_customer_features = [
            'CountryCode', 'CurrencyCode', 'ProviderId', 'ProductId',
            'ProductCategory', 'ChannelId', 'PricingStrategy',
            'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'
        ]
        
        # Filter for only existing columns in X
        static_customer_features = [col for col in static_customer_features if col in X.columns]

        # Get the first occurrence of these features for each customer from the preprocessed transaction data
        first_transaction_features = X.drop_duplicates(subset=[self.customer_id_col]).set_index(self.customer_id_col)[static_customer_features]
        customer_df = customer_df.merge(first_transaction_features, on=self.customer_id_col, how='left')

        return customer_df


class CustomWOETransformer(BaseEstimator, TransformerMixin):
    """
    A custom Weight of Evidence (WoE) transformer for categorical and numerical features.
    This implementation handles binning for numerical columns and calculates WoE based on a binary target.
    It's a fallback if xverse is not available.
    
    Includes capping of WoE values to prevent -inf or +inf.
    """
    def __init__(self, features=None, target_column='is_high_risk', numerical_bins=10, woe_cap=20):
        self.features = features
        self.target_column = target_column
        self.numerical_bins = numerical_bins
        self.woe_cap = woe_cap 
        self.woe_maps = {}
        self.iv_values = {}
        self.feature_is_numeric = {}
        self.numerical_bin_edges = {}

    def fit(self, X, y=None):
        if self.features is None:
            self.features = X.select_dtypes(include=['object', 'category', 'int64', 'float64']).columns.tolist()
            if self.target_column in self.features:
                self.features.remove(self.target_column)

        if y is None:
            raise ValueError("Target variable 'y' must be provided for WOE transformer fitting.")

        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        for col in self.features:
            if col not in X.columns:
                print(f"Warning: Column '{col}' not found in X for WOE calculation. Skipping.")
                continue

            temp_df = pd.DataFrame({col: X[col], 'target': y})
            temp_df['target'] = temp_df['target'].astype(int) 

            if pd.api.types.is_numeric_dtype(temp_df[col]):
                self.feature_is_numeric[col] = True
                try:
                    temp_df_no_nan = temp_df.dropna(subset=[col])
                    temp_df_no_nan['binned_col'], self.numerical_bin_edges[col] = pd.qcut(
                        temp_df_no_nan[col], q=self.numerical_bins, labels=False, duplicates='drop', retbins=True
                    )
                    temp_df = temp_df.merge(
                        temp_df_no_nan[['binned_col']], left_index=True, right_index=True, how='left'
                    )
                except Exception as e:
                    print(f"Warning: pd.qcut failed for numerical column '{col}' ({e}). Falling back to pd.cut.")
                    try:
                        n_unique = temp_df[col].nunique()
                        effective_bins = min(self.numerical_bins, n_unique)
                        if effective_bins == 0:
                            temp_df['binned_col'] = np.nan
                        else:
                            temp_df['binned_col'], self.numerical_bin_edges[col] = pd.cut(
                                temp_df[col], bins=effective_bins, labels=False, include_lowest=True, duplicates='drop', retbins=True
                            )
                    except Exception as e_cut:
                        print(f"Error: pd.cut also failed for numerical column '{col}' ({e_cut}). Column will be skipped for WOE.")
                        self.feature_is_numeric[col] = False
                        continue

                group_col = 'binned_col'
            else:
                self.feature_is_numeric[col] = False
                group_col = col

            grouped = temp_df.groupby(group_col)['target'].agg(
                total_count='count',
                bad_count=lambda x: (x == 1).sum(),
                good_count=lambda x: (x == 0).sum()
            ).reset_index()

            total_bad = grouped['bad_count'].sum()
            total_good = grouped['good_count'].sum()

            epsilon = 1e-6
            
            grouped['bad_rate'] = grouped['bad_count'] / (total_bad + epsilon)
            grouped['good_rate'] = grouped['good_count'] / (total_good + epsilon)

            grouped['woe'] = np.log((grouped['bad_rate'] + epsilon) / (grouped['good_rate'] + epsilon))
            grouped['woe'] = np.clip(grouped['woe'], -self.woe_cap, self.woe_cap)

            grouped['iv_contribution'] = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']
            self.iv_values[col] = grouped['iv_contribution'].sum()
            
            self.woe_maps[col] = grouped.set_index(group_col)['woe'].to_dict()

        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.features:
            if col not in X_copy.columns:
                continue

            if col in self.woe_maps:
                if self.feature_is_numeric.get(col, False):
                    if col not in self.numerical_bin_edges:
                        X_copy[col] = 0
                        continue
                    
                    binned_col_series = pd.cut(
                        X_copy[col], bins=self.numerical_bin_edges[col], labels=False,
                        include_lowest=True, right=True, duplicates='drop'
                    )
                    X_copy[col] = binned_col_series.map(self.woe_maps[col])
                else:
                    X_copy[col] = X_copy[col].map(self.woe_maps[col])
                
                X_copy[col] = X_copy[col].fillna(0)
            else:
                if pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = X_copy[col].fillna(0)
                else:
                    X_copy[col] = 0

        return X_copy

# --- Start of Proxy Target Engineering Classes (Moved from src/proxy_target_engineering.py) ---
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

        # 1. Convert TransactionStartTime to datetime objects
        df_rfm[self.transaction_time_col] = pd.to_datetime(df_rfm[self.transaction_time_col], errors='coerce')

        # Drop rows where transaction_time_col could not be parsed
        df_rfm = df_rfm.dropna(subset=[self.transaction_time_col])

        if df_rfm.empty:
            raise ValueError("DataFrame is empty after dropping invalid transaction times. Cannot calculate RFM.")

        # 2. Define a snapshot date if not provided
        if snapshot_date is None:
            # Use one day after the latest transaction as the snapshot date
            snapshot_date = df_rfm[self.transaction_time_col].max() + datetime.timedelta(days=1)
        
        print(f"Using snapshot date for RFM calculation: {snapshot_date}")

        # Group by customer and calculate RFM components
        rfm_table = df_rfm.groupby(self.customer_id_col).agg(
            # Recency: Days since last transaction
            Recency=(self.transaction_time_col, lambda date: (snapshot_date - date.max()).days),
            # Frequency: Number of transactions
            Frequency=(self.customer_id_col, 'count'),
            # Monetary: Sum of transaction amounts
            Monetary=(self.amount_col, 'sum')
        ).reset_index()

        # Handle cases where Recency might be negative or very small if snapshot_date is too close
        rfm_table['Recency'] = rfm_table['Recency'].apply(lambda x: max(x, 1)) # Ensure Recency is at least 1

        return rfm_table


class CustomerSegmenter:
    """
    Segments customers using K-Means clustering on RFM metrics and assigns a 'high-risk' label.
    """
    def __init__(self, n_clusters=3, random_state=42, customer_id_col='CustomerId'):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.scaler = None
        self.cluster_centroids = None
        self.high_risk_cluster_label = None
        self.customer_id_col = customer_id_col

    def segment(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs K-Means clustering on RFM data and identifies the high-risk cluster.
        :param rfm_df: DataFrame with CustomerId, Recency, Frequency, Monetary columns.
        :return: DataFrame with CustomerId and 'is_high_risk' binary label.
        """
        rfm_data = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
        
        rfm_data['Recency_log'] = np.log1p(rfm_data['Recency'])
        rfm_data['Frequency_log'] = np.log1p(rfm_data['Frequency'])
        rfm_data['Monetary_log'] = np.log1p(rfm_data['Monetary'])
        
        rfm_processed = rfm_data[['Recency_log', 'Frequency_log', 'Monetary_log']]

        self.scaler = StandardScaler()
        rfm_scaled = self.scaler.fit_transform(rfm_processed)
        rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_processed.columns, index=rfm_processed.index)

        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        rfm_df['Cluster'] = self.kmeans_model.fit_predict(rfm_scaled_df)

        scaled_centroids = self.kmeans_model.cluster_centers_
        self.cluster_centroids = pd.DataFrame(self.scaler.inverse_transform(scaled_centroids), 
                                                columns=rfm_processed.columns)
        
        cluster_scores = scaled_centroids[:, 0] - scaled_centroids[:, 1] - scaled_centroids[:, 2]
        self.high_risk_cluster_label = np.argmax(cluster_scores)

        print("\nK-Means Cluster Centroids (Inverse Transformed to Original Scale):")
        print(self.cluster_centroids)
        print(f"\nIdentified High-Risk Cluster Label: {self.high_risk_cluster_label}")

        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == self.high_risk_cluster_label).astype(int)

        return rfm_df[[self.customer_id_col, 'is_high_risk']]


def get_proxy_target(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to orchestrate the proxy target variable engineering.
    Calculates RFM, clusters customers, and assigns the 'is_high_risk' label.
    :param df_raw: The raw transactions DataFrame.
    :return: DataFrame with CustomerId and 'is_high_risk' binary label.
    """
    print("\n--- Task 4: Proxy Target Variable Engineering ---")
    
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
# --- End of Proxy Target Engineering Classes ---


def process_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the entire data processing pipeline from raw transaction data
    to a model-ready customer-level DataFrame with a proxy target.
    """
    if df_raw is None:
        print("Raw DataFrame is None. Cannot process data.")
        return None

    print("\nStep 1: Generating Proxy Target Variable ('is_high_risk')...")
    high_risk_customers_df = get_proxy_target(df_raw.copy())
    print("Proxy target generation complete. Shape:", high_risk_customers_df.shape)

    print("\nStep 2: Initial Transaction-Level Preprocessing (DateTime Extraction, Column Dropping)...")

    transaction_level_cols_to_drop = [
        'Value', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'FraudResult'
    ]

    pre_agg_pipeline = Pipeline([
        ('datetime_extractor', DateTimeFeatureExtractor(date_column='TransactionStartTime')),
        ('column_dropper_initial', ColumnDropper(columns_to_drop=transaction_level_cols_to_drop))
    ])

    df_preprocessed = pre_agg_pipeline.fit_transform(df_raw.copy())
    print("Initial preprocessing complete. Shape:", df_preprocessed.shape)

    print("\nStep 3: Aggregating data to Customer-Level and merging proxy target...")
    customer_aggregator = CustomerAggregator(customer_id_col='CustomerId', amount_col='Amount')
    X_customer = customer_aggregator.fit_transform(df_preprocessed)
    print("Customer-level aggregation complete. Shape:", X_customer.shape)

    X_processed = pd.merge(X_customer, high_risk_customers_df, on='CustomerId', how='left')
    
    y_final = X_processed['is_high_risk']
    X_final = X_processed.drop(columns=['is_high_risk', 'CustomerId']) 

    print("\nStep 4: Handling Outliers on Numerical Features...")
    numerical_features_after_agg = X_final.select_dtypes(include=np.number).columns.tolist()
    
    cappable_numerical_features = [
        col for col in numerical_features_after_agg
        if col not in ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    ]

    outlier_capper = OutlierCapper(columns=cappable_numerical_features)
    X_final_capped = outlier_capper.fit_transform(X_final)
    print("Outlier capping complete.")

    print("\nStep 5: Applying WOE Transformation to Categorical Features...")
    categorical_features_for_woe = X_final_capped.select_dtypes(include='object').columns.tolist()
    
    for col in ['CountryCode', 'PricingStrategy']:
        if col in X_final_capped.columns and col not in categorical_features_for_woe:
            categorical_features_for_woe.append(col)

    if XVERSE_WOE_AVAILABLE:
        woe_transformer_xverse = WOE(col_names=categorical_features_for_woe, df=X_final_capped, target_column=y_final.name)
        woe_transformer_xverse.fit()
        X_final_woe = woe_transformer_xverse.transform(X_final_capped)

        print("Information Value (IV) for xverse WoE transformed features:")
        for col, iv_df in woe_transformer_xverse.iv_values.items():
            if 'IV' in iv_df.columns:
                print(f"  {col}: {iv_df['IV'].iloc[0]:.4f}")
            else:
                 print(f"  {col}: IV calculation not directly available or in a different format from xverse output.")

    else:
        woe_transformer_custom = CustomWOETransformer(features=categorical_features_for_woe, target_column='is_high_risk')
        woe_transformer_custom.fit(X_final_capped, y_final)
        X_final_woe = woe_transformer_custom.transform(X_final_capped)
        
        print("Information Value (IV) for Custom WoE transformed features:")
        for col, iv in woe_transformer_custom.iv_values.items():
            print(f"  {col}: {iv:.4f}")

    print("WOE transformation complete.")

    print("\nStep 6: Scaling all Numerical Features (including WoE transformed features)...")
    X_final_woe_numeric = X_final_woe.select_dtypes(include=np.number)
    
    if not np.isfinite(X_final_woe_numeric).all().all():
        print("Warning: Non-finite values detected after WOE transformation. Attempting to clean before scaling.")
        X_final_woe_numeric = X_final_woe_numeric.replace([np.inf, -np.inf], np.nan)
        X_final_woe_numeric = X_final_woe_numeric.fillna(0)


    final_features_for_scaling = X_final_woe_numeric.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final_woe_numeric[final_features_for_scaling])
    
    X_processed_final = pd.DataFrame(X_scaled, columns=final_features_for_scaling, index=X_final_woe.index)
    
    X_processed_final['is_high_risk'] = y_final.values

    print("Scaling complete. Data is now model-ready with proxy target.")

    return X_processed_final

if __name__ == '__main__':
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
        'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # Original FraudResult is now removed and replaced by proxy
    }
    dummy_df = pd.DataFrame(data)

    print("--- Running data processing with dummy data (with proxy target) ---")
    processed_df = process_data(dummy_df.copy())
    if processed_df is not None:
        print("\n--- Final Processed Data (first 5 rows) ---")
        print(processed_df.head())
        print("\n--- Final Processed Data Info ---")
        processed_df.info()
        print("\n--- Final Processed Data Describe ---")
        print(processed_df.describe())
        print("\nUnique values in is_high_risk (new target):", processed_df['is_high_risk'].unique())
        print("Value counts for is_high_risk:\n", processed_df['is_high_risk'].value_counts())
