import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# from xverse import WOE


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
            # Convert to datetime, coercing errors will turn invalid dates into NaT
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
        # Use errors='ignore' to prevent errors if a column doesn't exist
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
            # If no columns specified, apply to all numerical columns
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
    Calculates aggregate numerical features and a customer-level fraud target.
    Also merges static categorical customer-related features.
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount', fraud_col='FraudResult'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.fraud_col = fraud_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Calculate aggregate numerical features for each customer
        customer_df = X.groupby(self.customer_id_col).agg(
            total_transaction_amount=(self.amount_col, 'sum'),
            average_transaction_amount=(self.amount_col, 'mean'),
            transaction_count=(self.amount_col, 'count'),
            std_transaction_amount=(self.amount_col, 'std'),
            # Define customer-level target: 1 if any transaction was fraudulent, 0 otherwise
            customer_fraudulent=(self.fraud_col, 'max')
        ).reset_index()

        # Handle potential NaNs in std_transaction_amount if a customer only has one transaction
        customer_df['std_transaction_amount'] = customer_df['std_transaction_amount'].fillna(0)

        # Merge relevant original customer-level features that are static per customer
        # We assume these features (like CountryCode, ProductCategory) are consistent for a customer.
        # Taking the first unique value for these categorical features.
        static_customer_features = [
            'CountryCode', 'CurrencyCode', 'ProviderId', 'ProductId',
            'ProductCategory', 'ChannelId', 'PricingStrategy',
            'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear' # Extracted temporal features
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
    def __init__(self, features=None, target_column='customer_fraudulent', numerical_bins=10, woe_cap=20):
        self.features = features
        self.target_column = target_column
        self.numerical_bins = numerical_bins
        self.woe_cap = woe_cap # Cap for WoE values to prevent infinity
        self.woe_maps = {}
        self.iv_values = {}
        self.feature_is_numeric = {}
        self.numerical_bin_edges = {}

    def fit(self, X, y=None):
        if self.features is None:
            # If features are not specified, use all object/category columns
            self.features = X.select_dtypes(include=['object', 'category', 'int64', 'float64']).columns.tolist()
            # Remove target column if it accidentally gets included
            if self.target_column in self.features:
                self.features.remove(self.target_column)

        if y is None:
            raise ValueError("Target variable 'y' must be provided for WOE transformer fitting.")

        # Align X and y indices
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        for col in self.features:
            if col not in X.columns:
                print(f"Warning: Column '{col}' not found in X for WOE calculation. Skipping.")
                continue

            temp_df = pd.DataFrame({col: X[col], 'target': y})
            temp_df['target'] = temp_df['target'].astype(int) # Ensure target is int (0 or 1)

            if pd.api.types.is_numeric_dtype(temp_df[col]):
                self.feature_is_numeric[col] = True
                # Bin numerical columns
                try:
                    # Drop NaNs before qcut, as qcut doesn't handle them
                    temp_df_no_nan = temp_df.dropna(subset=[col])
                    # Ensure bin labels match the unique bin numbers generated by qcut
                    temp_df_no_nan['binned_col'], self.numerical_bin_edges[col] = pd.qcut(
                        temp_df_no_nan[col], q=self.numerical_bins, labels=False, duplicates='drop', retbins=True
                    )
                    # Merge binned_col back to original temp_df
                    temp_df = temp_df.merge(
                        temp_df_no_nan[['binned_col']], left_index=True, right_index=True, how='left'
                    )
                except Exception as e:
                    print(f"Warning: pd.qcut failed for numerical column '{col}' ({e}). Falling back to pd.cut.")
                    # Fallback to cut if qcut fails (e.g., too few unique values, or all same value)
                    try:
                        # Ensure bins parameter is not too high for very few unique values
                        n_unique = temp_df[col].nunique()
                        effective_bins = min(self.numerical_bins, n_unique)
                        if effective_bins == 0: # Handle case where all values are NaN or only one unique value
                            temp_df['binned_col'] = np.nan # No meaningful binning
                        else:
                            temp_df['binned_col'], self.numerical_bin_edges[col] = pd.cut(
                                temp_df[col], bins=effective_bins, labels=False, include_lowest=True, duplicates='drop', retbins=True
                            )
                    except Exception as e_cut:
                        print(f"Error: pd.cut also failed for numerical column '{col}' ({e_cut}). Column will be skipped for WOE.")
                        self.feature_is_numeric[col] = False # Mark as not successfully binned/WOE'd
                        continue # Skip to next feature

                group_col = 'binned_col'
            else:
                self.feature_is_numeric[col] = False
                group_col = col

            # Calculate WoE and IV
            # Group by the binned/original category and count good (0) and bad (1) outcomes
            grouped = temp_df.groupby(group_col)['target'].agg(
                total_count='count',
                bad_count=lambda x: (x == 1).sum(),
                good_count=lambda x: (x == 0).sum()
            ).reset_index()

            # Calculate overall good/bad counts for denominator
            total_bad = grouped['bad_count'].sum()
            total_good = grouped['good_count'].sum()

            # Handle cases where good/bad counts are zero to avoid division by zero or log(0)
            epsilon = 1e-6 # Small constant to avoid division by zero or log(0)
            
            grouped['bad_rate'] = grouped['bad_count'] / (total_bad + epsilon)
            grouped['good_rate'] = grouped['good_count'] / (total_good + epsilon)

            # Calculate WoE
            grouped['woe'] = np.log((grouped['bad_rate'] + epsilon) / (grouped['good_rate'] + epsilon))
            
            # Cap WoE values to prevent -inf or +inf
            grouped['woe'] = np.clip(grouped['woe'], -self.woe_cap, self.woe_cap)

            # Calculate IV (Information Value)
            grouped['iv_contribution'] = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']
            self.iv_values[col] = grouped['iv_contribution'].sum()
            
            # Store the mapping from original category/bin to WoE value
            self.woe_maps[col] = grouped.set_index(group_col)['woe'].to_dict()

        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.features:
            if col not in X_copy.columns:
                continue # Column not in DataFrame, skip

            if col in self.woe_maps: # Check if the column was successfully fitted for WOE
                if self.feature_is_numeric.get(col, False):
                    # For numerical columns, apply the same binning as in fit before mapping
                    if col not in self.numerical_bin_edges:
                        # If bin edges not found (e.g., binning failed during fit), handle gracefully
                        X_copy[col] = 0 # Default to 0 or another suitable value
                        continue
                    
                    # Ensure the bin edges are correctly used for pd.cut
                    # pd.cut requires actual bin edges
                    binned_col_series = pd.cut(
                        X_copy[col], bins=self.numerical_bin_edges[col], labels=False,
                        include_lowest=True, right=True, duplicates='drop'
                    )
                    # Map the binned values to their WOE values
                    X_copy[col] = binned_col_series.map(self.woe_maps[col])
                else:
                    # For categorical columns, map using the stored woe_map
                    X_copy[col] = X_copy[col].map(self.woe_maps[col])
                
                X_copy[col] = X_copy[col].fillna(0) # Filling NaNs (unseen categories/bins) with 0 or a sensible default
            else:
                # If column was not fitted (e.g., due to errors during binning/WOE calculation),
                # convert it to a neutral numerical value (e.g., 0) if it's not already.
                if pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = X_copy[col].fillna(0) # Fill any original NaNs
                else:
                    X_copy[col] = 0 # Assign 0 if it's an object and wasn't transformed

        return X_copy


def process_data(df_raw: pd.DataFrame, target_column: str = 'FraudResult') -> pd.DataFrame:
    """
    Orchestrates the entire data processing pipeline from raw transaction data
    to a model-ready customer-level DataFrame.
    """
    if df_raw is None:
        print("Raw DataFrame is None. Cannot process data.")
        return None

    # Step 1: Initial Transaction-Level Preprocessing
    print("Step 1: Initial Transaction-Level Preprocessing (DateTime Extraction, Column Dropping)...")

    # Define columns to drop at the transaction level
    # 'Value' is dropped due to perfect multicollinearity with 'Amount'.
    # ID columns are dropped as they are high cardinality and not direct features after aggregation.
    transaction_level_cols_to_drop = [
        'Value', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId'
    ]

    # Pre-aggregation pipeline (transaction-level transformations)
    pre_agg_pipeline = Pipeline([
        ('datetime_extractor', DateTimeFeatureExtractor(date_column='TransactionStartTime')),
        ('column_dropper_initial', ColumnDropper(columns_to_drop=transaction_level_cols_to_drop))
    ])

    df_preprocessed = pre_agg_pipeline.fit_transform(df_raw.copy()) # Use a copy to avoid modifying original df_raw
    print("Initial preprocessing complete. Shape:", df_preprocessed.shape)


    # Step 2: Aggregate to Customer Level
    print("\nStep 2: Aggregating data to Customer-Level and creating customer-level target...")
    # CustomerAggregator handles the aggregation and merges static customer features
    customer_aggregator = CustomerAggregator(fraud_col=target_column)
    df_customer_level = customer_aggregator.fit_transform(df_preprocessed)
    print("Customer-level aggregation complete. Shape:", df_customer_level.shape)

    # Separate features (X) and target (y)
    y_customer = df_customer_level['customer_fraudulent']
    # 'CustomerId' is dropped from X as it's an identifier, not a feature for the model
    X_customer = df_customer_level.drop(columns=['customer_fraudulent', 'CustomerId'])


    # Step 3: Outlier Capping on Numerical Features
    print("\nStep 3: Handling Outliers on Numerical Features...")
    # Identify numerical features after aggregation (and temporal feature extraction)
    numerical_features_after_agg = X_customer.select_dtypes(include=np.number).columns.tolist()
    
    # Exclude features like 'TransactionMonth', 'TransactionDay', 'TransactionHour', 'TransactionYear' from capping
    # as they represent cyclical time points and outliers might not be appropriate to cap.
    cappable_numerical_features = [
        col for col in numerical_features_after_agg
        if col not in ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    ]

    outlier_capper = OutlierCapper(columns=cappable_numerical_features)
    X_customer_capped = outlier_capper.fit_transform(X_customer)
    print("Outlier capping complete.")


    # Step 4: WOE Transformation for Categorical Features (and binned numerical if desired)
    print("\nStep 4: Applying WOE Transformation to Categorical Features...")
    # Identify categorical features for WOE, including those initially numeric but treated as categorical
    categorical_features_for_woe = X_customer_capped.select_dtypes(include='object').columns.tolist()
    
    # Add numerical features like CountryCode, PricingStrategy for WOE transformation
    for col in ['CountryCode', 'PricingStrategy']:
        if col in X_customer_capped.columns and col not in categorical_features_for_woe:
            categorical_features_for_woe.append(col)

    if XVERSE_WOE_AVAILABLE:
        # xverse WOE expects a DataFrame and target. It transforms the original columns in place.
        woe_transformer_xverse = WOE(col_names=categorical_features_for_woe, df=X_customer_capped, target_column=y_customer.name)
        woe_transformer_xverse.fit() # Fit will populate the object with maps
        X_customer_woe = woe_transformer_xverse.transform(X_customer_capped) # Transform will return the new DataFrame with WoE values

        # Get IV values from xverse (usually an attribute after fit)
        print("Information Value (IV) for xverse WoE transformed features:")
        # xverse stores iv_values as a dict of DataFrames, check its structure
        for col, iv_df in woe_transformer_xverse.iv_values.items():
            if 'IV' in iv_df.columns:
                print(f"  {col}: {iv_df['IV'].iloc[0]:.4f}") # Assuming IV is in the first row of 'IV' column
            else:
                 print(f"  {col}: IV calculation not directly available or in a different format from xverse output.")

    else:
        # Use CustomWOETransformer if xverse is not available
        woe_transformer_custom = CustomWOETransformer(features=categorical_features_for_woe, target_column='customer_fraudulent')
        woe_transformer_custom.fit(X_customer_capped, y_customer)
        X_customer_woe = woe_transformer_custom.transform(X_customer_capped)
        
        print("Information Value (IV) for Custom WoE transformed features:")
        for col, iv in woe_transformer_custom.iv_values.items():
            print(f"  {col}: {iv:.4f}")

    print("WOE transformation complete.")


    # Step 5: Final Scaling of all Numerical Features
    print("\nStep 5: Scaling all Numerical Features (including WoE transformed features)...")
    # All features should now be numerical (original numerical + WoE transformed)
    # Ensure no non-finite values are present before scaling
    X_customer_woe_numeric = X_customer_woe.select_dtypes(include=np.number)
    
    # Check for infinities or NaNs one last time before scaling
    if not np.isfinite(X_customer_woe_numeric).all().all():
        print("Warning: Non-finite values detected after WOE transformation. Attempting to clean before scaling.")
        # Replace inf with large finite number, NaN with mean/median or 0
        X_customer_woe_numeric = X_customer_woe_numeric.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN first
        X_customer_woe_numeric = X_customer_woe_numeric.fillna(0) # Fill NaNs with 0 (or mean/median if appropriate)


    final_features_for_scaling = X_customer_woe_numeric.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_customer_woe_numeric[final_features_for_scaling])
    
    # Create a DataFrame from the scaled data, preserving column names and index
    X_processed = pd.DataFrame(X_scaled, columns=final_features_for_scaling, index=X_customer_woe.index)
    
    # Re-attach the target variable
    X_processed['customer_fraudulent'] = y_customer.values # Ensure alignment of index/values

    print("Scaling complete. Data is now model-ready.")

    return X_processed

# Example usage (for testing within the script or in a notebook)
if __name__ == '__main__':
    # Create a dummy DataFrame for demonstration
    # This simulates your raw data structure based on df.info() and df.head()
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'],
        'BatchId': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'],
        'AccountId': ['A1', 'A1', 'A2', 'A2', 'A3', 'A3', 'A4', 'A4', 'A5', 'A5'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S5', 'S5'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C5'],
        'CurrencyCode': ['USD', 'USD', 'EUR', 'USD', 'EUR', 'GBP', 'USD', 'EUR', 'USD', 'GBP'],
        'CountryCode': [1, 1, 2, 1, 2, 3, 1, 2, 1, 3], # Numerical but should be treated categorical
        'ProviderId': ['P1', 'P1', 'P2', 'P1', 'P2', 'P3', 'P1', 'P2', 'P1', 'P3'],
        'ProductId': ['ProdA', 'ProdA', 'ProdB', 'ProdA', 'ProdB', 'ProdC', 'ProdA', 'ProdB', 'ProdA', 'ProdC'],
        'ProductCategory': ['CatX', 'CatX', 'CatY', 'CatX', 'CatY', 'CatZ', 'CatX', 'CatY', 'CatX', 'CatZ'],
        'ChannelId': ['Ch1', 'Ch1', 'Ch2', 'Ch1', 'Ch2', 'Ch3', 'Ch1', 'Ch2', 'Ch1', 'Ch3'],
        'Amount': [100.5, 200.0, 50.0, 15000.0, 75.0, 300.0, 120.0, 60.0, 180.0, 350.0], # High outlier for C2 (15000)
        'Value': [100.5, 200.0, 50.0, 15000.0, 75.0, 300.0, 120.0, 60.0, 180.0, 350.0], # Identical to Amount
        'TransactionStartTime': [
            '2023-01-01 10:00:00', '2023-01-01 11:30:00', '2023-01-02 14:00:00', # C1, C2
            '2023-01-03 09:00:00', '2023-01-03 16:00:00', '2023-01-04 10:00:00', # C2, C3, C3
            '2023-01-05 12:00:00', '2023-01-05 13:00:00', '2023-01-06 17:00:00', '2023-01-06 18:00:00' # C4, C4, C5, C5
        ],
        'PricingStrategy': [1, 1, 2, 1, 2, 3, 1, 2, 1, 3], # Numerical but should be treated categorical
        'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] # C2 has a fraudulent transaction (Amount=15000)
    }
    dummy_df = pd.DataFrame(data)

    print("--- Running data processing with dummy data ---")
    processed_df = process_data(dummy_df.copy(), target_column='FraudResult') # Pass the original target column name
    if processed_df is not None:
        print("\n--- Processed Data (first 5 rows) ---")
        print(processed_df.head())
        print("\n--- Processed Data Info ---")
        processed_df.info()
        print("\n--- Processed Data Describe ---")
        print(processed_df.describe())
        print("\nUnique values in customer_fraudulent (target):", processed_df['customer_fraudulent'].unique())

