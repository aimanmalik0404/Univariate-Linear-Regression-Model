# data_processor.py

import pandas as pd

class DataProcessor:
    def __init__(self, X_col, Y_col):
        self.X_col = X_col
        self.Y_col = Y_col
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    # ====== Clean Data ======
    def clean_data(self, df):
        try:
            df = df.drop_duplicates()
            df[self.X_col] = pd.to_numeric(df[self.X_col], errors='coerce')
            df[self.Y_col] = pd.to_numeric(df[self.Y_col], errors='coerce')
            df = df.dropna()
            print("? Data cleaned successfully!")
            print("\nCleaned CSV:")
            print(df)
            return df
        except Exception as e:
            print(f"? Error cleaning data: {e}")
            exit()

    # ====== Normalize Data ======
    def normalize_data(self, df):
        try:
            if df.empty:
                print("? Cannot normalize: DataFrame is empty after cleaning.")
                exit()

            self.x_mean = df[self.X_col].mean()
            self.x_std = df[self.X_col].std()
            self.y_mean = df[self.Y_col].mean()
            self.y_std = df[self.Y_col].std()

            df = df.copy()
            df['X_norm'] = (df[self.X_col] - self.x_mean) / self.x_std
            df['Y_norm'] = (df[self.Y_col] - self.y_mean) / self.y_std

            print("? Z-score normalization done!")
            return df
        except Exception as e:
            print(f"? Error in normalization: {e}")
            exit()
