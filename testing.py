# linear_regression_class_select_columns.py

import numpy as np
import pandas as pd

class LinearRegressionModel:
    def __init__(self, alpha=0.1, max_epochs=80, tolerance=0):
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.theta_0 = 0
        self.theta_1 = 0
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.X_col = None
        self.Y_col = None

    # ====== Load CSV ======
    def load_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            print("✅ CSV loaded successfully!")
            return df
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            exit()

    # ====== Select Columns ======
    def select_columns(self, df):
        print("\nAvailable columns in CSV:")
        for idx, col in enumerate(df.columns):
            print(f"{idx+1}. {col}")
        try:
            x_choice = int(input("Enter the number of the column to use as X: ")) - 1
            y_choice = int(input("Enter the number of the column to use as Y: ")) - 1
            self.X_col = df.columns[x_choice]
            self.Y_col = df.columns[y_choice]
            print(f"✅ Selected X: {self.X_col}, Y: {self.Y_col}")
        except Exception as e:
            print(f"❌ Invalid selection: {e}")
            exit()
        return df[[self.X_col, self.Y_col]]

    # ====== Clean Data ======
    def clean_data(self, df):
        try:
            df = df.drop_duplicates()
            # Only convert selected X and Y columns to numeric
            df[self.X_col] = pd.to_numeric(df[self.X_col], errors='coerce')
            df[self.Y_col] = pd.to_numeric(df[self.Y_col], errors='coerce')
            df = df.dropna()
            print("✅ Data cleaned successfully!")
            print("\nCleaned CSV:")
            print(df)
            return df
        except Exception as e:
            print(f"❌ Error cleaning data: {e}")
            exit()

    # ====== Normalize Data ======
    def normalize_data(self, df):
        try:
            self.x_mean = df[self.X_col].mean()
            self.x_std = df[self.X_col].std()
            self.y_mean = df[self.Y_col].mean()
            self.y_std = df[self.Y_col].std()

            df = df.copy()
            df['X_norm'] = (df[self.X_col] - self.x_mean) / self.x_std
            df['Y_norm'] = (df[self.Y_col] - self.y_mean) / self.y_std

            print("✅ Z-score normalization done!")
            return df
        except Exception as e:
            print(f"❌ Error in normalization: {e}")
            exit()

    # ====== Split Data ======
    def split_data(self, df, train_frac=0.8):
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(train_frac * len(df))
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        X_train = train_data['X_norm'].values
        y_train = train_data['Y_norm'].values
        X_test = test_data['X_norm'].values
        y_test = test_data['Y_norm'].values
        return X_train, y_train, X_test, y_test

    # ====== Train Model ======
    def train(self, X_train, y_train):
        m = len(X_train)
        prev_cost = float('inf')
        epochs = 0

        while epochs < self.max_epochs:
            y_pred = self.theta_0 + self.theta_1 * X_train
            error = y_pred - y_train
            cost = (1 / (2 * m)) * np.sum(error ** 2)

            if abs(prev_cost - cost) < self.tolerance:
                break
            prev_cost = cost

            d_theta0 = (1/m) * np.sum(error)
            d_theta1 = (1/m) * np.sum(error * X_train)

            self.theta_0 -= self.alpha * d_theta0
            self.theta_1 -= self.alpha * d_theta1
            epochs += 1

        print(f"✅ Training complete in {epochs} epochs")
        print(f"θ0 (intercept): {self.theta_0}")
        print(f"θ1 (slope): {self.theta_1}")

    # ====== Test Model ======
    def test(self, X_test, y_test):
        y_pred = self.theta_0 + self.theta_1 * X_test
        mse = (1 / len(X_test)) * np.sum((y_pred - y_test) ** 2)
        print(f"✅ Test MSE (normalized space): {mse}")

    # ====== Predict Single Value ======
    def predict(self, raw_x):
        norm_x = (raw_x - self.x_mean) / self.x_std
        norm_y = self.theta_0 + self.theta_1 * norm_x
        raw_y = (norm_y * self.y_std) + self.y_mean
        return norm_y, raw_y


if __name__ == "__main__":
    file_path = input("Enter the path to your CSV file: ")

    model = LinearRegressionModel(alpha=0.1, max_epochs=80, tolerance=0)

    df = model.load_csv(file_path)
    df = model.select_columns(df)
    df = model.clean_data(df)
    df = model.normalize_data(df)
    X_train, y_train, X_test, y_test = model.split_data(df)

    model.train(X_train, y_train)
    model.test(X_test, y_test)

    raw_x = float(input("Enter the value of X (original/raw value): "))
    norm_y, raw_y = model.predict(raw_x)
    print(f"Predicted Y (normalized): {norm_y}")
    print(f"Predicted Y (original scale): {raw_y}")
