# linear_regression_model.py

import numpy as np

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

    # ====== Set Normalization Stats ======
    def set_normalization_stats(self, x_mean, x_std, y_mean, y_std):
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

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
