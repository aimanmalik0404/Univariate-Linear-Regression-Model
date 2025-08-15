# main.py

from csv_loader import CSVLoader
from data_processor import DataProcessor
from linear_regression_model import LinearRegressionModel
import tkinter as tk
from tkinter import filedialog

# ====== Browse CSV file ======
def browse_csv():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv")]
    )
    if file_path:
        print(f"✅ Selected file: {file_path}")
        return file_path
    else:
        print("❌ No file selected.")
        exit()

file_path = browse_csv()

# ====== Load CSV and select columns ======
loader = CSVLoader()
df = loader.load_csv(file_path)
df = loader.select_columns(df)

# ====== Clean and normalize data ======
processor = DataProcessor(loader.X_col, loader.Y_col)
df = processor.clean_data(df)
df = processor.normalize_data(df)

# ====== Ask user for learning rate and max epochs ======
try:
    alpha = float(input("Enter learning rate (alpha), e.g., 0.1: "))
except ValueError:
    print("Invalid input. Using default alpha = 0.1")
    alpha = 0.1

try:
    max_epochs = int(input("Enter maximum number of epochs, e.g., 80: "))
except ValueError:
    print("Invalid input. Using default max_epochs = 80")
    max_epochs = 80

# ====== Train and test model ======
model = LinearRegressionModel(alpha=alpha, max_epochs=max_epochs, tolerance=0)
model.set_normalization_stats(processor.x_mean, processor.x_std, processor.y_mean, processor.y_std)

X_train, y_train, X_test, y_test = model.split_data(df)
model.train(X_train, y_train)
model.test(X_test, y_test)

# ====== Predict a single value ======
try:
    raw_x = float(input("Enter the value of X (original/raw value): "))
    norm_y, raw_y = model.predict(raw_x)
    print(f"Predicted Y (normalized): {norm_y}")
    print(f"Predicted Y (original scale): {raw_y}")
except ValueError:
    print("❌ Invalid input for X value.")
