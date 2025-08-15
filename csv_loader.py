# csv_loader.py

import pandas as pd

class CSVLoader:
    def __init__(self):
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
