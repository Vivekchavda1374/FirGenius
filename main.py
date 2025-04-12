# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("./Dataset/gym recommendation (1).csv")
print("Original column names:", df.columns.tolist())
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("Standardized column names:", df.columns.tolist())

# Drop rows with missing essential values
df = df.dropna(subset=['age', 'weight', 'height'])

# Calculate BMI if missing
if 'bmi' not in df.columns or df['bmi'].isnull().sum() > 0:
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)

# Fill missing values
df.fillna({
    'hypertension': 0,
    'diabetes': 0,
    'sex': 'other',
    'fitness_goal': 'general fitness'  # Note the updated column name with underscore
}, inplace=True)

categorical_columns = ['sex', 'fitness_goal', 'fitness_type']  # Updated to match standardized names

# Verify these columns exist in the dataframe
for col in categorical_columns:
    if col not in df.columns:
        print(f"not found in dataframe.")

# Define a function to safely encode columns
def safe_encode(dataframe, column_name):
    if column_name in dataframe.columns:
        le = LabelEncoder()
        dataframe[column_name] = le.fit_transform(dataframe[column_name].astype(str))
        return le
    else:
        print(f"Column '{column_name}' not found")
        return None

# Encode categorical data
label_encoders = {}
for col in categorical_columns:
    encoder = safe_encode(df, col)
    if encoder:
        label_encoders[col] = encoder

# Print a sample of the processed dataframe
print("\nProcessed dataframe sample:")
print(df.head())

# Print summary stats to verify data quality
print("\nDataframe information:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())
