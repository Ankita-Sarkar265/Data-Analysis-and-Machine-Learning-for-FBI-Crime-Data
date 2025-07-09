import pandas as pd

# Replace this with your actual file path if needed
df = pd.read_csv("data/crime_data.csv")

# Show first few rows to check column names
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# Replace 'location_type' with the actual column name you see above!
print("\nUnique values in suspected location column:")
print(df['location_type'].value_counts())
