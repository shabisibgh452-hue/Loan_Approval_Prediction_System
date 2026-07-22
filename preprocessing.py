import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("loan_data_new.csv")

# ==========================
# Remove Duplicates
# ==========================
df.drop_duplicates(inplace=True)

# ==========================
# Convert Numeric Columns Safely
# ==========================
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except (ValueError, TypeError):
        pass

# ==========================
# Handle Missing Values
# ==========================
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

# ==========================
# Encode Categorical Columns
# ==========================
le = LabelEncoder()

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# ==========================
# Save Clean Dataset
# ==========================
df.to_csv("loan_cleaned.csv", index=False)

print("=" * 50)
print("Preprocessing completed successfully!")
print("=" * 50)
print(df.head())
print(df.info())