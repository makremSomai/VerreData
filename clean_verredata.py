import pandas as pd

# Load the dataset with correct encoding
file_path = "verredata.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Remove empty rows
df = df.dropna().reset_index(drop=True)

# Convert comma (`,`) to dot (`.`) for numerical conversion
df["Conductivité thermique"] = df["Conductivité thermique"].str.replace(",", ".").astype(float)
df["Résistance Mécanique"] = df["Résistance Mécanique"].str.replace(",", ".").astype(float)

# Save cleaned dataset
df.to_csv("cleaned_verredata.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_verredata.csv'")
