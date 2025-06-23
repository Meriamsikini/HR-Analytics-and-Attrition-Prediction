import pandas as pd
import os

# Obtenir le chemin absolu du dossier contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin vers le fichier CSV
csv_path = os.path.join(script_dir, "../data/HR-Employees-cleaning.csv")

# Charger le fichier
df = pd.read_csv(csv_path)

# --- Feature Engineering ---

# 1. Tenure in days and months (from HireDate)
if 'HireDate' in df.columns:
    df['HireDate'] = pd.to_datetime(df['HireDate'], errors='coerce')
    df['TenureDays'] = (pd.Timestamp('today') - df['HireDate']).dt.days
    df['TenureMonths'] = df['TenureDays'] // 30
    # Remove negative or unrealistic tenure
    df['TenureDays'] = df['TenureDays'].apply(lambda x: x if x >= 0 else 0)
    df['TenureMonths'] = df['TenureMonths'].apply(lambda x: x if x >= 0 else 0)

# 2. Time since last promotion (from LastPromotionDate)
if 'LastPromotionDate' in df.columns:
    df['LastPromotionDate'] = pd.to_datetime(df['LastPromotionDate'], errors='coerce')
    df['MonthsSinceLastPromotion'] = ((pd.Timestamp('today') - df['LastPromotionDate']).dt.days // 30)
    # Remove negative values
    df['MonthsSinceLastPromotion'] = df['MonthsSinceLastPromotion'].apply(lambda x: x if x >= 0 else 0)

# 3. Age group (categorical)
df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 29, 39, 49, 59, 70], labels=['18-29', '30-39', '40-49', '50-59', '60+'])

# --- Remove raw date columns if tenure features exist ---
for col in ['HireDate', 'LastPromotionDate']:
    if col in df.columns:
        df = df.drop(columns=[col])

# --- Encoding ---

# Encode categorical variables except target
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [col for col in cat_cols if col != 'Attrition']

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# --- Double-check for missing values ---
df = df.fillna(0)  


# Save the encoded and feature-engineered data
df.to_csv("encode_featured.csv", index=False)
print("Encoded and feature-engineered data saved and ready for ML")