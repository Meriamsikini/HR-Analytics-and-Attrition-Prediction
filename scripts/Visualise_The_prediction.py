import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import os

# Obtenir le chemin absolu du dossier contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin vers le fichier CSV
csv_path = os.path.join(script_dir, "../models/encode_featured.csv")

# Charger le fichier
df = pd.read_csv(csv_path)



#df = pd.read_csv("encode_featured.csv")

X = df.drop(['Attrition'], axis=1)
y = df['Attrition']

# train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#  XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


results = pd.DataFrame({
    'EmployeeNumber': X_test['EmployeeNumber'] if 'EmployeeNumber' in X_test.columns else X_test.index,
    'Attrition_True': y_test,
    'RF_Prediction': y_pred_rf,
    'XGB_Prediction': y_pred_xgb
})


results['Will Stay?'] = results[['RF_Prediction', 'XGB_Prediction']].mode(axis=1)[0].apply(lambda x: "No" if x == 1 else "Yes")


st.title("Employee Attrition Prediction Dashboard")


st.header("Current Employees (Attrition = 0)")
st.dataframe(df[df["Attrition"] == 0])


st.header("Prediction Results ")
st.dataframe(results[['EmployeeNumber', 'Will Stay?', 'RF_Prediction', 'XGB_Prediction']])


st.header("Prediction Statistics")
stats = results['Will Stay?'].value_counts().reset_index()
stats.columns = ['Will Stay?', 'Count']
st.dataframe(stats)

# Diagramme : Répartition des prédictions
st.subheader("Bar Chart: Prediction Distribution")
fig_bar, ax_bar = plt.subplots()
sns.countplot(data=results, x='Will Stay?', palette="Set2", ax=ax_bar)
st.pyplot(fig_bar)

#Pie Chart
st.subheader("Pie Chart")
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(stats['Count'], labels=stats['Will Stay?'], autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'])
ax_pie.axis('equal')
st.pyplot(fig_pie)

# Confusion Matrix
st.subheader("Confusion Matrix: Random Forest")
cm = confusion_matrix(y_test, y_pred_rf)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig_cm)

# Classification Report
st.subheader("Classification Report - Random Forest")
st.text(classification_report(y_test, y_pred_rf))

st.subheader("Classification Report - XGBoost")
st.text(classification_report(y_test, y_pred_xgb))


#to run the app, use the command:streamlit run "HR-Analytics-and-Attrition-Prediction-main/scripts/Visualise_The_prediction.py"