import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="IBM Attrition Predictor", layout="centered")
st.title("Employee Attrition Prediction (IBM Dataset)")
st.write("Predict whether an employee is likely to leave the company.")

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.dropna(inplace=True)

# Convert target column to numeric
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Features including satisfaction levels
features = [
    'Age', 
    'DistanceFromHome', 
    'MonthlyIncome', 
    'TotalWorkingYears', 
    'YearsAtCompany',
    'JobSatisfaction',
    'EnvironmentSatisfaction'
]
X = df[features]
y = df["Attrition"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Balanced Logistic Regression model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
st.info(f"📈 F1 Score: {f1:.2f}")

# Individual Prediction UI
st.header("👤 Predict Attrition for One Employee")

age = st.number_input("Age", 18, 60, 30)
distance = st.number_input("Distance From Home", 0, 50, 5)
income = st.number_input("Monthly Income", 1000, 20000, 5000)
total_years = st.number_input("Total Working Years", 0, 40, 5)
years_company = st.number_input("Years At Company", 0, 40, 3)
job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
env_satisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)

if st.button("Predict Individual"):
    input_data = [[age, distance, income, total_years, years_company, job_satisfaction, env_satisfaction]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Prob of attrition
    result = "⚠️ Yes (Attrition)" if prediction == 1 else "✅ No (Stays)"
    st.subheader(f"Prediction: {result}")
    st.caption(f"Probability of Attrition: {probability * 100:.2f}%")

# Company-Wide Prediction
st.header("🏢 Company-Wide Attrition Prediction")
if st.button("Predict for All Employees"):
    df["PredictedAttrition"] = model.predict(X_scaled)

    stays = (df["PredictedAttrition"] == 0).sum()
    leaves = (df["PredictedAttrition"] == 1).sum()
    total = len(df)
    attrition_rate = (leaves / total) * 100

    st.success(f"✅ Employees Predicted to Stay: {stays}")
    st.warning(f"⚠️ Employees Predicted to Leave: {leaves}")
    st.info(f"📊 Overall Attrition Rate: {attrition_rate:.2f}%")

    # Pie Chart - Attrition Distribution 
    st.subheader("Overall Attrition Distribution")
    fig, ax = plt.subplots()
    colors = ["#4db6ac", "#ff8a65"]
    ax.pie([stays, leaves], 
       labels=["Stays", "Leaves"], 
       autopct='%1.1f%%', 
       startangle=90, 
       colors=colors,
       textprops={'fontsize': 12})
    ax.axis("equal")
    st.pyplot(fig)
    
# Department-Wise Prediction
st.header("🏢 Department-Wise Attrition Prediction")

if "Department" in df.columns:
    department_df = df.copy()
    department_df["PredictedAttrition"] = model.predict(X_scaled)

    # Group by Department and Prediction
    dept_attrition_counts = department_df.groupby(["Department", "PredictedAttrition"]).size().unstack(fill_value=0)

    # Rename columns for clarity
    dept_attrition_counts.columns = ["Stay", "Leave"]

    # Display as table
    st.dataframe(dept_attrition_counts)

    # Bar Chart - Department-wise Attrition
    st.subheader("Department-wise Attrition Overview")
    fig, ax = plt.subplots(figsize=(10, 5))
    dept_attrition_counts.plot(kind='bar',
                               stacked=True,
                               ax=ax,
                               color=["#81d4fa", "#ffab91"],  # Cool tones
                               edgecolor='black')
    ax.set_title("Department-wise Attrition", fontsize=14)
    ax.set_xlabel("Department", fontsize=12)
    ax.set_ylabel("Number of Employees", fontsize=12)
    ax.legend(["Stays", "Leaves"], title="Attrition Status")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

else:
    st.warning("⚠️ 'Department' column not found in the dataset.")
