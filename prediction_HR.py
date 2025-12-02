import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("Employee Attrition Prediction (HRDataset_v14)")
st.write("Predict whether an employee is likely to leave the company.")

# Load dataset
df = pd.read_csv("HRDataset_v14.csv")

# Filter only 'Active' and 'Terminated' employees
df = df[df["EmpStatusID"].isin([1, 3])]  # 1 = Terminated, 3 = Active
df = df.dropna(subset=["EmpSatisfaction", "Absences", "SpecialProjectsCount"])

# Map attrition: 1 = left, 0 = stayed
df["Attrition"] = df["EmpStatusID"].map({3: 0, 1: 1})

# Feature selection
features = ["EmpSatisfaction", "Absences", "SpecialProjectsCount"]
X = df[features]
y = df["Attrition"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with class balance
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
st.info(f"📈 F1 Score: {f1:.2f}")

# Individual Prediction Form
st.header("👤 Predict Attrition for One Employee")

satisfaction = st.slider("Employee Satisfaction (1–5)", 1, 5, 3)
absences = st.number_input("Number of Absences", min_value=0, max_value=60, value=5)
projects = st.slider("Special Projects Count", 0, 10, 1)

if st.button("Predict Individual"):
    input_data = [[satisfaction, absences, projects]]
    prediction = model.predict(input_data)[0]
    result = "⚠️ Yes (Attrition)" if prediction == 1 else "✅ No (Stays)"
    st.subheader(f"Prediction: {result}")

# Company-Wide Prediction
st.header("🏢 Company-Wide Attrition Prediction")
if st.button("Predict for All Employees"):
    full_predictions = model.predict(X)
    df["PredictedAttrition"] = full_predictions

    stays = (df["PredictedAttrition"] == 0).sum()
    leaves = (df["PredictedAttrition"] == 1).sum()
    total = len(df)
    attrition_rate = (leaves / total) * 100

    st.success(f"✅ Employees Predicted to Stay: {stays}")
    st.warning(f"⚠️ Employees Predicted to Leave: {leaves}")
    st.info(f"📊 Overall Attrition Rate: {attrition_rate:.2f}%")

    #  Pie Chart - Attrition Distribution 
    st.subheader("Overall Attrition Distribution")
    fig, ax = plt.subplots()
    colors = ["#4db6ac","#ff8a65"]  # Teal & Coral 
    ax.pie([stays, leaves], 
       labels=["Stays", "Leaves"], 
       autopct='%1.1f%%', 
       startangle=90, 
       colors=colors,
       textprops={'fontsize': 12})
    ax.axis("equal")
    st.pyplot(fig)

st.header("🏢 Department-Wise Attrition Prediction")

if "Department" in df.columns:
    # Predict attrition for all employees
    department_df = df.copy()
    department_df["PredictedAttrition"] = model.predict(department_df[features])

    # Group by Department and Prediction
    dept_attrition_counts = department_df.groupby(["Department", "PredictedAttrition"]).size().unstack(fill_value=0)

    # Rename columns for clarity
    dept_attrition_counts.columns = ["Stays", "Leaves"]

    # Display in Streamlit
    st.dataframe(dept_attrition_counts)

    # Bar Chart - Department-wise Attrition 
    st.subheader("Department-wise Attrition Overview")
    fig, ax = plt.subplots(figsize=(10, 5))
    dept_attrition_counts.plot(kind='bar', 
                           stacked=True, 
                           ax=ax, 
                           color=["#81d4fa", "#ffab91"],  # Light Blue & Peach
                           edgecolor='black')
    ax.set_title("Department-wise Attrition", fontsize=14)
    ax.set_xlabel("Department", fontsize=12)
    ax.set_ylabel("Number of Employees", fontsize=12)
    ax.legend(["Stays", "Leaves"], title="Attrition Status")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

else:
    st.warning("⚠️ 'Department' column not found in the dataset.")
