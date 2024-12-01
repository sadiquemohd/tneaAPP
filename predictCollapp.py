import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from st_aggrid import AgGrid

# Load the models and imputer
with open('xgboost_college_model.pkl', 'rb') as file:
    college_model = pickle.load(file)

with open('xgboost_branch_model.pkl', 'rb') as file:
    branch_model = pickle.load(file)

with open('imputer.pkl', 'rb') as file:
    imputer = pickle.load(file)

# Load dataset for college and branch details
df = pd.read_excel("allot_college_branch_ds.xlsx")

# Encode college, branch, and community codes
college_encoder = LabelEncoder()
branch_encoder = LabelEncoder()
community_encoder = LabelEncoder()

community_encoder.fit(df['COMMUNITY'])
college_encoder.fit(df['COLLEGE_CODE'])
branch_encoder.fit(df['BRANCH_CODE'])

# Sidebar Inputs
st.sidebar.header("Input Details")

community = st.sidebar.selectbox("Select Community", df['COMMUNITY'].unique())
aggregate_mark = st.sidebar.number_input("Enter Aggregate Marks", min_value=0, max_value=200, value=150)
rank = st.sidebar.number_input("Enter Rank (Optional)", min_value=1, max_value=100000, value=5000)
st.title("TNEA Collge Predictor")
# Prediction Button
if st.sidebar.button("Predict"):

    # Encode community
    community_encoded = community_encoder.transform([community])[0]

    # Create input array and handle missing features
    input_data = np.array([[community_encoded, aggregate_mark, rank]], dtype=np.float64)

    # Impute missing values
    input_data = imputer.transform(input_data)

    # Get probability predictions for top colleges and branches
    college_probabilities = college_model.predict_proba(input_data)
    branch_probabilities = branch_model.predict_proba(input_data)

    # Get top 10 colleges and branches based on probabilities
    top_10_college_indices = np.argsort(college_probabilities[0])[::-1][:10]
    top_10_branch_indices = np.argsort(branch_probabilities[0])[::-1][:10]

    # Decode college and branch codes
    top_10_colleges = college_encoder.inverse_transform(top_10_college_indices)
    top_10_branches = branch_encoder.inverse_transform(top_10_branch_indices)

    # Prepare data for display
    results = []
    for college_code, branch_code in zip(top_10_colleges, top_10_branches):
        college_info = df[df['COLLEGE_CODE'] == college_code].iloc[0]
        branch_info = df[df['BRANCH_CODE'] == branch_code].iloc[0]
        
        results.append({
            "College Code": college_code,
            "College Name": college_info['Name of the College'],
            "District": college_info['District'],
            "Zone": college_info['Zone'],
            "Branch Code": branch_code,
            "Branch Name": branch_info['Branch Name'],
            "College Probability": college_probabilities[0][college_encoder.transform([college_code])[0]],
            "Branch Probability": branch_probabilities[0][branch_encoder.transform([branch_code])[0]]
        })

    # Display results in an AgGrid
    st.header("Top 10 Predicted Colleges and Branches")
    AgGrid(pd.DataFrame(results), height=400, fit_columns_on_grid_load=True)
