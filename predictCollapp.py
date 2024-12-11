import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import time
import streamlit as st
import os

# File to store the hit count
counter_file = "counter.txt"

# Initialize counter if file doesn't exist
if not os.path.exists(counter_file):
    with open(counter_file, "w") as f:
        f.write("0")

# Read the current count
with open(counter_file, "r") as f:
    count = int(f.read())

# Increment the count
count += 1

# Save the updated count
with open(counter_file, "w") as f:
    f.write(str(count))

# Display the count on the sidebar
st.sidebar.write(f"Page Hits: {count}")

# Cache models
@st.cache_resource
def load_all_models():
    with open('xgboost_college_model.pkl', 'rb') as f1, \
         open('xgboost_branch_model.pkl', 'rb') as f2, \
         open('imputer.pkl', 'rb') as f3:
        college_model = joblib.load(f1)
        branch_model = joblib.load(f2)
        imputer = joblib.load(f3)
    return college_model, branch_model, imputer

# Initialize models
college_model, branch_model, imputer = load_all_models()

# Initialize session state for prediction control
if "predict_triggered" not in st.session_state:
    st.session_state.predict_triggered = False
st.markdown(
    """
    <h1 style='text-align: center; font-size: 18px;'>
        TNEA College Predictor
    </h1>
    <hr>
    """,
    unsafe_allow_html=True
)
# Sidebar Inputs
st.sidebar.header("Input Details")
community = st.sidebar.selectbox("Select Community", ['OC','BC','BCM','MBC','SC','SCA','ST'])
aggregate_mark = st.sidebar.text_input("Enter Aggregate Marks",)  # Text input for marks
rank = st.sidebar.text_input("Enter Rank (Optional)",0)  # Text input for rank

# Convert inputs to numeric and validate
try:
    aggregate_mark = float(aggregate_mark)
    rank=int(rank)
    valid_inputs = True
except ValueError:
    valid_inputs = False
    st.sidebar.error("Please enter valid numeric values for Aggregate Marks and Rank(optional).")

# Prediction Button
if st.sidebar.button("Predict") and valid_inputs:
    st.session_state.predict_triggered = True

# Only run predictions if the button was clicked
if st.session_state.predict_triggered:
    # Progress bar
    with st.spinner("Predicting..."):
        progress = st.progress(0)

        # Simulate a time-consuming task with progress updates
        for i in range(100):
            time.sleep(0.03)  # Simulate processing time
            progress.progress(i + 1)

        # Encode inputs
        # Load dataset for college and branch details
        df = pd.read_excel("allot_college_branch_ds.xlsx")

        # Encode college, branch, and community codes
        college_encoder = LabelEncoder()
        branch_encoder = LabelEncoder()
        community_encoder = LabelEncoder()

        community_encoder.fit(df['COMMUNITY'])
        college_encoder.fit(df['COLLEGE_CODE'])
        branch_encoder.fit(df['BRANCH_CODE'])
        community_encoded = community_encoder.transform([community])[0]

        # Prepare input array
        input_data = np.array([[community_encoded, aggregate_mark, rank]], dtype=np.float64)

        # Impute missing values
        input_data = imputer.transform(input_data)

        # Predict probabilities
        college_probabilities = college_model.predict_proba(input_data)
        branch_probabilities = branch_model.predict_proba(input_data)

        
        # Get top 10 colleges and branches based on probabilities
        top_10_college_indices = np.argsort(college_probabilities[0])[::-1][:10]
        top_10_branch_indices = np.argsort(branch_probabilities[0])[::-1][:10]

        # Decode college and branch codes
        top_10_colleges = college_encoder.inverse_transform(top_10_college_indices)
        top_10_branches = branch_encoder.inverse_transform(top_10_branch_indices)

        # Create a DataFrame for results
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
                "Branch Name": branch_info['Branch Name']
            })

        results_df = pd.DataFrame(results)
        results_df.reset_index(drop=True, inplace=True)

    # Display Results
    st.success("Prediction completed!")
    
    results_df["College Code"] = results_df["College Code"].astype(int)

    # Style the DataFrame to apply colors to College Name and Branch Name
    styled_df = results_df.style.applymap(
        lambda _: "background-color: #FFD700; color: black;",  # Gold background
        subset=["College Name"]
    ).applymap(
        lambda _: "background-color: #ADD8E6; color: black;",  # Light blue background
        subset=["Branch Name"]
    ).hide(axis="index")

    # Display the styled DataFrame without the index
    st.write("Here are your results:")
    st.table(styled_df)
    st.markdown(
    """
    <div style='margin-top: 30px; font-size: 14px; text-align: center; color: gray;'>
        <i>This is a predicted result based on past data only. Please consider it as a reference.</i>
    </div>
    """,
    unsafe_allow_html=True
)

    # Show DataFrame with filtering and sorting
    #st.dataframe(results_df, use_container_width=True)

    # Reset trigger after predictions
    st.session_state.predict_triggered = False
