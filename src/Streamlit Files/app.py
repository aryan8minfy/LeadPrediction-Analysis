import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model_path = "models/best_model_RandomForestClassifier.pkl"
model = joblib.load(model_path)

st.title("🧠 B2B Lead Conversion Prediction")
st.markdown("Fill in the lead information to predict conversion likelihood.")

# Define all input fields
def user_input_features():
    Lead_Origin = st.selectbox("Lead Origin", ['Landing Page Submission', 'API', 'Lead Add Form', 'Lead Import'])
    Lead_Source = st.selectbox("Lead Source", ['Google', 'Direct Traffic', 'Olark Chat', 'Organic Search', 'Reference', 'Welingak Website', 'Facebook', 'Others', 'Unknown'])
    Last_Activity = st.selectbox("Last Activity", ['Email Opened', 'SMS Sent', 'Page Visited on Website', 'Olark Chat Conversation', 'Email Bounced', 'Others'])
    Country = st.selectbox("Country", ['India', 'United States', 'United Arab Emirates', 'Singapore', 'Others', 'Unknown'])
    Specialization = st.selectbox("Specialization", ['Select', 'Business Administration', 'Media and Advertising', 'Human Resource Management', 'Others', 'Unknown'])
    Occupation = st.selectbox("Current Occupation", ['Unemployed', 'Working Professional', 'Student', 'Others', 'Unknown'])
    City = st.selectbox("City", ['Mumbai', 'Thane & Outskirts', 'Other Cities', 'Unknown'])
    Last_Notable_Activity = st.selectbox("Last Notable Activity", ['Email Opened', 'SMS Sent', 'Modified', 'Email Bounced', 'Others'])

    TotalVisits = st.number_input("Total Visits", min_value=0.0, step=1.0)
    Total_Time_Spent = st.number_input("Total Time Spent on Website", min_value=0.0, step=1.0)
    Page_Views_Per_Visit = st.number_input("Page Views Per Visit", min_value=0.0, step=1.0)

    # Binary inputs
    Do_Not_Email = st.selectbox("Do Not Email", ['Yes', 'No', 'Unknown'])
    Do_Not_Call = st.selectbox("Do Not Call", ['Yes', 'No', 'Unknown'])
    Search = st.selectbox("Search", ['Yes', 'No', 'Unknown'])
    Magazine = st.selectbox("Magazine", ['Yes', 'No', 'Unknown'])
    Newspaper_Article = st.selectbox("Newspaper Article", ['Yes', 'No', 'Unknown'])
    X_Education_Forums = st.selectbox("X Education Forums", ['Yes', 'No', 'Unknown'])
    Newspaper = st.selectbox("Newspaper", ['Yes', 'No', 'Unknown'])
    Digital_Ad = st.selectbox("Digital Advertisement", ['Yes', 'No', 'Unknown'])
    Through_Recommendations = st.selectbox("Through Recommendations", ['Yes', 'No', 'Unknown'])
    Receive_Updates = st.selectbox("Receive More Updates", ['Yes', 'No', 'Unknown'])
    Update_Supply_Chain = st.selectbox("Update me on Supply Chain Content", ['Yes', 'No', 'Unknown'])
    Get_DM_Content = st.selectbox("Get updates on DM Content", ['Yes', 'No', 'Unknown'])
    Agree_Cheque = st.selectbox("Agree to pay by Cheque", ['Yes', 'No', 'Unknown'])
    Free_Copy = st.selectbox("Free copy of Mastering Interview", ['Yes', 'No', 'Unknown'])

    data = {
        'Lead Origin': Lead_Origin,
        'Lead Source': Lead_Source,
        'Last Activity': Last_Activity,
        'Country': Country,
        'Specialization': Specialization,
        'What is your current occupation': Occupation,
        'City': City,
        'Last Notable Activity': Last_Notable_Activity,
        'TotalVisits': TotalVisits,
        'Total Time Spent on Website': Total_Time_Spent,
        'Page Views Per Visit': Page_Views_Per_Visit,
        'Do Not Email': Do_Not_Email,
        'Do Not Call': Do_Not_Call,
        'Search': Search,
        'Magazine': Magazine,
        'Newspaper Article': Newspaper_Article,
        'X Education Forums': X_Education_Forums,
        'Newspaper': Newspaper,
        'Digital Advertisement': Digital_Ad,
        'Through Recommendations': Through_Recommendations,
        'Receive More Updates About Our Courses': Receive_Updates,
        'Update me on Supply Chain Content': Update_Supply_Chain,
        'Get updates on DM Content': Get_DM_Content,
        'I agree to pay the amount through cheque': Agree_Cheque,
        'A free copy of Mastering The Interview': Free_Copy
    }

    return pd.DataFrame([data])

# Collect user input
input_df = user_input_features()

# Predict and display
if st.button("Predict Conversion"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("🎯 Prediction Result")
    st.success("✅ Likely to Convert" if prediction == 1 else "❌ Unlikely to Convert")
    st.metric(label="Conversion Probability", value=f"{probability*100:.2f}%")
