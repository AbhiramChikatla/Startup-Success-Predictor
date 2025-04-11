import streamlit as st
import pandas as pd
import joblib
import numpy as np
import google.generativeai as genai
import ast
import os
import random
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set page config
st.set_page_config(
    page_title="Startup Success Predictor", page_icon="🚀", layout="centered"
)


# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.joblib")


# Title and description
st.title("🚀 Startup Success Predictor")
st.markdown(
    """
This application predicts the likelihood of your startup's success based on various factors.
Enter your startup's details below to get a prediction.
"""
)

# Create input form
st.header("Startup Information")

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    # Age related features
    age_first_funding = st.number_input(
        "Age at First Funding (Years)",
        min_value=0.0,
        max_value=10.0,
        value=0.1,
        step=0.1,
    )
    age_last_funding = st.number_input(
        "Age at Last Funding (Years)",
        min_value=0.0,
        max_value=10.0,
        value=1.34,
        step=0.1,
    )
    age_first_milestone = st.number_input(
        "Age at First Milestone (Years)",
        min_value=0.0,
        max_value=10.0,
        value=1.37,
        step=0.1,
    )
    age_last_milestone = st.number_input(
        "Age at Last Milestone (Years)",
        min_value=0.0,
        max_value=10.0,
        value=1.67,
        step=0.1,
    )
    age_startup = st.number_input(
        "Current Age of Startup (Years)",
        min_value=0.0,
        max_value=10.0,
        value=4.31,
        step=0.1,
    )

with col2:
    # Funding related features
    funding_rounds = st.number_input(
        "Number of Funding Rounds", min_value=0, max_value=20, value=5
    )
    funding_total = st.number_input(
        "Total Funding Amount (USD in millions)", min_value=0.0, value=17.59, step=0.01
    )
    milestones = st.number_input(
        "Number of Milestones", min_value=0, max_value=20, value=3
    )
    avg_participants = st.number_input(
        "Average Participants per Round",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
    )

    # Investment rounds
    has_roundA = st.checkbox("Has Round A Funding", value=True)
    has_roundB = st.checkbox("Has Round B Funding", value=True)
    has_roundC = st.checkbox("Has Round C Funding", value=True)
    has_roundD = st.checkbox("Has Round D Funding", value=True)

with col3:
    # Location features
    st.subheader("Location")
    location = st.radio(
        "Select Location",
        ["California", "New York", "Massachusetts", "Texas", "Other State"],
        index=0,
    )

    # Industry features
    st.subheader("Industry")
    industry = st.radio(
        "Select Industry",
        [
            "Software",
            "Web",
            "Mobile",
            "Enterprise",
            "Advertising",
            "Games/Video",
            "E-commerce",
            "Biotech",
            "Consulting",
            "Other Category",
        ],
        index=0,
    )

    # Investor features
    st.subheader("Investors")
    has_VC = st.checkbox("Has VC Investment", value=True)
    has_angel = st.checkbox("Has Angel Investment", value=False)
    has_Investor = st.checkbox("Has Other Investors", value=False)
    has_Seed = st.checkbox("Has Seed Funding", value=False)
    is_top500 = st.checkbox("Is Top 500 Company", value=True)

# Submit button
submitted = st.button("Predict Success")

# Assuming you have a function `get_recommendations` that calls the Gemini API


def get_recommendations(features):
    """
    Generates recommendations based on the startup's selected features using Gemini LLM.

    Args:
    - features (dict): A dictionary where keys are feature names and values are user inputs.

    Returns:
    - list: A list of recommendation strings.
    """
    # Format the input features as a prompt
    prompt = f"""
  You are an expert startup advisor. Based on the following startup features, provide exactly 3 actionable recommendations to increase the chances of success:
  
  Startup Features:
  {features}
  
  ### Instructions:
  - Provide the recommendations as a list of strings.
  - Each recommendation should be **concise, practical, and data-driven**.
  - Format the response as follows(strictly follow this format and donot produce any other text or comments):
  [
    "Recommendation 1",
    "Recommendation 2",
    "Recommendation 3"
  ]
  example:
  [
    "Consider expanding your team.",
    "Look for additional funding opportunities.",
    "Focus on key metrics improvement.",
  ]
  """

    try:
        # Call the LLM
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        recommendations = ast.literal_eval(response_text)
        return recommendations

    except Exception as e:
     
        return [
            "Consider expanding your team.",
            "Look for additional funding opportunities.",
            "Focus on key metrics improvement.",
        ]



# Process the form submission
if submitted:
    # Validate inputs
    if any(
        value is None
        for value in [
            age_first_funding,
            age_last_funding,
            age_first_milestone,
            age_last_milestone,
            age_startup,
            funding_rounds,
            funding_total,
            milestones,
            avg_participants,
        ]
    ):
        st.error("Please fill in all the fields.")
    else:
        # Create feature dictionary
        features = {
            "age_first_funding_year": age_first_funding,
            "age_last_funding_year": age_last_funding,
            "age_first_milestone_year": age_first_milestone,
            "age_last_milestone_year": age_last_milestone,
            "funding_rounds": funding_rounds,
            "funding_total_usd": funding_total,
            "milestones": milestones,
            "is_CA": int(location == "California"),
            "is_NY": int(location == "New York"),
            "is_MA": int(location == "Massachusetts"),
            "is_TX": int(location == "Texas"),
            "is_otherstate": int(location == "Other State"),
            "is_software": int(industry == "Software"),
            "is_web": int(industry == "Web"),
            "is_mobile": int(industry == "Mobile"),
            "is_enterprise": int(industry == "Enterprise"),
            "is_advertising": int(industry == "Advertising"),
            "is_gamesvideo": int(industry == "Games/Video"),
            "is_ecommerce": int(industry == "E-commerce"),
            "is_biotech": int(industry == "Biotech"),
            "is_consulting": int(industry == "Consulting"),
            "is_othercategory": int(industry == "Other Category"),
            "has_VC": int(has_VC),
            "has_angel": int(has_angel),
            "has_roundA": int(has_roundA),
            "has_roundB": int(has_roundB),
            "has_roundC": int(has_roundC),
            "has_roundD": int(has_roundD),
            "avg_participants": avg_participants,
            "is_top500": int(is_top500),
            "has_RoundABCD": int(
                has_roundA and has_roundB and has_roundC and has_roundD
            ),
            "has_Investor": int(has_Investor),
            "has_Seed": int(has_Seed),
            "invalid_startup": 0,
            "age_startup_year": age_startup,
            "tier_relationships": random.randint(0, 1),
        }

        # Convert to DataFrame
        input_data = pd.DataFrame([features])
        

        # Load model and make prediction
        model = load_model()
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        recommendations = get_recommendations(features)

        # Display results
        st.header("Prediction Results")

        if prediction == 1:
            st.success(
                f"🎉 Your startup has a {probability*100:.2f}% chance of success!"
            )
        else:
            st.warning(
                f"⚠️ Your startup has a {probability*100:.2f}% chance of success."
            )

        # Get and display recommendations
        st.markdown("### Recommendations:")
        for rec in recommendations:
            st.markdown(f"- {rec}")

        # Add a disclaimer
        st.markdown(
            """
        ---
        *Note: This prediction is based on historical data and machine learning algorithms. 
        It should be used as a guide rather than a definitive answer.*
        """
        )

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown(
        """
    This application uses machine learning to predict startup success based on various factors including:
    - Funding history and rounds
    - Company age and milestones
    - Location and industry
    - Investor presence
    - Team size and participation
    
    The model was trained on historical startup data using XGBoost.
    """
    )

    st.header("How to Use")
    st.markdown(
        """
    1. Fill in your startup's details
    2. Click 'Predict Success'
    3. View your prediction and recommendations
    """
    )
