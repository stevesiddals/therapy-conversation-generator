# app.py
import streamlit as st
from therapy_simulator import TherapySessionGenerator, ClientProfile, TherapistProfile
import os
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Set title, with custom CSS to reduce spacing
st.set_page_config(page_title="Therapy Conversation Generator", layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        h1 {
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)
st.title("AI Therapy Conversation Generator")

# Model parameters in sidebar
with st.sidebar:
    st.header("Model Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 400, 250)
    num_exchanges = st.slider("Number of Exchanges", 1, 10, 3)

# Main content in two columns
col1, col2 = st.columns(2)

# Client Profile Configuration with sub-columns
with col1:
    st.header("Client Profile")

    # First row: Name, Age, Gender
    subcol1, subcol2, subcol3 = st.columns(3)
    with subcol1:
        client_name = st.text_input("Name", "Sarah")
    with subcol2:
        client_age = st.number_input("Age", 18, 100, 35)
    with subcol3:
        client_gender = st.selectbox("Gender", ["female", "male", "non-binary"])

    # Second row: Problem and Context in side-by-side columns
    subcol4, subcol5 = st.columns(2)
    with subcol4:
        client_problem = st.text_area("Presenting Problem",
                                      "depression and isolation",
                                      height=100)
    with subcol5:
        client_context = st.text_area("Current Situation",
                                      "Working remotely for 2 years and struggling to maintain social connections",
                                      height=100)

# Therapist Profile Configuration
with col2:
    st.header("Therapist Profile")
    therapy_approach = st.selectbox("Therapeutic Approach", [
        "Cognitive Behavioral Therapy (CBT)",
        "Person-Centered Therapy",
        "Psychodynamic Therapy",
        "Solution-Focused Brief Therapy",
        "Internal Family Systems Therapy"
    ])
    therapy_style = st.selectbox("Therapeutic Style", [
        "collaborative and solution-focused",
        "empathetic and non-directive",
        "analytical and insight-oriented",
        "practical and goal-oriented"
    ])

# Generate button
if st.button("Generate Conversation", type="primary"):
    try:
        # Show loading spinner
        with st.spinner("Generating conversation..."):
            # Create profiles
            client = ClientProfile(
                name=client_name,
                age=client_age,
                gender=client_gender,
                presenting_problem=client_problem,
                context=client_context
            )

            therapist = TherapistProfile(
                approach=therapy_approach,
                style=therapy_style
            )

            # Generate conversation
            generator = TherapySessionGenerator(api_key=api_key)
            session = generator.generate_session(
                client=client,
                therapist=therapist,
                temperature=temperature,
                max_tokens=max_tokens,
                num_exchanges=num_exchanges
            )

            # Display conversation
            st.subheader("Conversation")
            for msg in session.conversation:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
