# app.py
import streamlit as st
from therapy_simulator import (
    TherapySessionGenerator, ClientProfile, TherapistProfile,
    PromptConfig, TherapySession
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Configure page
st.set_page_config(page_title="Therapy Conversation Generator", layout="wide")

# Custom CSS to reduce spacing
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

# Title
st.title("AI Therapy Conversation Generator")

# Sidebar configuration
with st.sidebar:
    st.header("Model Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 500, 250)
    num_exchanges = st.slider("Number of Exchanges", 1, 10, 3)

    # Advanced Settings in an expander
    with st.expander("Advanced Prompt Settings"):
        st.subheader("System Prompts")
        therapist_system = st.text_area(
            "Therapist System Prompt",
            "You are a therapist."
        )
        client_system = st.text_area(
            "Client System Prompt",
            "You are a client in a therapy session."
        )

        st.subheader("Context Templates")
        therapist_context = st.text_area(
            "Therapist Context Template",
            """You are practicing {approach}, with a {style} style. Remain compassionate and validating, providing a safe space for the client to explore their experiences. What follows is the therapy conversation so far."""
        )

        client_context = st.text_area(
            "Client Context Template",
            """You are {name}, {age} years old and {gender}. You came to therapy because {presenting_problem}. Your context: {context}. What follows is the therapy conversation so far."""
        )

        st.subheader("Response Instructions")
        therapist_instruction = st.text_input(
            "Therapist Instruction",
            "Now respond as the therapist."
        )
        client_instruction = st.text_input(
            "Client Instruction",
            "Take a moment to process what the therapist said, then respond naturally as yourself."
        )

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

    # Second row: Problem and Context
    subcol4, subcol5 = st.columns(2)
    with subcol4:
        client_problem = st.text_area("Presenting Problem",
                                      "depression and isolation",
                                      height=100)
    with subcol5:
        client_context = st.text_area("Context",
                                      "Working remotely for 2 years and struggling to maintain social connections",
                                      height=100)

# Therapist Profile Configuration
with col2:
    st.header("Therapist Profile")
    therapy_approach = st.selectbox("Therapeutic Approach", [
        "Internal Family Systems Therapy",
        "Cognitive Behavioral Therapy (CBT)",
        "Person-Centered Therapy",
        "Psychodynamic Therapy",
        "Solution-Focused Brief Therapy"
    ])
    therapy_style = st.selectbox("Therapeutic Style", [
        "empathetic and non-directive",
        "collaborative and solution-focused",
        "analytical and insight-oriented",
        "practical and goal-oriented"
    ])

# Generate button
if st.button("Generate Conversation", type="primary"):
    try:
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

            # Create prompt config
            prompt_config = PromptConfig(
                therapist_system=therapist_system,
                client_system=client_system,
                therapist_context=therapist_context,
                client_context=client_context,
                therapist_instruction=therapist_instruction,
                client_instruction=client_instruction
            )

            # Generate conversation
            generator = TherapySessionGenerator(api_key=api_key)
            session = generator.generate_session(
                client=client,
                therapist=therapist,
                temperature=temperature,
                max_tokens=max_tokens,
                num_exchanges=num_exchanges,
                prompt_config=prompt_config
            )

            # Display conversation
            st.subheader("Conversation")
            for msg in session.conversation:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")