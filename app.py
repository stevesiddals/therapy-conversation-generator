# app.py
import streamlit as st
from therapy_simulator import (
    TherapySessionGenerator, ClientProfile, TherapistProfile,
    PromptConfig, TherapySession
)
from mongo_storage import MongoStorage
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")
def update_researcher_name(conv_id):
    storage = get_storage()
    new_name = st.session_state[f"researcher_{conv_id}"]
    storage.update_researcher(conv_id, new_name)

# Initialize MongoDB storage lazily
@st.cache_resource
def get_storage():
    return MongoStorage(mongodb_uri)


# Configure page
st.set_page_config(page_title="Therapy Conversation Generator", layout="wide")

# Custom CSS
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

# Create tabs
tab1, tab2 = st.tabs(["Generate", "Review"])

with tab1:
    # Create a form for all inputs
    with st.form("conversation_settings"):
        # Sidebar configuration
        with st.sidebar:
            # Storage options (at top, no header)
            save_conversation = st.checkbox("Save Conversation", value=True)
            researcher = st.text_input("Researcher Name", value="Anonymous")

            st.header("Model Parameters")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            max_tokens = st.slider("Max Tokens", 50, 500, 200)  # Changed default
            num_exchanges = st.slider("Number of Exchanges", 1, 10, 2)  # Changed default

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
                    """{approach} {style} What follows is the therapy conversation so far."""
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

        # Client Profile Configuration
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
                client_problem = st.text_area(
                    "Presenting Problem",
                    "depression and isolation",
                    height=100
                )
            with subcol5:
                client_context = st.text_area(
                    "Context",
                    "Working remotely for 2 years and struggling to maintain social connections",
                    height=100
                )

        # Therapist Profile Configuration
        with col2:
            st.header("Therapist Profile")

            therapy_approach = st.selectbox(
                "You are practicing",
                [
                    "Internal Family Systems Therapy",
                    "Unified Protocol for Emotional Disorders (UP)",
                    "Cognitive Behavioral Therapy (CBT)",
                    "Person-Centered Therapy",
                    "Psychodynamic Therapy",
                    "Solution-Focused Brief Therapy",
                    "Other (blank, please specify in therapeutic style)"
                ]
            )

            default_style = "Your aim is to provide a warm, safe and compassionate space for emotional exploration and healing."
            therapy_style = st.text_area(
                "Therapeutic Style or Additional Context",
                value=default_style,
                help="Use this field to add details about your therapeutic style, or to specify your approach if you selected 'Other' above"
            )

        # Submit button at the bottom
        generate_pressed = st.form_submit_button("Generate Conversation", type="primary")

    # Generate conversation if form is submitted
    if generate_pressed:
        try:
            st.subheader("Conversation")

            # Create a temporary status container
            status_container = st.empty()
            status_container.info("Connecting to API...")

            # Create profiles first (fast operation)
            client = ClientProfile(
                name=client_name,
                age=client_age,
                gender=client_gender,
                presenting_problem=client_problem,
                context=client_context
            )

            approach_text = (
                f"You are practicing {therapy_approach}" if therapy_approach != "Other (blank, please specify in therapeutic style)" else ""
            )

            # Create therapist profile with appropriate fields for template
            therapist = TherapistProfile(
                approach=approach_text,
                style=therapy_style
            )

            prompt_config = PromptConfig(
                therapist_system=therapist_system,
                client_system=client_system,
                therapist_context=therapist_context,
                client_context=client_context,
                therapist_instruction=therapist_instruction,
                client_instruction=client_instruction
            )

            # Initialize generator
            generator = TherapySessionGenerator(api_key=api_key)

            # Create placeholder for conversation
            messages = []
            session = None

            # Generate and display messages as they come
            with st.spinner("Generating conversation..."):

                # Clear the status once we start getting responses
                status_container.empty()

                for message, current_session in generator.generate_session(
                        client=client,
                        therapist=therapist,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        num_exchanges=num_exchanges,
                        prompt_config=prompt_config
                ):
                    messages.append(message)
                    session = current_session

                    with st.chat_message(message["role"]):
                        st.write(message["content"])

            # Save conversation after generation complete
            if save_conversation and session:
                storage = get_storage()
                storage.save_therapy_session(session, researcher)
                st.success("Conversation saved")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Review tab
with tab2:
    st.header("Stored Conversations")

    # Add filters in columns
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        researcher_filter = st.text_input("Filter by researcher")
    with filter_col2:
        approach_filter = st.selectbox(
            "Filter by approach",
            ["All"] + [
                "Internal Family Systems Therapy",
                "Unified Protocol for emotional disorders (UP)",
                "Cognitive Behavioral Therapy (CBT)",
                "Person-Centered Therapy",
                "Psychodynamic Therapy",
                "Solution-Focused Brief Therapy"
            ]
        )

    # Create table header with all columns in one row
    header_cols = st.columns([1.5, 1.5, 0.7, 0.7, 1.5, 1.5, 2.5, 1, 0.5])
    header_cols[0].markdown("**Researcher**")
    header_cols[1].markdown("**Client**")
    header_cols[2].markdown("**Age**")
    header_cols[3].markdown("**Gender**")
    header_cols[4].markdown("**Problem**")
    header_cols[5].markdown("**Approach**")
    header_cols[6].markdown("**Context**")
    header_cols[7].markdown("**View**")
    header_cols[8].markdown("**Del**")

    # Get storage only when needed
    storage = get_storage()

    # List conversations
    conversations = storage.list_conversations()

    # Apply filters
    if researcher_filter:
        conversations = [c for c in conversations if c.get("researcher", "").lower() == researcher_filter.lower()]
    if approach_filter != "All":
        conversations = [c for c in conversations if c["approach"] == approach_filter]

    # Display conversations as single-line rows
    for conv in conversations:
        cols = st.columns([1.5, 1.5, 0.7, 0.7, 1.5, 1.5, 2.5, 1, 0.5])
        researcher_name = cols[0].text_input(
            "Researcher",
            value=conv.get('researcher', 'Not specified'),
            key=f"researcher_{conv['id']}",
            label_visibility="collapsed",
            on_change=update_researcher_name,
            args=(conv['id'],)  # Only pass the conv_id
        )
        cols[1].write(conv['client_name'])
        cols[2].write(str(conv['age']))
        cols[3].write(conv['gender'])
        cols[4].write(conv['presenting_problem'][:20] + '...' if len(conv['presenting_problem']) > 20 else conv[
            'presenting_problem'])
        approach_display = conv['approach'].replace('You are practicing ', '').replace('Therapy', '').strip()
        cols[5].write(approach_display[:20])
        cols[6].write(conv.get('context', '')[:30] + '...' if conv.get('context', '') and len(
            conv.get('context', '')) > 30 else conv.get('context', ''))

        # View button

        if cols[7].button("👁️", key=f"view_{conv['id']}", help="View conversation"):
            session = storage.get_therapy_session(conv['id'])
            if session:
                with st.expander("Conversation Details", expanded=True):
                    # Show timestamp at the top
                    st.markdown(
                        f"**Session Details** - {datetime.fromisoformat(conv['timestamp']).strftime('%Y-%m-%d %H:%M')}")

                    # Create two columns for client and therapist profiles
                    profile_col1, profile_col2 = st.columns(2)

                    # Client Profile
                    with profile_col1:
                        st.header("Client Profile")

                        # First row: Name, Age, Gender
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.markdown(f"**Name:** {session.metadata.client_profile.name}")
                        with subcol2:
                            st.markdown(f"**Age:** {session.metadata.client_profile.age}")
                        with subcol3:
                            st.markdown(f"**Gender:** {session.metadata.client_profile.gender}")

                        # Second row: Problem and Context
                        st.markdown(f"**Presenting Problem:** {session.metadata.client_profile.presenting_problem}")
                        st.markdown(f"**Context:** {session.metadata.client_profile.context}")

                    # Therapist Profile
                    with profile_col2:
                        st.header("Therapist Profile")
                        st.markdown(f"**Approach:** {session.metadata.therapist_profile.approach}")
                        st.markdown(f"**Style:** {session.metadata.therapist_profile.style}")

                    # Add a separator
                    st.markdown("---")

                    # Show conversation
                    st.header("Conversation")
                    for msg in session.conversation:
                        with st.chat_message(msg["role"]):
                            st.write(msg["content"])
        # Delete button
        if cols[8].button("🗑️", key=f"delete_{conv['id']}", help="Delete conversation"):
            if storage.delete_conversation(conv['id']):
                st.success("Conversation deleted")
                st.rerun()