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
from typing import Optional

from streamlit.web import cli as stcli
import sys
import json
from streamlit.web.server.server import Server

# Load environment variables
load_dotenv()
api_keys = {
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY")
}
mongodb_uri = os.getenv("MONGODB_URI")
def update_researcher_name(conv_id):
    storage = get_storage()
    new_name = st.session_state[f"researcher_{conv_id}"]
    storage.update_researcher(conv_id, new_name)

# Initialize MongoDB storage lazily
@st.cache_resource
def get_storage():
    return MongoStorage(mongodb_uri)

# Feedback functions
def handle_add_feedback(conv_id: str, researcher: str, comment: str, rating: Optional[str] = None):
    """Handle adding feedback through Streamlit."""
    storage = get_storage()

    # Debug before save
    st.write("=== Debug: Adding Feedback ===")
    st.write(f"Conversation ID: {conv_id}")
    st.write(f"Researcher: {researcher}")
    st.write(f"Comment: {comment}")
    st.write(f"Rating: {rating}")

    success = storage.add_feedback(conv_id, researcher, comment, rating)

    if success:
        # Debug after save
        st.write("=== Debug: Save Result ===")
        st.write("Feedback saved successfully")
        feedbacks = storage.get_feedback(conv_id)
        st.write(f"Total feedback count: {len(feedbacks)}")
        st.write("Latest feedback:", feedbacks[-1] if feedbacks else "None")

        st.success("Feedback added successfully")
        st.rerun()
    else:
        st.error("Failed to add feedback")

def handle_delete_feedback(conv_id: str, timestamp):
    """Handle deleting feedback through Streamlit."""
    storage = get_storage()
    success = storage.delete_feedback(conv_id, timestamp)
    if success:
        st.success("Feedback deleted")
        st.rerun()
    else:
        st.error("Failed to delete feedback")


def handle_streamlit_event():
    """Handle messages from the component via query parameters."""
    query_params = st.query_params

    # Create a dedicated debug container at the top
    debug_container = st.empty()

    if query_params:
        debug_container.info(f"Processing query parameters: {dict(query_params)}")

        if 'type' in query_params:
            event_type = query_params['type']
            conv_id = query_params.get('convId', '')

            if event_type == 'add_feedback':
                researcher = query_params.get('researcher', 'Anonymous')
                comment = query_params.get('comment', '')
                rating = query_params.get('rating', 'neutral')

                if comment.strip():
                    debug_container.info(f"Adding feedback: {researcher} - {comment}")
                    success = handle_add_feedback(conv_id, researcher, comment, rating)
                    if success:
                        debug_container.success("Feedback added successfully!")
                    else:
                        debug_container.error("Failed to add feedback")

                    # Clear query params and rerun
                    st.query_params.clear()
                    st.rerun()

        elif event_type == 'delete_feedback':
            timestamp = query_params.get('timestamp', '')
            handle_delete_feedback(conv_id, timestamp)
            # Clear query params after handling
            st.query_params.clear()
            st.rerun()

def get_rating_emoji(rating: Optional[float]) -> str:
    """Convert numerical rating to emoji."""
    if rating is None:
        return "💬"

    # Round down to nearest integer
    rating_int = int(rating)
    rating_emojis = {
        1: "😟",  # very_negative
        2: "🙁",  # negative
        3: "😐",  # neutral
        4: "🙂",  # positive
        5: "😊"  # very_positive
    }
    return rating_emojis.get(rating_int, "💬")

def render_native_feedback(conv_id, current_researcher_name):
    """Render a native Streamlit feedback interface."""
    storage = get_storage()
    feedback_list = storage.get_feedback(conv_id)

    # Initialize session state for rating if not exists
    if f"rating_{conv_id}" not in st.session_state:
        st.session_state[f"rating_{conv_id}"] = "neutral"

    # Rating configuration using emojis
    rating_config = {
        'very_negative': {'icon': '😟', 'color': '#ef4444', 'label': 'Very Negative'},
        'negative': {'icon': '🙁️', 'color': '#f97316', 'label': 'Negative'},
        'neutral': {'icon': '😐', 'color': '#6b7280', 'label': 'Neutral'},
        'positive': {'icon': '🙂️', 'color': '#22c55e', 'label': 'Positive'},
        'very_positive': {'icon': '😊', 'color': '#3b82f6', 'label': 'Very Positive'}
    }

    # Custom CSS for selected button state and to hide form borders
    st.markdown("""
        <style>
        .selected-rating button {
            background-color: #e5e7eb !important;
            border-color: #666 !important;
        }
        /* Remove form borders */
        .stForm {
            border: none !important;
            padding: 0 !important;
        }
        /* Hide the CTRL+Enter text */
        .stTextArea .st-emotion-cache-16idsys p {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    # Container for existing feedback
    st.write("#### Existing Feedback")
    if not feedback_list:
        st.info("No feedback yet")
    else:
        for feedback in feedback_list:
            with st.container():
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    if isinstance(feedback['timestamp'], str):
                        timestamp_str = datetime.fromisoformat(feedback['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        timestamp_str = feedback['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

                    st.markdown(
                        f"""
                        **{feedback['researcher_name']}** · {timestamp_str}  
                        {feedback['comment']}
                        """
                    )
                with col2:
                    rating_info = rating_config[feedback.get('rating', 'neutral')]
                    st.markdown(
                        f"<div style='text-align: center; font-size: 1.2em;'>{rating_info['icon']}</div>",
                        unsafe_allow_html=True
                    )
                with col3:
                    if st.button("🗑️", key=f"delete_{conv_id}_{timestamp_str.replace(' ', '_')}"):
                        if storage.delete_feedback(conv_id, feedback['timestamp']):
                            st.success("Feedback deleted")
                            st.rerun()
                        else:
                            st.error("Failed to delete feedback")
                st.divider()

    # New feedback section
    st.write("#### Add Feedback")

    # Rating selector first
    st.write("Rating:")
    rating_cols = st.columns(5)
    for i, (rating, config) in enumerate(rating_config.items()):
        with rating_cols[i]:
            if st.session_state[f"rating_{conv_id}"] == rating:
                st.markdown('<div class="selected-rating">', unsafe_allow_html=True)

            if st.button(
                    f"{config['icon']}\n{config['label']}",
                    key=f"rating_button_{rating}_{conv_id}",
                    use_container_width=True,
                    type="secondary"
            ):
                st.session_state[f"rating_{conv_id}"] = rating
                st.rerun()

            if st.session_state[f"rating_{conv_id}"] == rating:
                st.markdown('</div>', unsafe_allow_html=True)

    # Form for comment and submit button
    with st.form(key=f"feedback_form_{conv_id}", clear_on_submit=True):
        comment = st.text_area(
            "Your feedback",
            max_chars=1000,
            key=f"feedback_comment_{conv_id}"
        )
        submitted = st.form_submit_button("Add Feedback")

        if submitted and comment.strip():
            # Use current_researcher_name from the sidebar
            success = storage.add_feedback(
                conv_id,
                current_researcher_name,  # Use the current researcher name
                comment,
                st.session_state[f"rating_{conv_id}"]
            )
            if success:
                st.success("Feedback added!")
                st.session_state[f"rating_{conv_id}"] = "neutral"
                st.rerun()
            else:
                st.error("Failed to add feedback")

# Configure page
st.set_page_config(page_title="Therapy Conversation Generator", layout="wide")
handle_streamlit_event()

# Initialize session state for expanded states if not exists
if 'expanded_conversation' not in st.session_state:
    st.session_state.expanded_conversation = None
if 'expanded_feedback' not in st.session_state:
    st.session_state.expanded_feedback = None
if 'feedback_added' not in st.session_state:
    st.session_state.feedback_added = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

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
tab1, tab2, tab3 = st.tabs(["Generate", "Review", "About"])

with tab1:
    # Create a form for all inputs
    with st.form("conversation_settings"):
        # Sidebar configuration
        with st.sidebar:
            # Storage options (at top, no header)
            researcher = st.text_input("Researcher Name", value="Anonymous")
            save_conversation = st.checkbox("Save Conversation", value=True)

            st.header("Model Parameters")
            model = st.selectbox(
                "Model",
                options=list(TherapySessionGenerator.MODELS.keys()),
                format_func=lambda x: TherapySessionGenerator.MODELS[x][0]
            )
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
            generator = TherapySessionGenerator(api_keys=api_keys)

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
                        model=model,
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

    # Create table header
    header_cols = st.columns([1.0, 0.7, 0.5, 0.7, 1.7, 1.5, 2.3, 0.6, 0.5, 0.7, 0.6])
    header_cols[0].markdown("**Researcher**")
    header_cols[1].markdown("**Client**")
    header_cols[2].markdown("**Age**")
    header_cols[3].markdown("**Gender**")
    header_cols[4].markdown("**Problem**")
    header_cols[5].markdown("**Approach**")
    header_cols[6].markdown("**Context**")
    header_cols[7].markdown("**Model**")
    header_cols[8].markdown("**View**")
    header_cols[9].markdown("**Feedback**")
    header_cols[10].markdown("**Delete**")

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
        # Create a container for this conversation row and its details
        row_container = st.container()

        with row_container:
            # Main row with columns
            cols = st.columns([1.0, 0.7, 0.5, 0.7, 1.7, 1.5, 2.3, 0.6, 0.5, 0.7, 0.6])

            # Basic information columns
            researcher_name = cols[0].text_input(
                "Researcher",
                value=conv.get('researcher', 'Not specified'),
                key=f"researcher_{conv['id']}",
                label_visibility="collapsed",
                on_change=update_researcher_name,
                args=(conv['id'],)
            )
            cols[1].write(conv['client_name'])
            cols[2].write(str(conv['age']))
            cols[3].write(conv['gender'])
            cols[4].write(conv['presenting_problem'][:27] + '...' if len(conv['presenting_problem']) > 27 else conv[
                'presenting_problem'])
            approach_display = conv['approach'].replace('You are practicing ', '').replace('Therapy', '').strip()
            cols[5].write(approach_display[:30])
            cols[6].write(conv.get('context', '')[:38] + '...' if conv.get('context', '') and len(
                conv.get('context', '')) > 38 else conv.get('context', ''))
            cols[7].write(
                TherapySessionGenerator.MODELS.get(conv['metadata']['model'], (conv['metadata']['model'],))[0][:6])
            # View button
            if cols[8].button("👁️", key=f"view_{conv['id']}", help="View conversation"):
                if st.session_state.expanded_conversation == conv['id']:
                    st.session_state.expanded_conversation = None
                else:
                    st.session_state.expanded_conversation = conv['id']
                    st.session_state.expanded_feedback = None

            # Feedback button
            feedback_count = conv['feedback_count']
            average_rating = conv.get('average_rating')
            emoji = get_rating_emoji(average_rating)
            feedback_btn_label = f"{emoji} {feedback_count}" if feedback_count > 0 else "💬"

            if cols[9].button(
                    feedback_btn_label,
                    key=f"feedback_{conv['id']}",
                    help="View/add feedback"
            ):
                if st.session_state.expanded_feedback == conv['id']:
                    st.session_state.expanded_feedback = None
                else:
                    st.session_state.expanded_feedback = conv['id']
                    st.session_state.expanded_conversation = None

            # Delete button
            if cols[10].button("🗑️", key=f"delete_{conv['id']}", help="Delete conversation"):
                if storage.delete_conversation(conv['id']):
                    st.success("Conversation deleted")
                    st.rerun()

        # Conversation details container
        if st.session_state.expanded_conversation == conv['id']:
            details_container = st.container()
            with details_container:
                session = storage.get_therapy_session(conv['id'])
                if session:
                    # Show timestamp at the top
                    st.markdown(
                        f"**Generated on:** {datetime.fromisoformat(conv['timestamp']).strftime('%Y-%m-%d %H:%M')}. "
                        f"**Model:** {TherapySessionGenerator.MODELS.get(session.metadata.model, (session.metadata.model,))[0]}")
                    # Create two columns for client and therapist profiles
                    profile_col1, profile_col2 = st.columns(2)

                    # Client Profile
                    with profile_col1:
                        st.header("Client Profile")
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.markdown(f"**Name:** {session.metadata.client_profile.name}")
                        with subcol2:
                            st.markdown(f"**Age:** {session.metadata.client_profile.age}")
                        with subcol3:
                            st.markdown(f"**Gender:** {session.metadata.client_profile.gender}")
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

        # Feedback container
        if st.session_state.expanded_feedback == conv['id']:
            render_native_feedback(conv['id'], researcher)

with tab3:
    st.markdown("""
        ## Purpose
        * **Explore how AI therapy works** by generating conversations with a variety of clients situations and therapeutic approaches
        * **Review conversations** others have generated to get ideas and inspiration for how AI works in this context
        * **Compare different therapeutic approaches** for different client contexts
        * **Contribute your ideas** to further research

        ## Background
        Generative AI may have potential to provide meaningful emotional support. A recently published study interviewed
        19 individuals about their experiences of using AI for mental health, finding that:
        * Users experienced meaningful emotional support and guidance
        * AI interactions helped improve real-world relationships
        * Participants reported healing from trauma and loss
        * Users emphasised the need for more sophisticated safety measures

        ["It happened to be the perfect thing": experiences of generative AI chatbots for mental health](https://www.nature.com/articles/s44184-024-00097-4)

        ## Important Disclaimer
        **This is a research tool**, designed to help study AI-generated therapeutic conversations and should not be 
        used as a substitute for professional mental health support. Use at your own risk.

        ## Source
        The source code for this project is available on GitHub: 
        [therapy-conversation-generator](https://github.com/stevesiddals/therapy-conversation-generator)
        For research inquiries or more information about this tool, you can connect with the creator, Steve Siddals,
        via [LinkedIn](https://www.linkedin.com/in/stevensiddals/). The project was built in collaboration with 
        Anthropic's Claude AI assistant. 

    """)