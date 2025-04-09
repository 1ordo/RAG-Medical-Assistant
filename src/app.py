"""
Medical Assistant AI - Streamlit Interface

This module provides a Streamlit web interface for the Medical Assistant AI,
allowing users to interact with the clustering-based medical analysis system.
"""

import streamlit as st
import asyncio
import logging
import os
import base64
from ai_functions import ai_function, clear_all_chat_json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define available functions for the AI
AVAILABLE_FUNCTIONS = {
    "get_patient_state": {
        "description": "Analyze patient data using medical clustering algorithms to determine condition severity and recommended treatments.",
        "parameters": {
            "name": {
                "type": "string",
                "description": "Patient's full name"
            },
            "diagnosed_with": {
                "type": "string",
                "description": "Primary diagnosis or symptoms"
            }
        }
    }
}


def init_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_chat_history():
    """Clear existing chat history"""
    chat_json_path = "chat.json"
    if os.path.exists(chat_json_path):
        try:
            clear_all_chat_json()
            logger.info("Chat history cleared")
        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")


def setup_ui():
    """Set up the Streamlit user interface"""
    st.set_page_config(page_title="Medical Assistant AI", page_icon="🏥", layout="wide")
    
    # Apply custom CSS styling
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7f9;
        }
        .stButton button {
            background-color: #0083B8;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🏥 Medical Assistant AI")
    st.markdown("### Powered by Advanced Medical Clustering Analysis")


async def process_user_input(prompt):
    """Process user input and generate AI response"""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical data..."):
            try:
                # Generate AI response
                response, function_call = await ai_function(prompt)
                
                # Display appropriate response based on whether a function was called
                if function_call:
                    st.info(f"🔍 Analyzing patient data")
                    st.success("Clustering analysis complete")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "function_call": function_call
                    })
                    st.markdown(response)
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Response processing error: {str(e)}")


def display_sidebar():
    """Display the sidebar with information about available functions"""
    with st.sidebar:
        st.header("🏥 Medical Analysis Tools")
        st.markdown("---")
        
        for fname, finfo in AVAILABLE_FUNCTIONS.items():
            st.subheader(f"📊 {fname}")
            st.markdown(f"*{finfo['description']}*")
            st.markdown("---")
        
        st.info(
            "ℹ️ All recommendations are based on statistical analysis of patient clusters "
            "and should be verified by a licensed medical professional."
        )


def main():
    """Main application entry point"""
    init_session_state()
    clear_chat_history()
    setup_ui()
    display_sidebar()
    
    # Process user input
    if prompt := st.chat_input("How can I assist you with patient care today?"):
        asyncio.run(process_user_input(prompt))


if __name__ == "__main__":
    main()
