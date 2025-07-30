"""
Medical Assistant AI - Streamlit Interface

This module provides a Streamlit web interface for the Medical Assistant AI,
allowing users to interact with Ollama Llama 3.2 or Google Gemini for medical assistance and clustering-based analysis.
"""

import streamlit as st
import asyncio
import logging
import os
import base64
from ai_functions import (
    ai_function, clear_all_chat_json, get_ai_provider, get_available_providers, 
    set_ai_provider, GEMINI_API_KEY, OLLAMA_BASE_URL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define available functions for the AI
AVAILABLE_FUNCTIONS = {
    "get_patient_state": {
        "description": "Analyze a specific named patient with a confirmed diagnosis using medical clustering algorithms. Only used when both patient name and confirmed diagnosis are provided.",
        "parameters": {
            "name": {
                "type": "string",
                "description": "Patient's full name (required)"
            },
            "diagnosed_with": {
                "type": "string",
                "description": "Confirmed medical diagnosis (not symptoms)"
            }
        }
    }
}


def init_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_provider" not in st.session_state:
        # Try to get available providers and set default
        try:
            available = get_available_providers()
            if available:
                st.session_state.selected_provider = available[0]
            else:
                st.session_state.selected_provider = "auto"
        except:
            st.session_state.selected_provider = "auto"


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
    
    # Show current AI provider with a more dynamic approach
    try:
        current_provider = get_ai_provider()
        if current_provider == 'gemini':
            st.markdown("### 🤖 Powered by Google Gemini & Advanced Medical Clustering")
            st.success("🤖 Using Google Gemini for AI responses")
        else:
            st.markdown("### 🦙 Powered by Ollama Llama 3.2 & Advanced Medical Clustering")
            st.success("🦙 Using Ollama for AI responses")
    except Exception as e:
        st.error(f"⚠️ AI Provider Configuration Issue: {str(e)}")
        st.markdown("### ⚠️ Medical Assistant AI - Configuration Required")
        st.info("💡 Use the sidebar to configure your AI provider")
    
    st.markdown("Ask me about your symptoms, get medical insights, or analyze patient data!")
    
    # Add example queries
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **For Symptom Assessment:**
        - "I have a persistent cough, fever of 101°F, and fatigue for the past 3 days. What could this be?"
        - "What should I do for a severe headache with sensitivity to light?"
        - "I'm experiencing chest pain and shortness of breath. Is this serious?"
        
        **For Patient Analysis (requires name + confirmed diagnosis):**
        - "Can you analyze patient John Doe diagnosed with pneumonia?"
        - "Please evaluate patient Mary Smith who has been diagnosed with diabetes mellitus"
        """)
    
    st.markdown("---")


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
        
        # AI Provider Switcher
        st.subheader("🤖 AI Provider Selection")
        
        # Add refresh button to re-detect providers
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄", help="Refresh available providers"):
                # Clear session state to force re-detection
                if "selected_provider" in st.session_state:
                    del st.session_state.selected_provider
                st.rerun()
        
        try:
            available_providers = get_available_providers()
            current_provider = get_ai_provider()
            
            if available_providers:
                provider_options = []
                provider_labels = {
                    'ollama': '🦙 Ollama (Local)',
                    'gemini': '🤖 Google Gemini (Cloud)'
                }
                
                for provider in available_providers:
                    provider_options.append(provider_labels.get(provider, provider))
                
                # Add auto option if multiple providers available
                if len(available_providers) > 1:
                    provider_options.insert(0, "🚀 Auto-detect")
                    available_providers.insert(0, "auto")
                
                # Find current selection index
                try:
                    current_index = available_providers.index(st.session_state.selected_provider)
                except (ValueError, KeyError):
                    current_index = 0
                
                selected_label = st.selectbox(
                    "Choose AI Provider:",
                    provider_options,
                    index=current_index,
                    help="Select which AI model to use for responses"
                )
                
                # Map back to provider name
                selected_provider = available_providers[provider_options.index(selected_label)]
                
                # Update session state and AI functions if changed
                if selected_provider != st.session_state.selected_provider:
                    st.session_state.selected_provider = selected_provider
                    set_ai_provider(selected_provider)
                    st.rerun()
                
                # Set the provider for this session
                set_ai_provider(selected_provider)
                
                # Show current status
                if selected_provider == "auto":
                    st.info(f"🔄 Currently using: {current_provider.title()}")
                else:
                    if selected_provider == current_provider:
                        st.success(f"✅ Active: {current_provider.title()}")
                    else:
                        st.warning(f"⚠️ Selected {selected_provider} but using {current_provider}")
                
            else:
                st.error("❌ No AI providers available")
                st.markdown("""
                **Setup Instructions:**
                
                **For Gemini:**
                - Set `GEMINI_API_KEY` environment variable
                
                **For Ollama:**
                - Install and start Ollama
                - Run: `ollama pull llama3.2`
                """)
                
        except Exception as e:
            st.error(f"❌ Configuration Error: {str(e)}")
            st.markdown("""
            **Setup Instructions:**
            
            **For Gemini:**
            - Set `GEMINI_API_KEY` environment variable
            - Set `AI_PROVIDER=gemini` (optional)
            
            **For Ollama:**
            - Install and start Ollama
            - Run: `ollama pull llama3.2`
            - Set `AI_PROVIDER=ollama` (optional)
            """)
        
        st.markdown("---")
        
        # Provider Info
        st.subheader("📊 Provider Information")
        if GEMINI_API_KEY:
            st.info("🔑 Gemini API configured")
        else:
            st.warning("⚠️ Gemini API not configured")
        
        try:
            import requests
            health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            health_response.raise_for_status()
            st.info("🦙 Ollama server running")
        except:
            st.warning("⚠️ Ollama server not running")
        
        st.markdown("---")
        
        for fname, finfo in AVAILABLE_FUNCTIONS.items():
            st.subheader(f"📊 {fname}")
            st.markdown(f"*{finfo['description']}*")
            st.markdown("---")
        
        st.info(
            "ℹ️ This AI provides general medical information and patient cluster analysis. "
            "Always consult with licensed healthcare professionals for medical decisions. "
            "For emergencies, call your local emergency services immediately."
        )


def main():
    """Main application entry point"""
    init_session_state()
    clear_chat_history()
    setup_ui()
    display_sidebar()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Process user input
    if prompt := st.chat_input("Describe your symptoms or ask about patient care..."):
        asyncio.run(process_user_input(prompt))


if __name__ == "__main__":
    main()
