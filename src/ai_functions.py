"""
AI Functions for Medical Assistant AI

This module provides the main AI functionality for the Medical Assistant application,
handling the integration with Google's Gemini model and processing function calls.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Tuple, Optional
import google.generativeai as genai
from clustering import get_patient_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from environment variable
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')

# Define function schemas
FUNCTIONS = [{
    "name": "get_patient_state",
    "description": "Search the clusters for patient data, and return the state of the patient",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "name of the patient"},
            "diagnosed_with": {"type": "string", "description": "diagnosis of the patient"}
        },
        "required": ["name", "diagnosed_with"]
    }
}]

# System prompt for the AI
SYSTEM_PROMPT = """
You are an experienced medical professional with access to advanced clustering algorithms for patient analysis. 
When discussing patient cases:
1. DO NOT FORGET use the tools given to analyze any patient data as long as name and diagnosis are provided.
2. State the severity, estimated cost, APR MDC Description and Priority Admission to Intensive Care and recommended treatments and any given data.
3. Use medical terminology appropriately
4. Provide evidence-based recommendations
5. Maintain medical ethics and privacy
"""


async def ai_function(message: str) -> Tuple[str, Optional[str]]:
    """
    Process a user message and generate a response using Google's Gemini model.
    
    Args:
        message: The user's input message
        
    Returns:
        tuple: (response text, function call name if applicable)
    """
    if not GOOGLE_API_KEY:
        return "API key not found. Please set the GOOGLE_API_KEY environment variable.", None

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    function_call = None
    
    try:
        # Generate the content based on the message
        response = await asyncio.to_thread(
            model.generate_content,
            [{"text": message}],
            generation_config={"temperature": 0.1},  # Low temperature for more deterministic decision
            tools=[{"function_declarations": FUNCTIONS}]
        )
        
        # Check if there is a current chat history
        result = get_chat_json()
        
        try:
            # Check if function call is present in the response
            function_call = response.candidates[0].content.parts[0].function_call
            
            history = []
            
            # Process the function call if it's 'get_patient_state'
            if function_call and function_call.name == "get_patient_state":
                # Extract necessary parameters from function_call.args
                name = function_call.args["name"]
                diagnosed_with = function_call.args["diagnosed_with"]
                function_call = "get_patient_state"
                
                # Call the function with the extracted arguments
                search_results = get_patient_state(name, diagnosed_with)
                
                name, diagnosed_with, severity, estimated_cost, apr_mdc, priority = search_results
                diag_result = f"Patient: {name}\nDiagnosed with: {diagnosed_with}\nSeverity: {severity}\nEstimated Cost: {estimated_cost}\nAPR MDC Description: {apr_mdc}\nPriority: {priority}"
                context = f"patient results: {diag_result}\nQuestion: {message}"
                
                if result:
                    history_json = result[0]
                    history = json.loads(history_json)
                    history.append({'role': 'user', 'parts': [context]})
                else:
                    history = []
                    history.append({'role': 'user', 'parts': [context + SYSTEM_PROMPT]})
            else:
                # Process the case where no function call was made
                if result:
                    history_json = result[0]
                    history = json.loads(history_json)
                    history.append({'role': 'user', 'parts': [message]})
                else:
                    history = []
                    history.append({'role': 'user', 'parts': [message + SYSTEM_PROMPT]})
            
            # Generate the response based on the history
            answer, history = await generate_chat_response(model, history)
            # Save chat history
            await asyncio.to_thread(save_all_chat_json, [json.dumps(history)])
            
            return answer, function_call
        
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Something went wrong. This feature is still in Beta.", function_call
    
    except Exception as e:
        logger.error(f"Error in AI function: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Something went wrong. This feature is still in Beta.", None


async def generate_chat_response(model, history):
    """Generate a chat response using the Gemini model"""
    try:
        # Set up generation configuration
        generation_config = genai.GenerationConfig(max_output_tokens=500, temperature=0.7)
        
        # Generate response
        response = await asyncio.to_thread(
            model.generate_content,
            history,
            generation_config=generation_config
        )
        
        # Extract answer from response
        answer_parts = response.parts
        if any(part.text for part in answer_parts):
            answer = next(part.text for part in answer_parts if part.text)
            history.append({'role': 'model', 'parts': [response.text]})
        else:
            answer = "Try to ask in a different way!"
        
        return answer, history
        
    except asyncio.TimeoutError:
        return "API response timed out. Maybe you're asking for a really long answer?", history
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Something went wrong. This feature is still in Beta.", history


# Chat history functions

def get_chat_json():
    """Load chat history from the chat.json file"""
    try:
        with open('chat.json') as f:
            try:
                chat = json.load(f)
                return chat
            except json.JSONDecodeError:
                return []
    except FileNotFoundError:
        create_chat_json([])
        return []


def create_chat_json(chat):
    """Create or overwrite the chat.json file"""
    with open('chat.json', 'w') as f:
        json.dump(chat, f, indent=4)


def append_message_to_chat_json(message):
    """Append a message to the chat history"""
    chat = get_chat_json()
    chat.append(message)
    create_chat_json(chat)
    

def save_all_chat_json(messages):
    """Save all messages to the chat history"""
    create_chat_json(messages)


def clear_all_chat_json():
    """Clear the chat history"""
    create_chat_json([])
