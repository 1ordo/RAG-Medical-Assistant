"""
AI Functions for Medical Assistant AI

This module provides the main AI functionality for the Medical Assistant application,
handling the integration with Ollama Llama 3.2 model and processing function calls.
"""

import asyncio
import json
import logging
import os
import requests
from typing import Dict, Any, Tuple, Optional
from clustering import get_patient_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')

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
You are an experienced medical assistant AI with extensive knowledge of medical conditions, symptoms, and treatments. 
When helping users with their health concerns:

1. Listen carefully to their symptoms and provide helpful, evidence-based information
2. Suggest possible conditions that match their symptoms
3. Recommend when they should seek immediate medical attention
4. Provide general health advice and self-care recommendations
5. Always remind users that your advice does not replace professional medical consultation
6. Be empathetic and supportive while maintaining medical professionalism
7. If provided with patient data (name and diagnosis), use the clustering analysis tools to provide detailed insights

Important disclaimers to remember:
- Always advise users to consult with healthcare professionals for serious symptoms
- Never provide specific medication dosages or prescriptions
- Encourage immediate medical attention for severe symptoms like chest pain, difficulty breathing, severe injuries, etc.
- Be clear that you are an AI assistant and not a replacement for professional medical care

Current user query: """



async def call_ollama(messages: list, use_tools: bool = False) -> dict:
    """
    Call Ollama API with the given messages.
    
    Args:
        messages: List of conversation messages
        use_tools: Whether to include function calling tools
        
    Returns:
        dict: Response from Ollama API
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    if use_tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
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
                }
            }
        ]
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API: {str(e)}")
        raise


async def ai_function(message: str) -> Tuple[str, Optional[str]]:
    """
    Process a user message and generate a response using Ollama Llama 3.2.
    
    Args:
        message: The user's input message
        
    Returns:
        tuple: (response text, function call name if applicable)
    """
    function_call = None
    
    try:
        # Check if Ollama is running
        try:
            health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            health_response.raise_for_status()
        except requests.exceptions.RequestException:
            return "Ollama server is not running. Please start Ollama and ensure the llama3.2 model is installed.", None
        
        # Get chat history
        result = get_chat_json()
        
        # Prepare messages for Ollama
        messages = []
        
        if result:
            try:
                history_json = result[0]
                history = json.loads(history_json)
                # Convert history to Ollama format
                for entry in history:
                    if entry.get('role') == 'user':
                        messages.append({"role": "user", "content": entry['parts'][0]})
                    elif entry.get('role') == 'model':
                        messages.append({"role": "assistant", "content": entry['parts'][0]})
            except (json.JSONDecodeError, KeyError, IndexError):
                messages = []
        
        # Add system prompt and current message
        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        user_message = {"role": "user", "content": message}
        
        if not messages:
            messages = [system_message, user_message]
        else:
            messages.append(user_message)
        
        # Check if this looks like a patient inquiry that needs function calling
        patient_keywords = ["patient", "name", "diagnosed", "condition", "medical record", "cluster analysis"]
        needs_function_call = any(keyword.lower() in message.lower() for keyword in patient_keywords)
        
        # Try function calling first if it seems appropriate
        if needs_function_call:
            try:
                response = await call_ollama(messages, use_tools=True)
                
                # Check if function call was made
                if "message" in response and "tool_calls" in response["message"]:
                    tool_calls = response["message"]["tool_calls"]
                    if tool_calls and len(tool_calls) > 0:
                        tool_call = tool_calls[0]
                        if tool_call["function"]["name"] == "get_patient_state":
                            function_call = "get_patient_state"
                            args = json.loads(tool_call["function"]["arguments"])
                            name = args.get("name", "")
                            diagnosed_with = args.get("diagnosed_with", "")
                            
                            # Call the function
                            search_results = get_patient_state(name, diagnosed_with)
                            name, diagnosed_with, severity, estimated_cost, apr_mdc, priority = search_results
                            
                            diag_result = f"Patient: {name}\nDiagnosed with: {diagnosed_with}\nSeverity: {severity}\nEstimated Cost: {estimated_cost}\nAPR MDC Description: {apr_mdc}\nPriority: {priority}"
                            
                            # Generate final response with function results
                            context_message = {"role": "user", "content": f"Based on this patient analysis data: {diag_result}\n\nOriginal question: {message}"}
                            final_messages = messages + [context_message]
                            
                            final_response = await call_ollama(final_messages, use_tools=False)
                            answer = final_response["message"]["content"]
                            
                            # Save to chat history
                            history = []
                            for msg in final_messages:
                                if msg["role"] == "user":
                                    history.append({"role": "user", "parts": [msg["content"]]})
                                elif msg["role"] == "assistant":
                                    history.append({"role": "model", "parts": [msg["content"]]})
                            
                            history.append({"role": "model", "parts": [answer]})
                            await asyncio.to_thread(save_all_chat_json, [json.dumps(history)])
                            
                            return answer, function_call
            except Exception as e:
                logger.warning(f"Function calling failed, falling back to regular chat: {str(e)}")
        
        # Regular chat without function calling
        response = await call_ollama(messages, use_tools=False)
        answer = response["message"]["content"]
        
        # Save to chat history
        history = []
        for msg in messages:
            if msg["role"] == "user":
                history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                history.append({"role": "model", "parts": [msg["content"]]})
        
        history.append({"role": "model", "parts": [answer]})
        await asyncio.to_thread(save_all_chat_json, [json.dumps(history)])
        
        return answer, function_call
        
    except Exception as e:
        logger.error(f"Error in AI function: {str(e)}")
        import traceback
        traceback.print_exc()
        return "I'm having trouble connecting to the AI model. Please make sure Ollama is running and the llama3.2 model is installed.", None


async def generate_chat_response(messages):
    """Generate a chat response using Ollama"""
    try:
        response = await call_ollama(messages, use_tools=False)
        answer = response["message"]["content"]
        
        # Convert messages to history format
        history = []
        for msg in messages:
            if msg["role"] == "user":
                history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                history.append({"role": "model", "parts": [msg["content"]]})
        
        history.append({"role": "model", "parts": [answer]})
        return answer, history
        
    except asyncio.TimeoutError:
        return "API response timed out. Maybe you're asking for a really long answer?", []
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Something went wrong. This feature is still in Beta.", []


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
