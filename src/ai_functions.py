import asyncio
import json
import logging
import os
import requests
from typing import Dict, Any, Tuple, Optional
from clustering import get_patient_state, load_data, preprocess_data, find_closest_diagnosis
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for AI providers
AI_PROVIDER = os.environ.get('AI_PROVIDER', 'auto')  # 'ollama', 'gemini', or 'auto'
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')

# Session state for dynamic provider switching
_session_ai_provider = None

# Configure Gemini if API key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

IMPORTANT: The get_patient_state function should ONLY be used when:
- The user explicitly provides a SPECIFIC PATIENT NAME (like "John Doe", "Patient Smith")
- AND mentions a CONFIRMED DIAGNOSIS (like "diagnosed with pneumonia", "has been diagnosed with diabetes")
- Example: "Can you analyze patient John Doe who was diagnosed with pneumonia?"

DO NOT use this function for:
- General symptom descriptions ("I have a cough and fever")
- Questions asking "What could this be?"
- Symptom analysis without a specific patient name and confirmed diagnosis

For symptom descriptions without a patient name and confirmed diagnosis, provide general medical advice, possible conditions to consider, and recommend consulting a healthcare professional.

When users describe respiratory symptoms (fever, cough, sore throat, loss of smell/taste), consider:
- COVID-19 and other viral respiratory infections
- Influenza
- Upper respiratory tract infections
- Bronchitis or pneumonia
- Always recommend testing for COVID-19 if symptoms are consistent

Important disclaimers to remember:
- Always advise users to consult with healthcare professionals for serious symptoms
- Never provide specific medication dosages or prescriptions
- Encourage immediate medical attention for severe symptoms like chest pain, difficulty breathing, severe injuries, etc.
- Be clear that you are an AI assistant and not a replacement for professional medical care
- For respiratory symptoms with fever, recommend COVID-19 testing and isolation until ruled out

Current user query: """

def set_ai_provider(provider):
    """Set the AI provider for the current session"""
    global _session_ai_provider
    _session_ai_provider = provider

def get_ai_provider():
    """Get the current AI provider based on configuration and availability"""
    # Check if a session provider is set (from UI)
    if _session_ai_provider and _session_ai_provider != 'auto':
        if _session_ai_provider == 'gemini' and GEMINI_API_KEY:
            return 'gemini'
        elif _session_ai_provider == 'ollama':
            try:
                # Check if Ollama is running
                health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                health_response.raise_for_status()
                return 'ollama'
            except requests.exceptions.RequestException:
                raise Exception(f"Ollama not available at {OLLAMA_BASE_URL}")
    
    # Fallback to environment variable or auto-detection
    provider_preference = AI_PROVIDER.lower() if not _session_ai_provider else _session_ai_provider.lower()
    
    if provider_preference == 'gemini' and GEMINI_API_KEY:
        return 'gemini'
    elif provider_preference == 'ollama':
        try:
            # Check if Ollama is running
            health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            health_response.raise_for_status()
            return 'ollama'
        except requests.exceptions.RequestException:
            # Fallback to Gemini if available
            if GEMINI_API_KEY:
                logger.warning("Ollama not available, falling back to Gemini")
                return 'gemini'
            else:
                raise Exception("Ollama not available and no Gemini API key configured")
    else:
        # Auto-detection
        if GEMINI_API_KEY:
            return 'gemini'
        else:
            try:
                health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                health_response.raise_for_status()
                return 'ollama'
            except requests.exceptions.RequestException:
                raise Exception("Neither Ollama nor Gemini are available")

def get_available_providers():
    """Get list of available AI providers"""
    available = []
    
    # Check Ollama
    try:
        health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        health_response.raise_for_status()
        available.append('ollama')
    except requests.exceptions.RequestException:
        pass
    
    # Check Gemini
    if GEMINI_API_KEY:
        available.append('gemini')
    
    return available

async def call_gemini(messages: list, use_tools: bool = False) -> dict:
    """Call Google Gemini API"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Convert messages to Gemini format
        gemini_messages = []
        system_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
        
        # Combine system prompt with the conversation
        if gemini_messages and system_content:
            gemini_messages[0]["parts"][0] = system_content + "\n\n" + gemini_messages[0]["parts"][0]
        
        # For function calling with Gemini
        if use_tools:
            # Define the function schema for Gemini
            get_patient_state_function = genai.protos.FunctionDeclaration(
                name="get_patient_state",
                description="Analyze a specific named patient with a confirmed diagnosis",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "name": genai.protos.Schema(type=genai.protos.Type.STRING, description="Patient's full name (required)"),
                        "diagnosed_with": genai.protos.Schema(type=genai.protos.Type.STRING, description="Confirmed medical diagnosis (not symptoms)")
                    },
                    required=["name", "diagnosed_with"]
                )
            )
            
            tool = genai.protos.Tool(function_declarations=[get_patient_state_function])
            model_with_tools = genai.GenerativeModel(GEMINI_MODEL, tools=[tool])
            
            # Start chat with tools
            chat = model_with_tools.start_chat()
            
            # Send the latest message
            latest_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
            response = chat.send_message(latest_message)
            
            # Check if function call was made
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        return {
                            "message": {
                                "content": "",
                                "tool_calls": [{
                                    "function": {
                                        "name": part.function_call.name,
                                        "arguments": dict(part.function_call.args)
                                    }
                                }]
                            }
                        }
                    elif part.text:
                        return {"message": {"content": part.text}}
            
            return {"message": {"content": response.text}}
        else:
            # Regular chat without tools
            if gemini_messages:
                # For conversation with history, use the chat interface
                if len(gemini_messages) > 1:
                    chat = model.start_chat(history=gemini_messages[:-1])
                    latest_message = gemini_messages[-1]["parts"][0]
                    response = chat.send_message(latest_message)
                else:
                    # Single message
                    chat = model.start_chat()
                    latest_message = gemini_messages[0]["parts"][0]
                    response = chat.send_message(latest_message)
                return {"message": {"content": response.text}}
            else:
                return {"message": {"content": "No message to process"}}
                
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise

async def call_ai_provider(messages: list, use_tools: bool = False) -> dict:
    """Call the appropriate AI provider based on configuration"""
    provider = get_ai_provider()
    
    if provider == 'gemini':
        return await call_gemini(messages, use_tools)
    else:
        return await call_ollama(messages, use_tools)

async def call_ollama(messages: list, use_tools: bool = False) -> dict:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9}
    }
    if use_tools:
        payload["tools"] = [{
            "type": "function",
            "function": {
                "name": "get_patient_state",
                "description": "Analyze a specific named patient with a confirmed diagnosis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Patient's full name (required)"},
                        "diagnosed_with": {"type": "string", "description": "Confirmed medical diagnosis (not symptoms)"}
                    },
                    "required": ["name", "diagnosed_with"]
                }
            }
        }]

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API: {str(e)}")
        raise

async def ai_function(message: str) -> Tuple[str, Optional[str]]:
    function_call = None
    try:
        # Check which AI provider is available
        provider = get_ai_provider()
        
        if provider == 'ollama':
            # Check Ollama health
            try:
                health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                health_response.raise_for_status()
            except requests.exceptions.RequestException:
                if GEMINI_API_KEY:
                    provider = 'gemini'
                    logger.info("Ollama not available, switching to Gemini")
                else:
                    return "Ollama server not running and no Gemini API key configured. Please start Ollama or set GEMINI_API_KEY.", None
        elif provider == 'gemini':
            if not GEMINI_API_KEY:
                return "Gemini API key not configured. Please set GEMINI_API_KEY environment variable.", None

        result = get_chat_json()
        messages = []
        if result:
            try:
                history_json = result[0]
                history = json.loads(history_json)
                for entry in history:
                    role, content = entry.get('role'), entry['parts'][0]
                    if role in ('user', 'model'):
                        messages.append({"role": role if role == 'user' else 'assistant', "content": content})
            except (json.JSONDecodeError, KeyError, IndexError):
                messages = []

        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        user_message = {"role": "user", "content": message}

        messages = [system_message] + messages + [user_message] if not messages else messages + [user_message]

        # Only trigger function call if the message clearly indicates a specific patient with a confirmed diagnosis
        # Look for patterns like "patient [name] diagnosed with" or "analyze [name] with [diagnosis]"
        message_lower = message.lower()
        has_patient_name = any(keyword in message_lower for keyword in ["patient ", "analyze ", "john", "jane", "smith", "doe"])
        has_diagnosis_keywords = any(keyword in message_lower for keyword in ["diagnosed with", "diagnosis of", "confirmed", "has been diagnosed"])
        needs_function_call = has_patient_name and has_diagnosis_keywords

        if needs_function_call:
            try:
                response = await call_ai_provider(messages, use_tools=True)
                if "message" in response and "tool_calls" in response["message"]:
                    tool_call = response["message"]["tool_calls"][0]
                    if tool_call["function"]["name"] == "get_patient_state":
                        function_call = "get_patient_state"
                        args = tool_call["function"]["arguments"]
                        args = json.loads(args) if isinstance(args, str) else args

                        name = args.get("name", "")
                        diagnosed_with = args.get("diagnosed_with", "")

                        result = get_patient_state(name, diagnosed_with)
                        patient_name, diagnosis, severity, estimated_cost, apr_mdc, priority = result

                        answer = f"""**Medical Analysis Results for {patient_name or 'Patient'}**\n\n**Diagnosis:** {diagnosis}\n**Severity Level:** {severity}\n**Estimated Treatment Cost:** ${estimated_cost:,.2f}\n**Medical Category (APR MDC):** {apr_mdc}\n**Priority Level:** {priority}\n\n**Recommendations:**\n- Based on clustering, this condition is **{severity.lower()}** severity\n- The patient should be given **{priority.lower()}**\n- Estimated cost: **${estimated_cost:,.2f}**\n\n**Important:** Always consult professionals for medical decisions.\n\n*Powered by {provider.title()} AI*"""

                        history = [*messages, {"role": "assistant", "content": answer}]
                        formatted = [{"role": m["role"] if m["role"] == "user" else "model", "parts": [m["content"]]} for m in history]
                        await asyncio.to_thread(save_all_chat_json, [json.dumps(formatted)])
                        return answer, function_call
            except Exception as e:
                logger.warning(f"Function call fallback: {e}")

        # Fallback to general suggestion based on symptoms
        response = await call_ai_provider(messages, use_tools=False)
        answer = response["message"]["content"]

        # For symptom descriptions, provide additional context but don't call get_patient_state
        try:
            input_symptoms = message.lower()
            df, _ = preprocess_data(load_data())
            diagnoses = df['CCS Diagnosis Description'].dropna().unique()
            # Filter out obviously pediatric diagnoses
            filtered_diagnoses = [d for d in diagnoses if not any(x in d.lower() for x in ['newborn', 'neonate', 'perinatal'])]
            matched_diag = find_closest_diagnosis(input_symptoms, filtered_diagnoses)

            if matched_diag and isinstance(matched_diag, str) and matched_diag.strip():
                if not any(x in matched_diag.lower() for x in ['newborn', 'neonate', 'perinatal']):
                    # Add specific COVID-19 note if symptoms suggest it
                    covid_symptoms = ['loss of smell', 'loss of taste', 'fever', 'cough', 'tired', 'fatigue']
                    has_covid_symptoms = sum(1 for symptom in covid_symptoms if symptom in input_symptoms)
                    
                    answer += f"\n\n💡 *Based on your symptoms, this could potentially be related to:* **{matched_diag}**"
                    
                    if has_covid_symptoms >= 3:
                        answer += f"\n🦠 *Note: Your combination of symptoms (especially loss of smell/taste with fever and fatigue) could also suggest COVID-19 or other viral respiratory infections. Consider getting tested.*"
                    
                    answer += f"\n⚠️ *This is just a data-based suggestion. Please consult a healthcare professional for proper diagnosis and treatment.*"
                    answer += f"\n\n*Powered by {provider.title()} AI*"
                else:
                    logger.info(f"Matched diagnosis '{matched_diag}' filtered out due to neonatal keyword.")
            else:
                # If no match found, provide general guidance for respiratory symptoms
                respiratory_keywords = ['cough', 'fever', 'throat', 'smell', 'taste', 'breathing', 'chest']
                if any(keyword in input_symptoms for keyword in respiratory_keywords):
                    answer += f"\n\n💡 *Your symptoms suggest a possible respiratory infection. Given the combination of fever, cough, sore throat, and loss of smell, consider getting evaluated for viral infections including COVID-19.*"
                    answer += f"\n⚠️ *Please consult a healthcare professional for proper diagnosis and testing.*"
                    answer += f"\n\n*Powered by {provider.title()} AI*"
                else:
                    answer += f"\n\n*Powered by {provider.title()} AI*"
                    logger.info("No confident diagnosis match found in the database.")
        except Exception as e:
            logger.warning(f"Symptom match error: {e}")

        history = [*messages, {"role": "assistant", "content": answer}]
        formatted = [{"role": m["role"] if m["role"] == "user" else "model", "parts": [m["content"]]} for m in history]
        await asyncio.to_thread(save_all_chat_json, [json.dumps(formatted)])

        return answer, function_call
    except Exception as e:
        logger.error(f"AI Error: {e}")
        return "There was an error processing your request.", None

# Chat History Helpers

def get_chat_json():
    try:
        with open('chat.json') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        create_chat_json([])
        return []

def create_chat_json(chat):
    with open('chat.json', 'w') as f:
        json.dump(chat, f, indent=4)

def save_all_chat_json(messages):
    create_chat_json(messages)

def clear_all_chat_json():
    create_chat_json([])
