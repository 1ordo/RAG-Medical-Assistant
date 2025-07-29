import asyncio
import json
import logging
import os
import requests
from typing import Dict, Any, Tuple, Optional
from clustering import get_patient_state, load_data, preprocess_data, find_closest_diagnosis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')

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

Important disclaimers to remember:
- Always advise users to consult with healthcare professionals for serious symptoms
- Never provide specific medication dosages or prescriptions
- Encourage immediate medical attention for severe symptoms like chest pain, difficulty breathing, severe injuries, etc.
- Be clear that you are an AI assistant and not a replacement for professional medical care

Current user query: """

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
        try:
            health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            health_response.raise_for_status()
        except requests.exceptions.RequestException:
            return "Ollama server not running. Please start Ollama.", None

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
                response = await call_ollama(messages, use_tools=True)
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

                        answer = f"""**Medical Analysis Results for {patient_name or 'Patient'}**\n\n**Diagnosis:** {diagnosis}\n**Severity Level:** {severity}\n**Estimated Treatment Cost:** ${estimated_cost:,.2f}\n**Medical Category (APR MDC):** {apr_mdc}\n**Priority Level:** {priority}\n\n**Recommendations:**\n- Based on clustering, this condition is **{severity.lower()}** severity\n- The patient should be given **{priority.lower()}**\n- Estimated cost: **${estimated_cost:,.2f}**\n\n**Important:** Always consult professionals for medical decisions."""

                        history = [*messages, {"role": "assistant", "content": answer}]
                        formatted = [{"role": m["role"] if m["role"] == "user" else "model", "parts": [m["content"]]} for m in history]
                        await asyncio.to_thread(save_all_chat_json, [json.dumps(formatted)])
                        return answer, function_call
            except Exception as e:
                logger.warning(f"Function call fallback: {e}")

        # Fallback to general suggestion based on symptoms
        response = await call_ollama(messages, use_tools=False)
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
                    # Just provide the possible condition without detailed analysis
                    answer += f"\n\n💡 *Based on your symptoms, this could potentially be related to:* **{matched_diag}**"
                    answer += f"\n⚠️ *This is just a data-based suggestion. Please consult a healthcare professional for proper diagnosis and treatment.*"
                else:
                    logger.info(f"Matched diagnosis '{matched_diag}' filtered out due to neonatal keyword.")
            else:
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
