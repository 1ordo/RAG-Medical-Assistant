# Medical Assistant AI

A medical decision support system that leverages machine learning clustering algorithms and AI models (Ollama Llama 3.2 or Google Gemini) to analyze patient data and provide medical assistance based on symptoms. This AI can help with general medical questions and provide insights on patient severity, costs, and treatment priority. **Note: The data used in this implementation is not accurate and is solely for demonstration purposes.**

The clustering algorithm is specifically designed for the dataset `Hospital_Inpatient_Discharges_SPARCS_De-Identified_2012.csv`. However, the source of this dataset is currently unknown. If you are using this implementation, ensure that the dataset complies with your use case and legal requirements.

## Features

- **Dual AI Support**: Choose between Ollama (local) or Google Gemini (cloud) AI providers
- **Symptom Analysis**: Ask about your symptoms and get medical insights
- **Patient Data Analysis**: Process patient diagnoses and provide severity assessment
- **Cost Estimation**: Predict estimated treatment costs based on similar medical cases
- **Priority Classification**: Determine treatment priority categories based on diagnosis severity
- **Interactive Chat Interface**: User-friendly Streamlit chat interface with automatic AI provider detection

## AI Provider Options

### 🦙 Ollama (Local AI)
- **Pros**: Private, no API costs, works offline
- **Cons**: Requires local installation and GPU/CPU resources
- **Setup**: Install Ollama and download Llama 3.2 model

### 🤖 Google Gemini (Cloud AI)
- **Pros**: No local resources needed, fast responses, latest AI technology
- **Cons**: Requires API key, usage costs, needs internet connection
- **Setup**: Get API key from Google AI Studio

## Project Structure

```
medical-assistant-ai/
├── data/                   # Dataset directory
│   └── Hospital_Inpatient_Discharges.csv
├── src/                    # Source code
│   ├── app.py              # Streamlit application interface
│   ├── ai_functions.py     # Ollama & Gemini AI integration
│   ├── clustering.py       # ML clustering algorithms
├── tests/                  # Test files
├── Dockerfile              # Docker configuration
├── requirements.txt        # Dependencies
├── main.py                 # Main entry point
├── .env.example            # Environment variables template
├── setup_ollama.py         # Ollama setup helper
└── LICENSE                 # MIT License
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- **Option 1**: [Ollama](https://ollama.ai) installed with Llama 3.2 model
- **Option 2**: Google Gemini API key from [Google AI Studio](https://ai.google.dev/)

### Quick Setup

1. Clone this repository
```bash
git clone https://github.com/1ordo/RAG-Medical-Assistant.git
cd RAG-Medical-Assistant
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Configure your AI provider:

#### Option A: Using Ollama (Local AI)
```bash
# Run the Ollama setup helper (optional but recommended)
python setup_ollama.py

# Or manually install:
# 1. Install Ollama from https://ollama.ai
# 2. Start Ollama: ollama serve
# 3. Pull Llama 3.2: ollama pull llama3.2
```

#### Option B: Using Google Gemini (Cloud AI)
```bash
# Copy environment template and configure
cp .env.example .env

# Edit .env file and set:
# GEMINI_API_KEY=your_api_key_here
# AI_PROVIDER=gemini
```

#### Option C: Dual Setup (Recommended)
Set up both providers for maximum flexibility. The app will automatically choose the available one or fall back if one fails.

### Installation

4. Run the application
```bash
python main.py
```

## Environment Variables

You can configure the AI provider using environment variables:

```bash
# AI Provider Configuration
AI_PROVIDER=gemini           # 'ollama' or 'gemini' (auto-detects if not set)

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Gemini Configuration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

The application will automatically detect and use the available AI provider. If both are configured, it will prefer the one specified in `AI_PROVIDER` and fall back to the other if the first is unavailable.

## Example Usage

Once the application is running, you can ask questions like:

### General Medical Questions
- "I have a persistent cough, fever of 101°F, and fatigue for the past 3 days. What could this be?"
- "What should I do for a severe headache with sensitivity to light?"
- "I'm experiencing chest pain and shortness of breath. Is this serious?"

### Patient Data Analysis
- "Can you analyze patient John Doe diagnosed with pneumonia?"
- "What's the severity and cost estimate for a patient with diabetes?"

The AI will provide helpful medical insights while always reminding you to consult with healthcare professionals for serious medical decisions.

### Using Docker

To run the application in a Docker container:

```bash
docker build -t RAG-Medical-Assistant .

# For Ollama (make sure Ollama is accessible)
docker run -p 8501:8501 RAG-Medical-Assistant

# For Gemini (pass API key as environment variable)
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key -e AI_PROVIDER=gemini RAG-Medical-Assistant
```

Note: For Ollama, ensure the service is accessible from within the Docker container or modify the configuration accordingly.

Then access the application at http://localhost:8501

## How It Works

This system combines two powerful approaches:

### Medical Knowledge (AI Models)
Both Ollama Llama 3.2 and Google Gemini provide comprehensive medical knowledge and can help with:
- Symptom analysis and possible conditions
- General health advice and recommendations
- When to seek medical attention
- Self-care suggestions

### Patient Data Analysis (K-means Clustering)
For specific patient cases, the system uses K-means clustering to analyze hospital discharge data:

1. Encoding the input diagnosis
2. Finding the most similar cluster
3. Determining severity level based on cluster characteristics
4. Estimating costs based on similar cases
5. Assigning appropriate treatment priority

## Dataset

The clustering system uses a hospital inpatient discharge dataset from New York State's SPARCS (Statewide Planning and Research Cooperative System) database. You can download the dataset from:

**Download Link**: [Hospital Inpatient Discharges (SPARCS De-Identified) - 2012](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/u4ud-w55t/about_data)

After downloading, rename the file to `Hospital_Inpatient_Discharges.csv` and place it in the `data/` directory.

The dataset contains information about diagnoses, severity levels, costs, and medical department classifications. The required columns for the clustering algorithm are:
- CCS Diagnosis Description
- APR Severity of Illness Description
- APR MDC Description
- Total Costs

## Important Disclaimers

⚠️ **This application is for educational and demonstration purposes only**
- Always consult with licensed healthcare professionals for medical decisions
- The AI provides general information, not personalized medical advice
- For emergencies, contact your local emergency services immediately
- Patient data analysis is based on historical data and may not reflect current medical standards

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions from the community! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear and descriptive messages.
4. Push your branch to your forked repository.
5. Open a pull request to the main repository.

Please ensure your contributions adhere to the project's coding standards and include appropriate tests where applicable. For major changes, consider opening an issue first to discuss your ideas.

## Disclaimer

This software is for educational and research purposes only and is not intended to be used as a medical device or to make medical decisions. All recommendations should be reviewed by qualified healthcare professionals.

