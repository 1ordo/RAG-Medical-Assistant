# Medical Assistant AI

A medical decision support system that leverages machine learning clustering algorithms and the Ollama Llama 3.2 model to analyze patient data and provide medical assistance based on symptoms. This AI can help with general medical questions and provide insights on patient severity, costs, and treatment priority. **Note: The data used in this implementation is not accurate and is solely for demonstration purposes.**

The clustering algorithm is specifically designed for the dataset `Hospital_Inpatient_Discharges_SPARCS_De-Identified_2012.csv`. However, the source of this dataset is currently unknown. If you are using this implementation, ensure that the dataset complies with your use case and legal requirements.

## Features

- **Symptom Analysis**: Ask about your symptoms and get medical insights
- **Patient Data Analysis**: Process patient diagnoses and provide severity assessment
- **Cost Estimation**: Predict estimated treatment costs based on similar medical cases
- **Priority Classification**: Determine treatment priority categories based on diagnosis severity
- **Interactive Chat Interface**: User-friendly Streamlit chat interface powered by Ollama Llama 3.2

## Project Structure

```
medical-assistant-ai/
├── data/                   # Dataset directory
│   └── Hospital_Inpatient_Discharges.csv
├── src/                    # Source code
│   ├── app.py              # Streamlit application interface
│   ├── ai_functions.py     # Ollama Llama 3.2 integration
│   ├── clustering.py       # ML clustering algorithms
├── tests/                  # Test files
├── Dockerfile              # Docker configuration
├── requirements.txt        # Dependencies
├── main.py                 # Main entry point
├── setup_ollama.py         # Ollama setup helper
└── LICENSE                 # MIT License
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai) installed and running
- Llama 3.2 model downloaded via Ollama

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

3. Run the Ollama setup helper (optional but recommended)
```bash
python setup_ollama.py
```

4. Or manually install Ollama and the model:
   - Install Ollama from https://ollama.ai
   - Start Ollama: `ollama serve`
   - Pull Llama 3.2: `ollama pull llama3.2`

### Installation

5. Run the application
```bash
python main.py
```

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
docker run -p 8501:8501 RAG-Medical-Assistant
```

Note: You'll need to ensure Ollama is accessible from within the Docker container or modify the configuration accordingly.

Then access the application at http://localhost:8501

## How It Works

This system combines two powerful approaches:

### Medical Knowledge (Ollama Llama 3.2)
The Llama 3.2 model provides general medical knowledge and can help with:
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

The clustering system uses a hospital inpatient discharge dataset that contains information about diagnoses, severity levels, costs, and medical department classifications. The dataset should be in CSV format with the following columns:
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

