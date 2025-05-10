# Medical Assistant AI

A medical decision support system that leverages machine learning clustering algorithms and the Google Gemini AI model to analyze patient data and provide insights on severity, costs, and treatment priority. **Note: The data used in this implementation is not accurate and is solely for demonstration purposes.**

The clustering algorithm is specifically designed for the dataset `Hospital_Inpatient_Discharges_SPARCS_De-Identified_2012.csv`. However, the source of this dataset is currently unknown. If you are using this implementation, ensure that the dataset complies with your use case and legal requirements.

## Features

- **Patient Analysis**: Process patient diagnoses and provide severity assessment
- **Cost Estimation**: Predict estimated treatment costs based on similar medical cases
- **Priority Classification**: Determine treatment priority categories based on diagnosis severity
- **Interactive Chat Interface**: User-friendly Streamlit chat interface for medical professionals

## Project Structure

```
medical-assistant-ai/
├── data/                   # Dataset directory
│   └── Hospital_Inpatient_Discharges.csv
├── src/                    # Source code
│   ├── app.py              # Streamlit application interface
│   ├── ai_functions.py     # Google Gemini AI integration
│   ├── clustering.py       # ML clustering algorithms
├── tests/                  # Test files
├── Dockerfile              # Docker configuration
├── requirements.txt        # Dependencies
├── main.py                 # Main entry point
└── LICENSE                 # MIT License
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Google AI Platform API key](https://ai.google.dev/) for the Gemini model

### Installation

1. Clone this repository
```bash
git clone https://github.com/1ordo/RAG-Medical-Assistant.git
cd RAG-Medical-Assistant
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Set up environment variables for API keys
```bash
# Linux/macOS
export GOOGLE_API_KEY='your-api-key-here'

# Windows
set GOOGLE_API_KEY=your-api-key-here
```

4. Run the application
```bash
python main.py
```

### Using Docker

To run the application in a Docker container:

```bash
docker build -t RAG-Medical-Assistant .
docker run -p 8501:8501 -e GOOGLE_API_KEY='your-api-key-here' RAG-Medical-Assistant
```

The `GOOGLE_API_KEY` environment variable must be passed to the container at runtime using the `-e` flag. This ensures the application can authenticate with the Google Gemini API.

Then access the application at http://localhost:8501

## How It Works

This system uses K-means clustering to analyze hospital discharge data and group similar diagnoses based on severity and cost. The clustering model helps evaluate new patient cases by:

1. Encoding the input diagnosis
2. Finding the most similar cluster
3. Determining severity level based on the cluster characteristics
4. Estimating costs based on similar cases
5. Assigning appropriate treatment priority

The Google Gemini AI model is used to process natural language queries and extract patient information from user messages.

## Dataset

The system uses a hospital inpatient discharge dataset that contains information about diagnoses, severity levels, costs, and medical department classifications. The dataset should be in CSV format with the following columns:
- CCS Diagnosis Description
- APR Severity of Illness Description
- APR MDC Description
- Total Costs

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

