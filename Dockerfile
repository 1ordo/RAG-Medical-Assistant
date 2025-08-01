FROM python:3.12-slim-bullseye

WORKDIR /app

# Install system dependencies for curl (needed for Ollama health checks)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY data/ ./data/
COPY main.py .
COPY .env.example .env

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Set Ollama environment variables (can be overridden at runtime)
ENV OLLAMA_HOST=http://localhost:11434
ENV OLLAMA_MODEL=llama3.2
ENV OLLAMA_TIMEOUT=30

# Remove Google API key reference since we're using Ollama
# ENV GOOGLE_API_KEY=""

# Expose the port Streamlit runs on
EXPOSE 8501

# Add health check for the application
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["python", "main.py"]