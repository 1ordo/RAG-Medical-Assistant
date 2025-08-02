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

# COPY the example if it exists
# You can just attempt to copy it and ignore the error during build:
COPY .env.example .env.example

# Then conditionally create .env if needed
RUN if [ -f .env.example ]; then cp .env.example .env; \
    elif [ ! -f .env ]; then echo "OLLAMA_HOST=http://host.docker.internal:11434" > .env; fi

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Set Ollama environment variables (Docker Desktop compatible)
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3.2
ENV OLLAMA_TIMEOUT=30

# Expose the port Streamlit runs on
EXPOSE 8501

# Add health check for the application
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/ || exit 1

# Command to run the application
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
