FROM python:3.12-slim-bullseye

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY data/ ./data/
COPY main.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Set default environment variables (can be overridden at runtime)
ENV GOOGLE_API_KEY=""

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["python", "main.py"]