FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean

# Copy dependency files
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all necessary project files
COPY src/ ./src
COPY artifacts/ ./artifacts
COPY checkpoints/ ./checkpoints

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "src/streamlit_app.py"]
