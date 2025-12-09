FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code + frontend
COPY src/ ./src
COPY frontend/ ./frontend

# Expose port
EXPOSE 8080

# Run the API
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]