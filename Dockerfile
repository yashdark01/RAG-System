# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/

# Set PYTHONPATH to ensure imports work correctly
ENV PYTHONPATH=/app

# Expose port 8000
EXPOSE 8000

# Command to run the application using uvicorn (standard for FastAPI)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]