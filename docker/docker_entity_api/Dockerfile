# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if any (currently none specific to entity service)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Install dependencies from requirements.txt
# Note: This installs ALL dependencies, including GPU ones. Consider a separate requirements file for this service if needed.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./virtual_streamer ./virtual_streamer

# Expose the port the app runs on
EXPOSE 8002

# Define the command to run the application using Uvicorn
# Run the unified API app located at virtual_streamer/api/main.py
CMD ["uvicorn", "virtual_streamer.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
