# MLOps_Lab_CIE/Dockerfile
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements first to leverage Docker cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default command to run when the container starts
CMD ["python", "train.py"]