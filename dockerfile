FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory
RUN mkdir -p /outputs

# Copy inference script
COPY run_inference.py .

# Set entrypoint
ENTRYPOINT ["python", "/workspace/run_inference.py"]
