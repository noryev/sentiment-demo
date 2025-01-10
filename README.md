# Building a Simple Sentiment Analysis Inference Job

This guide walks through creating a basic sentiment analysis inference job using a lightweight model. This demo is perfect for showing how to structure a Lilypad module without requiring significant computational resources aka a gpu!.

## Project Structure
```
sentiment-demo/
├── Dockerfile
├── run_inference.py
├── requirements.txt
└── README.md
```

## Step 1: Create requirements.txt
```txt
transformers==4.36.0
torch==2.1.0
```

## Step 2: Create run_inference.py
```python
import os
import json
from transformers import pipeline

def main():
    # Get input from environment variable
    text = os.environ.get('INPUT_TEXT', 'Default text for analysis')
    
    # Initialize the sentiment analysis pipeline
    # This will use a small model suitable for CPU inference
    classifier = pipeline("sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1)  # -1 forces CPU usage
    
    try:
        # Perform inference
        result = classifier(text)
        
        # Format output
        output = {
            'input_text': text,
            'sentiment': result[0]['label'],
            'confidence': float(result[0]['score']),
            'status': 'success'
        }
    except Exception as e:
        output = {
            'input_text': text,
            'error': str(e),
            'status': 'error'
        }
    
    # Save output to the designated output directory
    os.makedirs('/outputs', exist_ok=True)
    output_path = '/outputs/result.json'
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
```

## Step 3: Create Dockerfile
```dockerfile
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
```

## Step 4: Build and Test Locally

1. Build the Docker image:
```bash
docker build -t sentiment-demo:latest .
```

2. Test locally with a sample input:
```bash
docker run -e INPUT_TEXT="I love this amazing workshop!" \
    -v $(pwd)/outputs:/outputs \
    sentiment-demo:latest
```

3. Check the results:
```bash
cat outputs/result.json
```

Expected output:
```json
{
  "input_text": "I love this amazing workshop!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998,
  "status": "success"
}
```

## Live Demo Script

Here's a sequence of commands you can use during the live demo:

1. Show the project structure:
```bash
tree sentiment-demo
```

2. Explain key components:
- Show requirements.txt
- Walk through run_inference.py
- Explain Dockerfile structure

3. Build the image:
```bash
docker build -t sentiment-demo:latest .
```

4. Run inference with different examples:
```bash
# Positive example
docker run -e INPUT_TEXT="This is a fantastic demo!" \
    -v $(pwd)/outputs:/outputs \
    sentiment-demo:latest

# Negative example
docker run -e INPUT_TEXT="This demo is confusing and complicated" \
    -v $(pwd)/outputs:/outputs \
    sentiment-demo:latest
```

5. Show real-time results:
```bash
cat outputs/result.json
```

## Common Issues and Solutions

1. Docker Build Issues:
   - Ensure Docker daemon is running
   - Check internet connection for package downloads
   - Verify Python version compatibility

2. Runtime Issues:
   - Verify the outputs directory exists and has proper permissions
   - Check environment variable is being passed correctly
   - Ensure enough system memory (at least 2GB recommended)

## Next Steps for Lilypad Integration

After demonstrating the local inference job, you can show how to:

1. Push the image to a registry:
```bash
docker tag sentiment-demo:latest your-registry/sentiment-demo:latest
docker push your-registry/sentiment-demo:latest
```

2. Create lilypad_module.json.tmpl
3. Initialize as a Git repository
4. Create a tag for versioning

This provides a natural transition to the Lilypad module creation section of your workshop.