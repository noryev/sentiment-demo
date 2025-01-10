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