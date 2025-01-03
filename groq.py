import requests
import os

# Load the Groq API Key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def summarize_text(text):
    api_url = "https://api.groq.com/openai/v1/chat/completions"  # Updated variable name for clarity

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Prepare the payload for the Groq API request
    payload = {
        "messages": [
            {"role": "system", "content": "You are an intelligent assistant specializing in legal documents."},
            {"role": "user", "content": f"Please provide a concise summary of this legal text:\n\n{text}"}
        ],
        "model": "llama3-8b-8192",  # Ensure correct model version is used
        "max_tokens": 200,  # Slightly increased token limit for more detailed summaries
    }

    # Send the API request
    try:
        response = requests.post(api_url, json=payload, headers=headers)

        # Process the response
        if response.status_code == 200:
            summary = response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            return summary if summary else "No summary was returned by the API."
        else:
            return f"API Error: {response.status_code}, Details: {response.text}"
    except requests.exceptions.RequestException as e:
        # Handle exceptions during the API call
        return f"Request Error: {str(e)}"
