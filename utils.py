import os
import requests
from dotenv import load_dotenv
from langchain.llms import BaseLLM

# Load environment variables from .env file
load_dotenv()

# Securely get the API token
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
API_TOKEN = os.getenv("HF_API_TOKEN")  # Get from environment variable
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}  # Use the secure token

class HuggingFaceLLM(BaseLLM):
    """Custom LLM that integrates Hugging Face API calls."""
    
    def _call(self, prompt: str) -> str:
        """Handles the API call to Hugging Face and returns the generated response."""
        payload = {"inputs": prompt}
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            return response.json()[0].get("generated_text", "No response generated.")
        else:
            return f"Error: {response.json()}"
    
    def _generate(self, prompt: str) -> str:
        """Implementation of the abstract method _generate."""
        return self._call(prompt)
    
    def _llm_type(self) -> str:
        """Implementation of the abstract method _llm_type."""
        return "huggingface"

    def _identifying_params(self):
        return {}
