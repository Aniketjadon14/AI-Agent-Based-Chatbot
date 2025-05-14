import os
from langchain.llms import Ollama
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def get_ollama_llm(model_name=None):
    """
    Initialize the Ollama LLM model. 
    If model_name is not provided, it will use the value from the .env file or default to 'llama3'.
    """
    if model_name is None:
        model_name = os.getenv('OLLAMA_MODEL_NAME', 'llama3')  # Default to 'llama3' if not set in the .env file

    try:
        # Initialize the Ollama model from Langchain
        llm = Ollama(model=model_name)
        return llm
    except Exception as e:
        # Handle potential errors during initialization or model setup
        print(f"Error initializing Ollama model: {e}")
        return None

def generate_response_from_ollama(query, model_name=None):
    """
    Use the Ollama LLM to generate a response for the given query.
    """
    try:
        # Get the Ollama model
        llm = get_ollama_llm(model_name)
        
        if llm is None:
            return "Error: Failed to initialize Ollama model."
        
        # Generate response using Ollama model
        response = llm(query)
        
        # Check if the response is valid
        if response:
            return response
        else:
            return "Error: No response from Ollama model."
    
    except Exception as e:
        # Handle any errors during response generation
        print(f"Error generating response from Ollama: {e}")
        return "Sorry, I encountered an error while processing your request."

# Example usage
if __name__ == "__main__":
    query = "Tell me about Llama models."
    response = generate_response_from_ollama(query)
    print(f"Response: {response}")
