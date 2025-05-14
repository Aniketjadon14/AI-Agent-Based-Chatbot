from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from agents.tools import agent_tools
from models.ollama_model import get_ollama_llm

# Initialize the LLM
llm = get_ollama_llm()

# Initialize the agent with error handling enabled
agent = initialize_agent(
    agent_tools, 
    llm, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    handle_parsing_errors=True  # This is correctly placed
)

# Take input query from the user
query = input("Enter your question: ")

# Run the agent with error handling in place
try:
    response = agent.run(query)  # Handle errors internally
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
