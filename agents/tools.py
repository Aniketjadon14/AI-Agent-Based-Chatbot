from langchain.agents import Tool
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import re

# Function to check and sanitize input query
def sanitize_query(query):
    # Remove any special characters that could cause issues with eval or parsing
    query = re.sub(r"[^a-zA-Z0-9\s\+\-\*\/\(\)\.]", "", query)  # Only allow alphanumeric and safe operators
    return query

# 🔍 Tool 1: Query from Chroma DB
def query_custom_knowledge(query):
    try:
        # Sanitize query to avoid any unterminated string issues
        query = sanitize_query(query)

        # Debug log for tracking query
        print(f"[DEBUG] Running query for personal notes: {query}")

        # Initialize Chroma and the retriever
        db = Chroma(persist_directory="embeddings/chroma", embedding_function=OllamaEmbeddings(model="llama3"))
        retriever = db.as_retriever()
        docs = retriever.get_relevant_documents(query)

        # If no documents are found, log and return a "no match" message
        if not docs:
            print("[DEBUG] No relevant documents found.")
            return "NO_MATCH"

        # Extract content from the top 3 documents
        result = "\n".join([doc.page_content for doc in docs[:3]])
        print(f"[DEBUG] Retrieved documents: {result}")  # Debug log for the retrieved documents
        return result

    except Exception as e:
        print(f"[ERROR] Error while querying Chroma DB: {str(e)}")
        return "Error: Something went wrong with querying the personal notes."

# 🧮 Tool 2: Simple calculator
def simple_calculator_tool(query):
    try:
        # Sanitize query before evaluation
        query = sanitize_query(query)

        # Debug log for tracking query
        print(f"[DEBUG] Running calculator tool for query: {query}")

        # Perform the calculation (caution with eval, ensure the input is safe)
        result = str(eval(query))  # Simple eval calculator
        print(f"[DEBUG] Calculator result: {result}")  # Debug log for the result
        return result

    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(f"[ERROR] Calculator error: {error_message}")  # Debug log for errors
        return error_message

# 🛠️ Final list of all tools used by the agent
custom_tool = Tool(
    name="PersonalNotesRetriever",
    func=query_custom_knowledge,
    description="Useful for answering questions about my personal notes or documents"
)

calculator_tool = Tool(
    name="Calculator",
    func=simple_calculator_tool,
    description="Performs basic calculations"
)

agent_tools = [custom_tool, calculator_tool]  # List of tools for the agent
