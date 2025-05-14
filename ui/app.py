import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from agents.tools import agent_tools
from models.ollama_model import get_ollama_llm

st.set_page_config(page_title="Personal AI Agent Chatbot")
st.title("🤖 Personal AI Agent Chatbot")

query = st.text_input("Ask a question:")

if st.button("Run Agent") and query:
    llm = get_ollama_llm()
    agent = initialize_agent(agent_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    with st.spinner("Thinking..."):
        response = agent.run(query)
    st.success(response)