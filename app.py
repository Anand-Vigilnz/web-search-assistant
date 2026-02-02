"""
Streamlit chatbot UI for the LangChain agent with MCP tools.
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


# Page configuration
st.set_page_config(
    page_title="MCP Agent Chatbot",
    page_icon="🤖",
    layout="centered",
)

# Initialize session state for message history and cached agent
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_llm" not in st.session_state:
    st.session_state.agent_llm = None
if "agent_tools" not in st.session_state:
    st.session_state.agent_tools = None
if "agent_tool_map" not in st.session_state:
    st.session_state.agent_tool_map = None

# Title
st.title("🤖 MCP Agent Chatbot")
st.caption("Powered by LangChain, OpenAI (gpt-4o-mini), and MCP tools")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input (lazy import agent only when user sends a message)
if prompt := st.chat_input("Ask me anything..."):
    from agent import run, get_llm_and_tools

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use cached agent if available so we don't re-connect to MCP on every message
                llm = st.session_state.agent_llm
                tools = st.session_state.agent_tools
                tool_map = st.session_state.agent_tool_map
                if llm is None or tools is None or tool_map is None:
                    llm, tools, tool_map = get_llm_and_tools()
                    st.session_state.agent_llm = llm
                    st.session_state.agent_tools = tools
                    st.session_state.agent_tool_map = tool_map
                response = run(prompt, llm=llm, tools=tools, tool_map=tool_map)
                
                st.write(response)
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with info
with st.sidebar:
    st.text_input("MCP URL", value=os.getenv("MCP_URL", "https://devws.vigilnz.com/sse"))
    st.text_input("VIGILNZ API KEY", value=os.getenv("VIGILNZ_API_KEY"), type="password")
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
