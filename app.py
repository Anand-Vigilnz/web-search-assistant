"""
Streamlit chatbot UI for the LangChain agent with MCP tools.
Config priority: UI > .env > defaults
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Default values (lowest priority)
DEFAULT_MCP_URL = "https://devws.vigilnz.com/sse"


def get_config():
    """Resolve config with priority: UI (session state) > .env > defaults."""
    mcp_url_ui = (st.session_state.get("mcp_url") or "").strip()
    api_key_ui = (st.session_state.get("vigilnz_api_key") or "").strip()
    openai_key_ui = (st.session_state.get("openai_api_key") or "").strip()

    return {
        "mcp_url": mcp_url_ui or os.getenv("MCP_URL", DEFAULT_MCP_URL),
        "vigilnz_api_key": api_key_ui or os.getenv("VIGILNZ_API_KEY") or os.getenv("VIGIL_API_KEY", ""),
        "openai_api_key": openai_key_ui or os.getenv("OPENAI_API_KEY", ""),
    }


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
if "agent_config_used" not in st.session_state:
    st.session_state.agent_config_used = None

# Sidebar with config (runs first so UI values are available when processing messages)
with st.sidebar:
    st.subheader("Configuration")
    st.caption("Leave empty to use .env or defaults")
    st.text_input(
        "MCP URL",
        value=os.getenv("MCP_URL", DEFAULT_MCP_URL),
        key="mcp_url",
        placeholder=DEFAULT_MCP_URL,
    )
    st.text_input(
        "VIGILNZ API KEY",
        value=os.getenv("VIGILNZ_API_KEY", ""),
        type="password",
        key="vigilnz_api_key",
        placeholder="From .env if not set",
    )
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.agent_llm = None
        st.session_state.agent_tools = None
        st.session_state.agent_tool_map = None
        st.session_state.agent_config_used = None
        st.rerun()

# Title
st.title("🤖 ToolGuard Demo App")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input (lazy import agent only when user sends a message)
if prompt := st.chat_input("Ask me anything..."):
    from agent import run, get_llm_and_tools

    config = get_config()
    # Force recreate agent if config changed (e.g. user updated MCP URL or API key in sidebar)
    config_key = (config["mcp_url"], config["vigilnz_api_key"], config["openai_api_key"])
    force_recreate = st.session_state.agent_config_used != config_key

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use cached agent if available and config unchanged
                llm = st.session_state.agent_llm
                tools = st.session_state.agent_tools
                tool_map = st.session_state.agent_tool_map
                if llm is None or tools is None or tool_map is None or force_recreate:
                    llm, tools, tool_map = get_llm_and_tools(
                        force_recreate=force_recreate,
                        openai_api_key=config["openai_api_key"] or None,
                        mcp_url=config["mcp_url"] or None,
                        vigilnz_api_key=config["vigilnz_api_key"] or None,
                    )
                    st.session_state.agent_llm = llm
                    st.session_state.agent_tools = tools
                    st.session_state.agent_tool_map = tool_map
                    st.session_state.agent_config_used = config_key
                response = run(prompt, llm=llm, tools=tools, tool_map=tool_map)
                
                st.write(response)
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
