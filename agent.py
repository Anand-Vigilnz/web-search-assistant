"""
LangChain ReAct agent with OpenAI.
Builds and runs the agent with MCP tools.
"""

import os
import json
from typing import List, Optional, Any, Dict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cache the agent components per process
_cached_llm = None
_cached_tools: Optional[List] = None
_cached_tool_map: Optional[Dict[str, Any]] = None


def get_llm_and_tools(
    force_recreate: bool = False,
    openai_api_key: Optional[str] = None,
    mcp_url: Optional[str] = None,
    vigilnz_api_key: Optional[str] = None,
):
    """Get or create LLM and tools.
    Config priority: passed params > .env > defaults."""
    global _cached_llm, _cached_tools, _cached_tool_map

    if _cached_llm is not None and not force_recreate:
        return _cached_llm, _cached_tools, _cached_tool_map

    # Lazy imports to keep initial load fast
    from langchain_openai import ChatOpenAI
    from mcp_tools import get_langchain_tools_sync

    # Initialize LLM - priority: param > .env
    resolved_openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_openai_key:
        raise ValueError(
            "OPENAI_API_KEY is required. Set it in the UI sidebar, or in .env, or as OPENAI_API_KEY env var."
        )

    llm = ChatOpenAI(
        model="gpt-5-nano",
        api_key=resolved_openai_key,
        temperature=0.7,
    )

    # Get MCP tools wrapped as LangChain tools - priority: param > .env > defaults
    tools = get_langchain_tools_sync(mcp_url=mcp_url, api_key=vigilnz_api_key)

    if not tools:
        raise ValueError("No MCP tools available. Make sure the MCP server is running.")

    # Create a tool map for easy lookup
    tool_map = {tool.name: tool for tool in tools}

    # Bind tools to LLM if it supports tool calling
    try:
        llm_with_tools = llm.bind_tools(tools)
    except AttributeError:
        # Fallback: use LLM without tool binding (manual tool calling)
        llm_with_tools = llm

    _cached_llm = llm_with_tools
    _cached_tools = tools
    _cached_tool_map = tool_map

    return llm_with_tools, tools, tool_map


def create_agent(force_recreate: bool = False):
    """Create a simple agent interface."""
    # Return a simple dict-based agent interface
    return {"type": "simple_agent"}


def run(
    query: str,
    chat_history: List = None,
    llm=None,
    tools=None,
    tool_map=None,
) -> str:
    """
    Run the agent with a user query.

    Args:
        query: User's input message
        chat_history: Optional list of previous messages for context
        llm: Optional pre-fetched LLM (avoids MCP call when cached)
        tools: Optional pre-fetched tools list
        tool_map: Optional pre-fetched tool name -> tool mapping

    Returns:
        Agent's response as a string
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    if llm is None or tools is None or tool_map is None:
        llm, tools, tool_map = get_llm_and_tools()

    max_iterations = 10
    iteration = 0

    # Build messages
    messages = []

    # Add system message
    messages.append(SystemMessage(content="""You are a helpful AI assistant with access to various tools.
Use the tools when needed to answer the user's questions accurately.
When using tools, provide clear explanations of what you're doing and why.
If a tool call fails, explain the error and try an alternative approach if possible."""))

    # Add chat history if provided
    if chat_history:
        for role, content in chat_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # Add current user query
    messages.append(HumanMessage(content=query))
    
    # Agent loop
    while iteration < max_iterations:
        iteration += 1
        
        try:
            # Get response from LLM
            response = llm.invoke(messages)
            messages.append(response)
            
            # Check if LLM wants to use tools
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
            elif hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
                tool_calls = response.additional_kwargs["tool_calls"]
            
            if not tool_calls:
                # No tool calls, return the final answer
                return response.content
            
            # Execute tool calls
            for tool_call in tool_calls:
                tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
                
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                
                if tool_name in tool_map:
                    try:
                        tool = tool_map[tool_name]
                        # Execute the tool
                        if hasattr(tool, "invoke"):
                            tool_result = tool.invoke(tool_args)
                        elif hasattr(tool, "run"):
                            tool_result = tool.run(tool_args)
                        else:
                            tool_result = tool(**tool_args)
                        
                        # Add tool result to messages
                        from langchain_core.messages import ToolMessage
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call.get("id", f"{tool_name}_{iteration}")
                        ))
                    except Exception as e:
                        from langchain_core.messages import ToolMessage
                        messages.append(ToolMessage(
                            content=f"Error calling tool {tool_name}: {str(e)}",
                            tool_call_id=tool_call.get("id", f"{tool_name}_{iteration}")
                        ))
                else:
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=f"Tool {tool_name} not found",
                        tool_call_id=tool_call.get("id", f"{tool_name}_{iteration}")
                    ))
        
        except Exception as e:
            return f"Error during agent execution: {str(e)}"
    
    # Max iterations reached
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return last_message.content
    return "I reached the maximum number of iterations. Please try rephrasing your question."


async def arun(query: str, chat_history: List = None) -> str:
    """
    Async version of run.
    
    Args:
        query: User's input message
        chat_history: Optional list of previous messages for context
        
    Returns:
        Agent's response as a string
    """
    # For now, use sync version in async context
    from asyncio import to_thread
    return await to_thread(run, query, chat_history)
