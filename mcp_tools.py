"""
MCP session management and LangChain tool wrappers.
Uses proper async context manager pattern to avoid anyio TaskGroup cross-task issues.
"""

import os
import asyncio
import threading
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Default MCP connection configuration (env > defaults)
def _get_default_mcp_url() -> str:
    return os.getenv("MCP_URL", "https://devddg.vigilnz.com/sse")


def _get_default_mcp_api_key() -> str:
    return os.getenv("VIGILNZ_API_KEY") or os.getenv("VIGIL_API_KEY") or ""


# Dedicated worker thread and loop for MCP
_mcp_loop_ref: List[Optional[asyncio.AbstractEventLoop]] = [None]
_mcp_lock = threading.Lock()


def _mcp_worker(ready_event: threading.Event) -> None:
    """Run a long-lived event loop in this thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _mcp_loop_ref[0] = loop
    ready_event.set()
    loop.run_forever()


def _ensure_mcp_loop() -> asyncio.AbstractEventLoop:
    """Start the MCP worker thread and return its event loop."""
    with _mcp_lock:
        if _mcp_loop_ref[0] is not None:
            return _mcp_loop_ref[0]
        ready = threading.Event()
        t = threading.Thread(target=_mcp_worker, args=(ready,), daemon=True)
        t.start()
        ready.wait(timeout=5)
        if _mcp_loop_ref[0] is None:
            raise RuntimeError("MCP worker thread failed to start")
        return _mcp_loop_ref[0]


def _run_on_mcp_loop(coro):
    """Run a coroutine on the MCP worker loop and return the result."""
    loop = _ensure_mcp_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)


@asynccontextmanager
async def mcp_session(mcp_url: Optional[str] = None, api_key: Optional[str] = None):
    """Context manager for MCP session - ensures proper enter/exit in same task."""
    url = mcp_url if mcp_url is not None else _get_default_mcp_url()
    key = api_key if api_key is not None else _get_default_mcp_api_key()
    
    headers = {
        "X-API-Key": key,
    }
    
    # Use streamable HTTP transport (same as working simple_mcp_client); SSE fails behind proxies
    async with streamablehttp_client(url=url, headers=headers) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def list_mcp_tools_async(mcp_url: Optional[str] = None, api_key: Optional[str] = None):
    """List all available MCP tools."""
    async with mcp_session(mcp_url=mcp_url, api_key=api_key) as session:
        tools_result = await session.list_tools()
        return tools_result.tools


async def call_mcp_tool_async(tool_name: str, arguments: Dict[str, Any], 
                               mcp_url: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Call an MCP tool and return the result."""
    async with mcp_session(mcp_url=mcp_url, api_key=api_key) as session:
        result = await session.call_tool(tool_name, arguments=arguments)
        
        # Extract text content from result
        if result.content and len(result.content) > 0:
            if hasattr(result.content[0], 'text'):
                return result.content[0].text
            elif isinstance(result.content[0], str):
                return result.content[0]
            else:
                return str(result.content[0])
        return str(result)


# Cache for tool definitions to avoid reconnecting for every tool call
_cached_tools: Optional[List] = None
_cached_url: Optional[str] = None
_cached_key: Optional[str] = None


def create_langchain_tool_from_mcp(mcp_tool, mcp_url: str, api_key: str) -> StructuredTool:
    """Create a LangChain StructuredTool from an MCP tool definition."""
    
    tool_name = mcp_tool.name
    tool_description = mcp_tool.description or f"Tool: {tool_name}"
    
    # Build args schema from MCP tool's inputSchema if available
    args_schema = None
    if mcp_tool.inputSchema and isinstance(mcp_tool.inputSchema, dict):
        schema = mcp_tool.inputSchema
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        if properties:
            annotations = {}
            field_defaults = {}
            
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "string")
                prop_desc = prop_schema.get("description", f"Parameter: {prop_name}")
                is_required = prop_name in required
                
                # Map JSON schema types to Python types
                if prop_type == "string":
                    python_type = str
                elif prop_type == "integer":
                    python_type = int
                elif prop_type == "number":
                    python_type = float
                elif prop_type == "boolean":
                    python_type = bool
                elif prop_type == "array":
                    python_type = list
                else:
                    python_type = str
                
                if is_required:
                    annotations[prop_name] = python_type
                    field_defaults[prop_name] = Field(description=prop_desc)
                else:
                    annotations[prop_name] = Optional[python_type]
                    field_defaults[prop_name] = Field(default=None, description=prop_desc)
            
            ToolArgsModel = type(
                f"{tool_name}Args",
                (BaseModel,),
                {
                    "__annotations__": annotations,
                    **field_defaults,
                    "model_config": {"extra": "forbid"}
                }
            )
            args_schema = ToolArgsModel
    
    # Capture mcp_url and api_key in closure
    _url = mcp_url
    _key = api_key
    
    if args_schema:
        def tool_implementation(**kwargs) -> str:
            """Implementation that calls the MCP tool."""
            try:
                tool_args = {k: v for k, v in kwargs.items() if v is not None}
                return _run_on_mcp_loop(call_mcp_tool_async(tool_name, tool_args, _url, _key))
            except Exception as e:
                return f"Error calling tool {tool_name}: {str(e)}"
    else:
        class GenericArgs(BaseModel):
            arguments: Dict[str, Any] = Field(
                description=f"Arguments for {tool_name} as a dictionary"
            )
        args_schema = GenericArgs
        
        def tool_implementation(arguments: Dict[str, Any]) -> str:
            """Implementation that calls the MCP tool."""
            try:
                return _run_on_mcp_loop(call_mcp_tool_async(tool_name, arguments, _url, _key))
            except Exception as e:
                return f"Error calling tool {tool_name}: {str(e)}"
    
    return StructuredTool(
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
        func=tool_implementation,
    )


async def get_langchain_tools_async(mcp_url: Optional[str] = None, api_key: Optional[str] = None) -> List[StructuredTool]:
    """Get all MCP tools wrapped as LangChain tools."""
    global _cached_tools, _cached_url, _cached_key
    
    url = mcp_url if mcp_url is not None else _get_default_mcp_url()
    key = api_key if api_key is not None else _get_default_mcp_api_key()
    
    mcp_tools = await list_mcp_tools_async(mcp_url=url, api_key=key)
    langchain_tools = []
    
    for mcp_tool in mcp_tools:
        try:
            langchain_tool = create_langchain_tool_from_mcp(mcp_tool, url, key)
            langchain_tools.append(langchain_tool)
        except Exception as e:
            print(f"Warning: Failed to create LangChain tool for {mcp_tool.name}: {e}")
            continue
    
    return langchain_tools


def get_langchain_tools_sync(
    mcp_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[StructuredTool]:
    """Synchronous wrapper to get LangChain tools."""
    resolved_url = mcp_url if mcp_url is not None else _get_default_mcp_url()

    max_attempts = 3
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            return _run_on_mcp_loop(get_langchain_tools_async(mcp_url=mcp_url, api_key=api_key))
        except Exception as e:
            last_error = e
            err_msg = str(e)
            err_type = type(e).__name__
            
            # Known transient errors: retry
            is_transient = (
                "TaskGroup" in err_msg
                or "cancel" in err_msg.lower()
                or "_exceptions" in err_msg
                or "ExceptionGroup" in err_type
            )
            
            if is_transient and attempt < max_attempts - 1:
                import time
                time.sleep(0.5)
                continue
            
            raise ValueError(
                f"Failed to connect to MCP server at {resolved_url}: {e}. "
                "Ensure the MCP server is running and reachable."
            ) from e

    raise ValueError(
        f"Could not load MCP tools after {max_attempts} attempts. Last error: {last_error}. "
        f"Is the MCP server running at {resolved_url}?"
    )
