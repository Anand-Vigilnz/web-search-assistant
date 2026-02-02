"""
MCP session management and LangChain tool wrappers.
Consolidates logic from web-mcp-client.py into reusable MCP tools for LangChain.
All MCP async work runs in a dedicated worker thread with a long-lived event loop
so httpx/sniffio cleanup always sees an async context (fixes AsyncLibraryNotFoundError).
"""

import os
import asyncio
import threading
import nest_asyncio
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Apply nest_asyncio for compatibility with Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Default MCP connection configuration (env > defaults)
def _get_default_mcp_url() -> str:
    return os.getenv("MCP_URL", "https://devws.vigilnz.com/sse")


def _get_default_mcp_api_key() -> str:
    return os.getenv("VIGILNZ_API_KEY") or os.getenv("VIGIL_API_KEY") or ""


# Global session storage (accessed only from worker thread's event loop)
_session: Optional[ClientSession] = None
_read_stream = None
_write_stream = None
_stream_context = None
_session_mcp_url: Optional[str] = None
_session_api_key: Optional[str] = None

# Dedicated worker thread and loop for MCP so cleanup always runs in async context
_mcp_loop_ref: List[Optional[asyncio.AbstractEventLoop]] = [None]
_mcp_lock = threading.Lock()


def _mcp_worker(ready_event: threading.Event) -> None:
    """Run a long-lived event loop in this thread. All MCP connect/use/close runs here."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _mcp_loop_ref[0] = loop
    ready_event.set()
    loop.run_forever()


def _ensure_mcp_loop() -> asyncio.AbstractEventLoop:
    """Start the MCP worker thread and return its event loop. Idempotent."""
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
    """Run a coroutine on the MCP worker loop and return the result. Use from sync code."""
    loop = _ensure_mcp_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)


async def get_mcp_session(mcp_url: Optional[str] = None, api_key: Optional[str] = None) -> ClientSession:
    """Get or create a long-lived MCP session."""
    global _session, _read_stream, _write_stream, _stream_context, _session_mcp_url, _session_api_key

    url = mcp_url if mcp_url is not None else _get_default_mcp_url()
    key = api_key if api_key is not None else _get_default_mcp_api_key()

    # Reset session if config changed
    if _session is not None and (_session_mcp_url != url or _session_api_key != key):
        await close_session()
        _session_mcp_url = None
        _session_api_key = None

    if _session is None:
        _session_mcp_url = url
        _session_api_key = key
        headers = {
            "X-API-Key": key,
        }
        _stream_context = streamablehttp_client(url, headers=headers)
        _read_stream, _write_stream, _ = await _stream_context.__aenter__()
        _session = ClientSession(_read_stream, _write_stream)
        await _session.__aenter__()
        await _session.initialize()
    
    return _session


async def list_mcp_tools(mcp_url: Optional[str] = None, api_key: Optional[str] = None):
    """List all available MCP tools."""
    session = await get_mcp_session(mcp_url=mcp_url, api_key=api_key)
    tools_result = await session.list_tools()
    return tools_result.tools


def create_langchain_tool_from_mcp(mcp_tool) -> StructuredTool:
    """Create a LangChain StructuredTool from an MCP tool definition."""
    
    # Extract tool metadata
    tool_name = mcp_tool.name
    tool_description = mcp_tool.description or f"Tool: {tool_name}"
    
    # Build args schema from MCP tool's inputSchema if available
    args_schema = None
    if mcp_tool.inputSchema and isinstance(mcp_tool.inputSchema, dict):
        schema = mcp_tool.inputSchema
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        if properties:
            # Create a dynamic Pydantic model with actual parameter names
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
            
            # Create the model class dynamically
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
    
    # Create the tool implementation
    if args_schema:
        # Tool with specific parameters
        def tool_implementation(**kwargs) -> str:
            """Implementation that calls the MCP tool."""
            try:
                tool_args = {k: v for k, v in kwargs.items() if v is not None}
                # Run on worker loop so cleanup runs in async context
                return _run_on_mcp_loop(_call_tool_and_close(tool_name, tool_args))
            except Exception as e:
                return f"Error calling tool {tool_name}: {str(e)}"
    else:
        # Fallback: generic dict-based tool
        class GenericArgs(BaseModel):
            arguments: Dict[str, Any] = Field(
                description=f"Arguments for {tool_name} as a dictionary"
            )
        args_schema = GenericArgs
        
        def tool_implementation(arguments: Dict[str, Any]) -> str:
            """Implementation that calls the MCP tool."""
            try:
                return _run_on_mcp_loop(_call_tool_and_close(tool_name, arguments))
            except Exception as e:
                return f"Error calling tool {tool_name}: {str(e)}"
    
    # Create the LangChain tool
    return StructuredTool(
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
        func=tool_implementation,
    )


async def _call_mcp_tool_async(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Async helper to call MCP tool."""
    session = await get_mcp_session()
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


async def _call_tool_and_close(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Call MCP tool then close session so cleanup runs inside the same event loop."""
    try:
        return await _call_mcp_tool_async(tool_name, arguments)
    finally:
        await close_session()


async def get_langchain_tools(mcp_url: Optional[str] = None, api_key: Optional[str] = None) -> List[StructuredTool]:
    """Get all MCP tools wrapped as LangChain tools."""
    mcp_tools = await list_mcp_tools(mcp_url=mcp_url, api_key=api_key)
    langchain_tools = []
    
    for mcp_tool in mcp_tools:
        try:
            langchain_tool = create_langchain_tool_from_mcp(mcp_tool)
            langchain_tools.append(langchain_tool)
        except Exception as e:
            print(f"Warning: Failed to create LangChain tool for {mcp_tool.name}: {e}")
            continue
    
    return langchain_tools


def _reset_mcp_session():
    """Reset global MCP session so next call creates a fresh connection."""
    global _session, _stream_context, _read_stream, _write_stream, _session_mcp_url, _session_api_key
    _session = None
    _stream_context = None
    _read_stream = None
    _write_stream = None
    _session_mcp_url = None
    _session_api_key = None


def get_langchain_tools_sync(
    mcp_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[StructuredTool]:
    """Synchronous wrapper to get LangChain tools. Runs on MCP worker loop so cleanup sees async context.
    Config priority: passed params > .env > defaults."""
    resolved_url = mcp_url if mcp_url is not None else _get_default_mcp_url()

    async def _fetch_and_close() -> List[StructuredTool]:
        try:
            return await get_langchain_tools(mcp_url=mcp_url, api_key=api_key)
        finally:
            await close_session()

    for attempt in range(2):
        try:
            return _run_on_mcp_loop(_fetch_and_close())
        except asyncio.CancelledError:
            _reset_mcp_session()
            if attempt == 0:
                continue
            raise ValueError(
                "MCP connection was cancelled. Make sure the MCP server is running at "
                f"{resolved_url} and try again."
            )
        except Exception as e:
            if "WouldBlock" in type(e).__name__ or "EndOfStream" in type(e).__name__:
                _reset_mcp_session()
                if attempt == 0:
                    continue
            raise ValueError(
                f"Failed to connect to MCP server at {resolved_url}: {e}. "
                "Ensure the MCP server is running and reachable."
            ) from e

    raise ValueError(
        f"Could not load MCP tools after retry. Is the MCP server running at {resolved_url}?"
    )


async def close_session():
    """Close the MCP session and stream context while event loop is still running (avoids AsyncLibraryNotFoundError on teardown)."""
    global _session, _stream_context, _read_stream, _write_stream
    if _session:
        await _session.__aexit__(None, None, None)
        _session = None
    if _stream_context:
        await _stream_context.__aexit__(None, None, None)
        _stream_context = None
    _read_stream = None
    _write_stream = None
