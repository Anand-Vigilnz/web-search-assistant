import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_atlassian_mcp():
    API_TOKEN = "your_actual_token"
    EMAIL = "anand@vigilnz.com"
    SITE_URL = "https://vigilnz.atlassian.net"
    
    print("Starting server with debug output...\n")
    
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@atlassian/mcp-server-atlassian"],
        env={
            "ATLASSIAN_API_TOKEN": API_TOKEN,
            "ATLASSIAN_USER_EMAIL": EMAIL,
            "ATLASSIAN_SITE_URL": SITE_URL,
            "DEBUG": "true"  # Enable debug output
        }
    )
    
    try:
        # Manually start process to see stderr
        import subprocess
        
        proc = await asyncio.create_subprocess_exec(
            "npx", "-y", "@atlassian/mcp-server-atlassian",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **dict(os.environ),
                "ATLASSIAN_API_TOKEN": API_TOKEN,
                "ATLASSIAN_USER_EMAIL": EMAIL,
                "ATLASSIAN_SITE_URL": SITE_URL
            }
        )
        
        # Wait a bit and check stderr
        await asyncio.sleep(2)
        
        if proc.stderr:
            stderr_output = await proc.stderr.read(1024)
            if stderr_output:
                print("Server Error Output:")
                print(stderr_output.decode())
        
        proc.terminate()
        await proc.wait()
        
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()

import os
asyncio.run(test_atlassian_mcp())