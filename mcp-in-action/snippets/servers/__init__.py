"""MCP Snippets.

This package contains simple examples of MCP server features.
Each server demonstrates a single feature and can be run as a standalone server.

To run a server, use the command:
    uv run server basic_tool sse
"""

import importlib
import sys
from typing import Literal, cast

from dotenv import load_dotenv

load_dotenv()


def run_server():
    if len(sys.argv) < 2:
        print("Usage: server <server_name> [transport]")
        sys.exit(1)

    server_name = sys.argv[1]
    transport = sys.argv[2] if len(sys.argv) > 2 else "stdio"

    try:
        module = importlib.import_module(f".{server_name}", package=__name__)
        if getattr(module, "mcp", None):
            module.mcp.run(transport=cast(Literal["stdio", "sse"], transport))
        else:
            module.main()
    except ImportError:
        print(f"Server '{server_name}' not found.")
        sys.exit(1)
