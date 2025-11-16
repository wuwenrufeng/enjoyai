"""MCP Snippets.

This package contains simple examples of MCP server features.
Each server demonstrates a single feature and can be run as a standalone server.

To run a server, use the command:
    uv run server basic_tool sse
"""

import importlib
import sys
from dotenv import load_dotenv

load_dotenv()
print("loaded .env file")


def run_client():
    if len(sys.argv) < 1:
        print("Usage: client <client_name>")
        sys.exit(1)

    client_name = sys.argv[1]
    print(f"Running client: {client_name}")
    try:
        module = importlib.import_module(f".{client_name}", package=__name__)
        module.main()
    except ImportError:
        print(f"Client '{client_name}' not found.")
        sys.exit(1)
