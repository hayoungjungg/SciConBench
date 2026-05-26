#!/usr/bin/env python3
"""
Utility script to disable filtering on all remote MCP servers.

This script sends a request to clear the filter configuration on both
Serper+Jina and Semantic Scholar+Jina servers, effectively disabling filtering.

Usage:
    python -m sciconharness.remote_mcp_servers.disable_filtering
    python -m sciconharness.remote_mcp_servers.disable_filtering --restart
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import dotenv
import requests

# Load environment variables
project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(project_root / ".env")


def disable_filtering_on_server(
    server_name: str,
    server_base: str,
    mcp_auth_token: Optional[str] = None,
    timeout: int = 30
) -> bool:
    """
    Disable filtering on a single MCP server.
    
    Args:
        server_name: Name of the server (for logging)
        server_base: Base URL of the server (without /mcp path)
        mcp_auth_token: Optional authentication token
        timeout: Request timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    config_url = f"{server_base}/configure"
    
    # Build auth headers
    auth_headers = {}
    if mcp_auth_token:
        auth_headers["Authorization"] = f"Bearer {mcp_auth_token}"
    
    # Payload to disable filtering (empty strings)
    config_payload = {
        "source_title": "",
        "publication_date": ""
    }
    
    print(f"\n{'=' * 80}")
    print(f"Disabling filtering on {server_name}")
    print(f"{'=' * 80}")
    print(f"Server: {server_name}")
    print(f"URL: {config_url}")
    print(f"Payload: {config_payload}")
    print()
    
    try:
        # Send configuration request
        response = requests.post(
            config_url,
            json=config_payload,
            headers=auth_headers,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Successfully disabled filtering on {server_name}")
            print(f"  Response: {result.get('message', 'N/A')}")
            print(f"  Filtering enabled: {result.get('filtering_enabled', 'N/A')}")
            return True
        else:
            print(f"✗ Failed to disable filtering on {server_name}")
            print(f"  Status code: {response.status_code}")
            try:
                error_body = response.json()
                print(f"  Error: {error_body}")
            except:
                print(f"  Error response: {response.text[:500]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error connecting to {server_name}: {e}")
        return False


def verify_configuration(
    server_name: str,
    server_base: str,
    mcp_auth_token: Optional[str] = None,
    timeout: int = 10
) -> bool:
    """
    Verify that filtering is disabled on a server.
    
    Args:
        server_name: Name of the server (for logging)
        server_base: Base URL of the server (without /mcp path)
        mcp_auth_token: Optional authentication token
        timeout: Request timeout in seconds
        
    Returns:
        True if filtering is disabled, False otherwise
    """
    verify_url = f"{server_base}/verify-config"
    
    # Build auth headers
    auth_headers = {}
    if mcp_auth_token:
        auth_headers["Authorization"] = f"Bearer {mcp_auth_token}"
    
    try:
        response = requests.get(
            verify_url,
            headers=auth_headers,
            timeout=timeout
        )
        
        if response.status_code == 200:
            config = response.json()
            source_title = config.get("source_title")
            publication_date = config.get("publication_date")
            configured = config.get("configured", False)
            
            if not configured or (not source_title and not publication_date):
                print(f"✓ Verified: Filtering is disabled on {server_name}")
                return True
            else:
                print(f"⚠ Warning: Filtering may still be enabled on {server_name}")
                print(f"  source_title: {source_title}")
                print(f"  publication_date: {publication_date}")
                return False
        else:
            print(f"⚠ Could not verify configuration on {server_name} (status {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"⚠ Could not verify configuration on {server_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Disable filtering on all remote MCP servers"
    )
    parser.add_argument(
        "--serper-server-base",
        type=str,
        default=None,
        help="Base URL for Serper+Jina server (default: from SERPER_SERVER_BASE env var)"
    )
    parser.add_argument(
        "--semantic-server-base",
        type=str,
        default=None,
        help="Base URL for Semantic Scholar+Jina server (default: from SEMANTIC_SERVER_BASE env var)"
    )
    parser.add_argument(
        "--mcp-auth-token",
        type=str,
        default=None,
        help="MCP authentication token (default: from MCP_AUTH_TOKEN env var)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after disabling filtering"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the MCP servers (requires system-level access)"
    )
    
    args = parser.parse_args()
    
    # Get server URLs from environment or use defaults
    serper_server_base = args.serper_server_base or os.getenv(
        "SERPER_SERVER_BASE",
        "https://patriarchical-sherri-burdenedly.ngrok-free.dev/serper"
    )
    semantic_server_base = args.semantic_server_base or os.getenv(
        "SEMANTIC_SERVER_BASE",
        "https://patriarchical-sherri-burdenedly.ngrok-free.dev/semantic"
    )
    mcp_auth_token = args.mcp_auth_token or os.getenv("MCP_AUTH_TOKEN", "")
    
    print("=" * 80)
    print("DISABLING FILTERING ON ALL MCP SERVERS")
    print("=" * 80)
    print(f"Serper Server: {serper_server_base}")
    print(f"Semantic Server: {semantic_server_base}")
    print()
    
    # Disable filtering on both servers
    results = []
    
    # Serper+Jina server
    serper_success = disable_filtering_on_server(
        "Serper+Jina",
        serper_server_base,
        mcp_auth_token
    )
    results.append(("Serper+Jina", serper_success))
    
    # Semantic Scholar+Jina server
    semantic_success = disable_filtering_on_server(
        "Semantic Scholar+Jina",
        semantic_server_base,
        mcp_auth_token
    )
    results.append(("Semantic Scholar+Jina", semantic_success))
    
    # Verify configurations if requested
    if not args.no_verify:
        print("\n" + "=" * 80)
        print("VERIFYING CONFIGURATIONS")
        print("=" * 80)
        
        verify_configuration("Serper+Jina", serper_server_base, mcp_auth_token)
        verify_configuration("Semantic Scholar+Jina", semantic_server_base, mcp_auth_token)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for server_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{server_name}: {status}")
    
    all_success = all(success for _, success in results)
    
    if args.restart:
        print("\n" + "=" * 80)
        print("RESTARTING SERVERS")
        print("=" * 80)
        print("Note: Server restart requires system-level access.")
        print("You may need to manually restart the servers or use a process manager.")
        print("\nTo restart manually:")
        print("1. Find the server processes:")
        print("   ps aux | grep 'serper_jina.main'")
        print("   ps aux | grep 'semantic_scholar_jina.main'")
        print("2. Kill the processes:")
        print("   kill <PID>")
        print("3. Restart the servers:")
        print("   python -m sciconharness.remote_mcp_servers.serper_jina.main")
        print("   python -m sciconharness.remote_mcp_servers.semantic_scholar_jina.main")
    
    print()
    if all_success:
        print("✓ Filtering has been disabled on all servers.")
        sys.exit(0)
    else:
        print("✗ Some servers failed to disable filtering.")
        sys.exit(1)


if __name__ == "__main__":
    main()

