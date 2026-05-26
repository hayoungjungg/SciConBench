"""
Tests for remote MCP server configuration and verification.

Tests that:
1. Configuration requests are sent and accepted
2. Configuration is verified to be set correctly
3. Retry logic works with rate limiting
4. Fast failure on invalid configuration
"""

import os
import time
import pytest
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sciconharness.utils.query_utils import configure_remote_mcp_servers


class TestMCPServerConfiguration:
    """Test MCP server configuration and verification."""
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests module for testing."""
        with patch('sciconharness.utils.query_utils.requests') as mock_requests:
            yield mock_requests
    
    def test_configuration_success(self, mock_requests):
        """Test successful configuration and verification."""
        # Mock health check
        health_response = MagicMock()
        health_response.status_code = 200
        mock_requests.get.return_value = health_response
        
        # Mock configure response
        config_response = MagicMock()
        config_response.status_code = 200
        config_response.json.return_value = {
            "message": "Filter configuration set successfully",
            "source_title": "Test Title",
            "publication_date": "2024-01-15"
        }
        
        # Mock verify response
        verify_response = MagicMock()
        verify_response.status_code = 200
        verify_response.json.return_value = {
            "source_title": "Test Title",
            "publication_date": "2024-01-15",
            "configured": True
        }
        
        # Set up mock to return different responses for different URLs
        def mock_get(url, **kwargs):
            if "health" in url:
                return health_response
            elif "verify-config" in url:
                return verify_response
            return health_response
        
        def mock_post(url, **kwargs):
            return config_response
        
        mock_requests.get.side_effect = mock_get
        mock_requests.post.side_effect = mock_post
        
        # Should not raise
        configure_remote_mcp_servers(
            source_title="Test Title",
            publication_date="2024-01-15",
            serper_server_base="http://test-server",
            max_retries=1,
        )
        
        # Verify configure was called
        assert mock_requests.post.called
        # Verify verify-config was called
        assert any("verify-config" in str(call) for call in mock_requests.get.call_args_list)
    
    def test_configuration_verification_failure(self, mock_requests):
        """Test that configuration verification failure breaks fast."""
        # Mock health check
        health_response = MagicMock()
        health_response.status_code = 200
        mock_requests.get.return_value = health_response
        
        # Mock configure response (success)
        config_response = MagicMock()
        config_response.status_code = 200
        config_response.json.return_value = {
            "message": "Filter configuration set successfully",
            "source_title": "Test Title",
            "publication_date": "2024-01-15"
        }
        
        # Mock verify response (mismatch)
        verify_response = MagicMock()
        verify_response.status_code = 200
        verify_response.json.return_value = {
            "source_title": "Wrong Title",  # Mismatch!
            "publication_date": "2024-01-15",
            "configured": True
        }
        
        def mock_get(url, **kwargs):
            if "health" in url:
                return health_response
            elif "verify-config" in url:
                return verify_response
            return health_response
        
        def mock_post(url, **kwargs):
            return config_response
        
        mock_requests.get.side_effect = mock_get
        mock_requests.post.side_effect = mock_post
        
        # Should raise ValueError due to configuration mismatch
        # The error message will contain "configuration verification failed" or "configuration mismatch"
        with pytest.raises(ValueError, match="configuration"):
            configure_remote_mcp_servers(
                source_title="Test Title",
                publication_date="2024-01-15",
                serper_server_base="http://test-server",
                max_retries=1,
            )
    
    def test_configuration_retry_on_failure(self, mock_requests):
        """Test that configuration retries on failure."""
        # Mock health check to fail first time, succeed second time
        health_response_fail = MagicMock()
        health_response_fail.status_code = 500
        
        health_response_success = MagicMock()
        health_response_success.status_code = 200
        
        # Mock configure response
        config_response = MagicMock()
        config_response.status_code = 200
        config_response.json.return_value = {
            "message": "Filter configuration set successfully",
            "source_title": "Test Title",
            "publication_date": "2024-01-15"
        }
        
        # Mock verify response
        verify_response = MagicMock()
        verify_response.status_code = 200
        verify_response.json.return_value = {
            "source_title": "Test Title",
            "publication_date": "2024-01-15",
            "configured": True
        }
        
        call_count = {"get": 0, "post": 0}
        
        def mock_get(url, **kwargs):
            call_count["get"] += 1
            if "health" in url:
                if call_count["get"] <= 1:
                    raise requests.exceptions.ConnectionError("Connection failed")
                return health_response_success
            elif "verify-config" in url:
                return verify_response
            return health_response_success
        
        def mock_post(url, **kwargs):
            call_count["post"] += 1
            return config_response
        
        mock_requests.get.side_effect = mock_get
        mock_requests.post.side_effect = mock_post
        
        # Should succeed after retry
        with patch('time.sleep'):  # Speed up test
            configure_remote_mcp_servers(
                source_title="Test Title",
                publication_date="2024-01-15",
                serper_server_base="http://test-server",
                max_retries=3,
            )
        
        # Should have retried
        assert call_count["get"] > 1
    
    def test_configuration_not_configured(self, mock_requests):
        """Test that unconfigured state is detected."""
        # Mock health check
        health_response = MagicMock()
        health_response.status_code = 200
        mock_requests.get.return_value = health_response
        
        # Mock configure response
        config_response = MagicMock()
        config_response.status_code = 200
        config_response.json.return_value = {
            "message": "Filter configuration set successfully",
            "source_title": "Test Title",
            "publication_date": "2024-01-15"
        }
        
        # Mock verify response (not configured)
        verify_response = MagicMock()
        verify_response.status_code = 200
        verify_response.json.return_value = {
            "source_title": None,
            "publication_date": None,
            "configured": False
        }
        
        def mock_get(url, **kwargs):
            if "health" in url:
                return health_response
            elif "verify-config" in url:
                return verify_response
            return health_response
        
        def mock_post(url, **kwargs):
            return config_response
        
        mock_requests.get.side_effect = mock_get
        mock_requests.post.side_effect = mock_post
        
        # Should raise ValueError
        # The error message will contain "configuration verification failed" or "configuration not set"
        with pytest.raises(ValueError, match="configuration"):
            configure_remote_mcp_servers(
                source_title="Test Title",
                publication_date="2024-01-15",
                serper_server_base="http://test-server",
                max_retries=1,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

