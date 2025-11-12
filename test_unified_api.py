#!/usr/bin/env python3
"""
Test script for the unified API migration.

This script verifies that all endpoints are working correctly after the migration.
"""

import httpx
import sys
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def print_result(test_name: str, success: bool, message: str = ""):
    """Print test result with color."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"   {message}")


def test_health_check() -> bool:
    """Test the health check endpoint."""
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print_result("Health Check", True, f"Status: {data['status']}, Device: {data['device']}")
            return True
        else:
            print_result("Health Check", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Health Check", False, f"Error: {str(e)}")
        return False


def test_root_endpoint() -> bool:
    """Test the root endpoint."""
    try:
        response = httpx.get(f"{BASE_URL}/", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print_result("Root Endpoint", True, f"API: {data['name']}")
            return True
        else:
            print_result("Root Endpoint", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Root Endpoint", False, f"Error: {str(e)}")
        return False


def test_wav2lip_health() -> bool:
    """Test the Wav2Lip health endpoint."""
    try:
        response = httpx.get(f"{BASE_URL}/api/v1/wav2lip/health", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            model_loaded = data.get('model_loaded', False)
            cached_chars = data.get('cached_characters', 0)
            print_result(
                "Wav2Lip Health", 
                True, 
                f"Model loaded: {model_loaded}, Cached characters: {cached_chars}"
            )
            return True
        else:
            print_result("Wav2Lip Health", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Wav2Lip Health", False, f"Error: {str(e)}")
        return False


def test_api_docs() -> bool:
    """Test that API docs are accessible."""
    try:
        response = httpx.get(f"{BASE_URL}/docs", timeout=10.0)
        if response.status_code == 200:
            print_result("API Documentation", True, "Swagger UI accessible")
            return True
        else:
            print_result("API Documentation", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("API Documentation", False, f"Error: {str(e)}")
        return False


def test_characters_endpoint() -> bool:
    """Test the characters listing endpoint (now using internal storage)."""
    try:
        response = httpx.get(f"{BASE_URL}/api/v1/characters", timeout=10.0)
        # Should return 200 with list of characters (or empty list)
        if response.status_code == 200:
            data = response.json()
            count = len(data) if isinstance(data, list) else 0
            print_result("Characters Endpoint", True, f"Found {count} characters (internal storage)")
            return True
        else:
            print_result("Characters Endpoint", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Characters Endpoint", False, f"Error: {str(e)}")
        return False


def test_legacy_process_endpoint_exists() -> bool:
    """Test that the legacy /process endpoint exists (without actually processing)."""
    try:
        # Send a request that will fail validation but prove the endpoint exists
        response = httpx.post(
            f"{BASE_URL}/process",
            json={"invalid": "data"},
            timeout=10.0
        )
        # We expect 422 (validation error) which means endpoint exists
        if response.status_code in [422, 404]:
            if response.status_code == 422:
                print_result("Legacy /process Endpoint", True, "Endpoint exists (validation failed as expected)")
                return True
            else:
                print_result("Legacy /process Endpoint", False, "Endpoint not found (404)")
                return False
        elif response.status_code == 200:
            print_result("Legacy /process Endpoint", True, "Endpoint exists and processed request")
            return True
        else:
            print_result("Legacy /process Endpoint", False, f"Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Legacy /process Endpoint", False, f"Error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Unified API Migration")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print()
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("API Documentation", test_api_docs),
        ("Wav2Lip Health", test_wav2lip_health),
        ("Characters Endpoint", test_characters_endpoint),
        ("Legacy /process Endpoint", test_legacy_process_endpoint_exists),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print_result(name, False, f"Unexpected error: {str(e)}")
            results.append(False)
        print()
    
    # Summary
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"Test Summary: {passed}/{total} passed ({success_rate:.1f}%)")
    print("=" * 70)
    
    if passed == total:
        print("✅ All tests passed! Migration successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

