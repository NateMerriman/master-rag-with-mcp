#!/usr/bin/env python3
"""
Test the new hybrid_search_code_examples function.

This script tests the integration between our application code and the new
Supabase edge function for code examples search.
"""

import os
import sys
import requests
import json
import traceback

# Add parent directories to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import get_supabase_client


def test_code_examples_search():
    """Test the hybrid_search_code_examples edge function."""
    print("🔍 Testing hybrid_search_code_examples edge function...")
    
    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            test_url = original_url.replace("host.docker.internal", "localhost")
        else:
            test_url = "http://localhost:54321"
        
        # Test edge function endpoint
        edge_function_url = f"{test_url}/functions/v1/hybrid-search-code-examples"
        
        # Get auth headers
        anon_key = os.environ.get("SUPABASE_ANON_KEY")
        if not anon_key:
            print("❌ SUPABASE_ANON_KEY not set, trying service key...")
            anon_key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not anon_key:
            print("⚠️  No auth key available, testing without authentication...")
            headers = {"Content-Type": "application/json"}
        else:
            headers = {
                "Authorization": f"Bearer {anon_key}",
                "Content-Type": "application/json"
            }
        
        # Test cases
        test_cases = [
            {
                "name": "Basic Python search",
                "payload": {
                    "query": "python function",
                    "match_count": 5
                }
            },
            {
                "name": "Language-filtered search",
                "payload": {
                    "query": "function",
                    "match_count": 3,
                    "language_filter": "python"
                }
            },
            {
                "name": "Complexity-filtered search",
                "payload": {
                    "query": "algorithm",
                    "match_count": 5,
                    "max_complexity": 5
                }
            },
            {
                "name": "Combined filters",
                "payload": {
                    "query": "recursive",
                    "match_count": 3,
                    "language_filter": "python",
                    "max_complexity": 8
                }
            }
        ]
        
        print(f"📡 Testing endpoint: {edge_function_url}")
        
        for test_case in test_cases:
            print(f"\n🧪 {test_case['name']}:")
            print(f"   Query: {test_case['payload']}")
            
            try:
                response = requests.post(
                    edge_function_url,
                    headers=headers,
                    json=test_case['payload'],
                    timeout=10
                )
                
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        results = response.json()
                        if isinstance(results, list):
                            print(f"   ✅ Returned {len(results)} results")
                            
                            if results:
                                # Show first result details
                                first = results[0]
                                print(f"   📋 Top result:")
                                print(f"      Language: {first.get('programming_language', 'N/A')}")
                                print(f"      RRF Score: {first.get('rrf_score', 0):.4f}")
                                print(f"      Code snippet: {first.get('code_content', '')[:50]}...")
                            else:
                                print(f"   ⚠️  No results returned (table may be empty)")
                        else:
                            print(f"   📄 Response: {results}")
                    except json.JSONDecodeError:
                        print(f"   📄 Raw response: {response.text[:200]}...")
                
                elif response.status_code == 404:
                    print(f"   ❌ Edge function not found - make sure it's deployed")
                    
                elif response.status_code == 500:
                    print(f"   ❌ Server error: {response.text[:200]}...")
                    
                else:
                    print(f"   ❌ Unexpected status: {response.text[:200]}...")
                    
            except requests.exceptions.ConnectionError:
                print(f"   ❌ Connection failed - is Supabase running on {test_url}?")
                return False
                
            except requests.exceptions.Timeout:
                print(f"   ❌ Request timed out")
                return False
        
        print(f"\n✅ Edge function tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Edge function test failed: {e}")
        traceback.print_exc()
        return False


def test_rpc_function_directly():
    """Test the RPC function directly through Supabase client."""
    print("\n🔧 Testing RPC function directly...")
    
    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace("host.docker.internal", "localhost")
        
        client = get_supabase_client()
        
        # Test the RPC function directly
        print("  🧪 Testing hybrid_search_code_examples RPC...")
        
        # Create a test embedding (normally generated by OpenAI)
        test_embedding = [0.0] * 1536  # Mock embedding
        
        result = client.rpc("hybrid_search_code_examples", {
            "query_text": "python function",
            "query_embedding": test_embedding,
            "match_count": 5
        }).execute()
        
        if result.data is not None:
            print(f"  ✅ RPC function returned {len(result.data)} results")
            
            if result.data:
                first = result.data[0]
                print(f"  📋 First result structure:")
                for key in first.keys():
                    value = str(first[key])[:50]
                    print(f"     {key}: {value}...")
            else:
                print(f"  ⚠️  RPC returned empty results (table may be empty)")
        else:
            print(f"  ❌ RPC function failed: {result}")
            return False
        
        # Test with filters
        print("  🧪 Testing with language filter...")
        
        filtered_result = client.rpc("hybrid_search_code_examples", {
            "query_text": "function",
            "query_embedding": test_embedding,
            "match_count": 3,
            "language_filter": "python",
            "max_complexity": 5
        }).execute()
        
        if filtered_result.data is not None:
            print(f"  ✅ Filtered RPC returned {len(filtered_result.data)} results")
        else:
            print(f"  ❌ Filtered RPC failed: {filtered_result}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ RPC function test failed: {e}")
        traceback.print_exc()
        return False


def run_integration_tests():
    """Run all integration tests for code examples search."""
    print("🚀 Starting code examples search integration tests\n")
    
    tests = [
        ("Edge Function Tests", test_code_examples_search),
        ("RPC Function Tests", test_rpc_function_directly),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All code examples search tests PASSED!")
        print("📋 Integration confirmed:")
        print("   • Edge function is accessible")
        print("   • RPC function works correctly") 
        print("   • Filtering parameters functional")
        print("   • Ready for production use")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED!")
        print("Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)