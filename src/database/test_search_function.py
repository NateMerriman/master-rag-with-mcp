#!/usr/bin/env python3
"""
Test that the hybrid_search_crawled_pages function still works after migration.
"""

import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import get_supabase_client, search_documents


def test_hybrid_search():
    """Test that hybrid search still works after migration."""
    
    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace("host.docker.internal", "localhost")
        
        client = get_supabase_client()
        
        print("=== Hybrid Search Function Test ===\n")
        
        # Test 1: Simple search
        print("🔍 Test 1: Simple search for 'Claude'")
        results = search_documents(client, "Claude", match_count=5)
        
        if results:
            print(f"✅ Search returned {len(results)} results")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. RRF Score: {result.get('rrf_score', 0):.4f}")
                print(f"      URL: {result.get('url', 'Unknown')[:60]}...")
                print(f"      Content snippet: {result.get('content', '')[:100]}...")
                print()
        else:
            print("❌ Search returned no results")
            return False
        
        # Test 2: Technical search
        print("🔍 Test 2: Technical search for 'API integration'")
        results = search_documents(client, "API integration", match_count=3)
        
        if results:
            print(f"✅ Search returned {len(results)} results")
            print(f"   Best match RRF score: {results[0].get('rrf_score', 0):.4f}")
        else:
            print("❌ Technical search returned no results")
            return False
        
        # Test 3: Verify RRF components are working
        print("🔍 Test 3: Verify RRF scoring components")
        if results:
            result = results[0]
            has_rrf_score = 'rrf_score' in result
            has_semantic_rank = 'semantic_rank' in result
            has_full_text_rank = 'full_text_rank' in result
            
            print(f"   RRF Score present: {'✅' if has_rrf_score else '❌'}")
            print(f"   Semantic rank present: {'✅' if has_semantic_rank else '❌'}")
            print(f"   Full-text rank present: {'✅' if has_full_text_rank else '❌'}")
            
            if has_rrf_score and has_semantic_rank and has_full_text_rank:
                print(f"✅ All RRF components working correctly")
            else:
                print(f"❌ Some RRF components missing")
                return False
        
        print(f"\n🎉 All hybrid search tests passed!")
        print(f"📊 Summary:")
        print(f"   • Edge function connectivity: Working")
        print(f"   • Hybrid search RPC function: Working") 
        print(f"   • RRF scoring: Working")
        print(f"   • Schema changes: No impact on search functionality")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hybrid_search()
    sys.exit(0 if success else 1)