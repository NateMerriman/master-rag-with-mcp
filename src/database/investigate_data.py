#!/usr/bin/env python3
"""
Script to investigate the data discrepancy between sources and crawled_pages.
"""

import os
import sys
from collections import Counter

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..utils import get_supabase_client


def investigate_data_discrepancy():
    """Investigate why we have 729 sources vs 40 expected unique URLs."""
    
    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace("host.docker.internal", "localhost")
        
        client = get_supabase_client()
        
        print("=== Data Discrepancy Investigation ===\n")
        
        # 1. Get actual crawled_pages count and unique URLs
        print("📊 Crawled Pages Analysis:")
        result = client.table("crawled_pages").select("id", count="exact").execute()
        total_pages = result.count
        print(f"   Total crawled_pages records: {total_pages:,}")
        
        # Get all URLs from crawled_pages (with pagination to handle large datasets)
        all_urls = []
        offset = 0
        batch_size = 1000
        
        while True:
            result = client.table("crawled_pages").select("url").range(offset, offset + batch_size - 1).execute()
            if not result.data:
                break
            
            batch_urls = [row['url'] for row in result.data if row['url']]
            all_urls.extend(batch_urls)
            offset += batch_size
            
            print(f"   Fetched {len(all_urls):,} URLs so far...")
            
            # Break if we got fewer than batch_size (last batch)
            if len(result.data) < batch_size:
                break
        
        unique_urls_from_pages = len(set(all_urls))
        print(f"   Unique URLs in crawled_pages: {unique_urls_from_pages:,}")
        
        # 2. Get sources count and sample
        print(f"\n📊 Sources Analysis:")
        result = client.table("sources").select("source_id, url", count="exact").execute()
        sources_count = result.count
        print(f"   Total sources records: {sources_count:,}")
        
        # Get some sample sources
        result = client.table("sources").select("url").limit(10).execute()
        print(f"   Sample source URLs:")
        for i, source in enumerate(result.data):
            print(f"     {i+1}. {source['url']}")
        
        # 3. Check for duplicates or data issues
        print(f"\n🔍 Data Quality Check:")
        
        # Check if there are NULL or empty URLs in crawled_pages
        result = client.table("crawled_pages").select("id", count="exact").is_("url", "null").execute()
        null_urls = result.count
        
        result = client.table("crawled_pages").select("id", count="exact").eq("url", "").execute()
        empty_urls = result.count
        
        print(f"   NULL URLs in crawled_pages: {null_urls:,}")
        print(f"   Empty URLs in crawled_pages: {empty_urls:,}")
        
        # Check for duplicate URLs in sources table
        result = client.table("sources").select("url").execute()
        source_urls = [row['url'] for row in result.data]
        url_counts = Counter(source_urls)
        duplicates = {url: count for url, count in url_counts.items() if count > 1}
        
        if duplicates:
            print(f"   Duplicate URLs in sources: {len(duplicates)}")
            for url, count in list(duplicates.items())[:5]:
                print(f"     {count}x: {url}")
        else:
            print(f"   No duplicate URLs in sources table")
        
        # 4. Summary and recommendation
        print(f"\n📋 Summary:")
        print(f"   • Crawled pages: {total_pages:,} records")
        print(f"   • Unique URLs in crawled_pages: {unique_urls_from_pages:,}")
        print(f"   • Sources created: {sources_count:,}")
        print(f"   • NULL/empty URLs: {null_urls + empty_urls:,}")
        
        if sources_count == unique_urls_from_pages:
            print(f"   ✅ Migration worked correctly - you have more data than initially analyzed!")
            print(f"   ✅ The difference suggests your database has grown since the initial analysis")
        else:
            print(f"   ⚠️  There may be a data issue to investigate further")
        
        return sources_count == unique_urls_from_pages
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        return False


if __name__ == "__main__":
    success = investigate_data_discrepancy()
    sys.exit(0 if success else 1)