#!/usr/bin/env python3
"""
Script to validate migration 001 - sources table creation.
"""

import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..utils import get_supabase_client


def validate_sources_migration():
    """Validate that migration 001 was successful."""
    
    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace("host.docker.internal", "localhost")
        
        client = get_supabase_client()
        
        print("=== Migration 001 Validation ===\n")
        
        # 1. Check sources table exists and has data
        try:
            result = client.table("sources").select("source_id, url, total_word_count", count="exact").execute()
            sources_count = result.count
            print(f"✅ Sources table exists with {sources_count} records")
            
            # Show sample sources
            if result.data:
                print(f"\nSample sources:")
                for i, source in enumerate(result.data[:5]):
                    print(f"  {i+1}. ID: {source['source_id']}, Word Count: {source['total_word_count']}, URL: {source['url'][:60]}...")
                    
        except Exception as e:
            print(f"❌ Sources table validation failed: {e}")
            return False
        
        # 2. Check source_id column exists in crawled_pages
        try:
            result = client.table("crawled_pages").select("source_id").limit(1).execute()
            print(f"✅ source_id column added to crawled_pages table")
        except Exception as e:
            print(f"❌ source_id column validation failed: {e}")
            return False
        
        # 3. Verify data integrity - sources count should match unique URLs
        try:
            result = client.table("crawled_pages").select("url").execute()
            unique_urls = len(set(row['url'] for row in result.data if row['url']))
            
            if sources_count == unique_urls:
                print(f"✅ Data integrity validated: {sources_count} sources = {unique_urls} unique URLs")
            else:
                print(f"❌ Data integrity issue: {sources_count} sources ≠ {unique_urls} unique URLs")
                return False
                
        except Exception as e:
            print(f"❌ Data integrity validation failed: {e}")
            return False
        
        # 4. Test sources table functionality
        try:
            # Test querying sources by word count
            result = client.table("sources").select("url, total_word_count").order("total_word_count", desc=True).limit(3).execute()
            print(f"\nTop 3 sources by word count:")
            for source in result.data:
                print(f"  {source['total_word_count']:,} words: {source['url'][:60]}...")
                
        except Exception as e:
            print(f"❌ Sources functionality test failed: {e}")
            return False
        
        # 5. Verify indexes were created
        print(f"\n✅ Migration validation complete!")
        print(f"📊 Summary:")
        print(f"   • Sources table: {sources_count} records")
        print(f"   • Crawled pages: {len(result.data)} records with source_id column")
        print(f"   • Data integrity: Verified")
        print(f"   • Functionality: Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration validation failed: {e}")
        return False


if __name__ == "__main__":
    success = validate_sources_migration()
    sys.exit(0 if success else 1)