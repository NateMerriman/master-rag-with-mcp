#!/usr/bin/env python3
"""
Script to inspect current crawled_pages data for migration planning.
"""

import os
import sys
from collections import Counter
from urllib.parse import urlparse

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..utils import get_supabase_client


def analyze_current_data():
    """Analyze current crawled_pages data to understand migration scope."""
    
    try:
        # Override SUPABASE_URL for local testing (same pattern as performance tests)
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace("host.docker.internal", "localhost")
        
        client = get_supabase_client()
        
        # Get summary statistics
        print("=== Current crawled_pages Data Analysis ===\n")
        
        # Total records count
        result = client.table("crawled_pages").select("id", count="exact").execute()
        total_records = result.count
        print(f"Total records: {total_records:,}")
        
        # Get unique URLs and their chunk counts
        result = client.table("crawled_pages").select("url, chunk_number").execute()
        data = result.data
        
        # Analyze URLs
        url_counter = Counter()
        url_chunk_counts = {}
        
        for row in data:
            url = row['url']
            url_counter[url] += 1
            
            if url not in url_chunk_counts:
                url_chunk_counts[url] = 0
            url_chunk_counts[url] = max(url_chunk_counts[url], row.get('chunk_number', 0))
        
        print(f"Unique URLs: {len(url_counter)}")
        print(f"Average chunks per URL: {total_records / len(url_counter):.1f}")
        
        # Analyze domains
        domain_counter = Counter()
        for url in url_counter.keys():
            try:
                parsed = urlparse(url)
                domain = parsed.netloc
                domain_counter[domain] += url_counter[url]
            except:
                domain_counter['invalid_url'] += url_counter[url]
        
        print(f"\nTop 10 domains by chunk count:")
        for domain, count in domain_counter.most_common(10):
            print(f"  {domain}: {count:,} chunks")
        
        # Show some sample URLs for migration validation
        print(f"\nSample URLs (first 10):")
        for i, (url, count) in enumerate(url_counter.most_common(10)):
            print(f"  {url}: {count} chunks")
        
        # Analyze metadata structure
        result = client.table("crawled_pages").select("metadata").limit(5).execute()
        print(f"\nSample metadata structures:")
        for i, row in enumerate(result.data):
            metadata = row.get('metadata', {})
            keys = list(metadata.keys()) if metadata else []
            print(f"  Record {i+1}: {keys}")
        
        print(f"\n=== Migration Planning Summary ===")
        print(f"• Will create {len(url_counter)} source records")
        print(f"• Will add source_id foreign key to {total_records:,} crawled_pages records")
        print(f"• Largest source has {max(url_counter.values())} chunks")
        print(f"• Migration should be straightforward with URL-based grouping")
        
        return {
            'total_records': total_records,
            'unique_urls': len(url_counter),
            'url_chunk_counts': url_chunk_counts,
            'sample_urls': list(url_counter.keys())[:5],
            'domain_distribution': dict(domain_counter.most_common(5))
        }
        
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None


if __name__ == "__main__":
    analyze_current_data()