#!/usr/bin/env python3
"""
Database migration runner for the MCP Crawl4AI RAG project.
Supports running migrations and rollbacks with proper validation.
"""

import os
import sys
import argparse
from typing import Optional

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import get_supabase_client


def run_sql_file(client, filepath: str, description: str = None) -> bool:
    """
    Execute a SQL file using the Supabase client.
    
    Args:
        client: Supabase client
        filepath: Path to SQL file
        description: Optional description for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(filepath):
            print(f"❌ SQL file not found: {filepath}")
            return False
            
        with open(filepath, 'r') as f:
            sql_content = f.read()
        
        if not sql_content.strip():
            print(f"⚠️  SQL file is empty: {filepath}")
            return False
        
        desc = description or os.path.basename(filepath)
        print(f"📝 Executing {desc}...")
        
        # Execute the SQL using the raw SQL interface
        result = client.rpc('exec_sql', {'sql_command': sql_content}).execute()
        
        print(f"✅ Successfully executed {desc}")
        return True
        
    except Exception as e:
        print(f"❌ Error executing {desc or filepath}: {e}")
        return False


def run_sql_direct(client, sql: str, description: str = "SQL command") -> bool:
    """
    Execute SQL directly using Supabase REST API since psql is not available.
    Breaks down complex SQL into smaller chunks that Supabase can handle.
    
    Args:
        client: Supabase client
        sql: SQL string to execute
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import requests
        import re
        
        print(f"📝 Executing {description}...")
        
        # Get connection details
        supabase_url = os.getenv("SUPABASE_URL", "http://localhost:54321")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_key:
            print("❌ Missing SUPABASE_SERVICE_KEY")
            return False
        
        # Split SQL into individual statements
        # Remove comments and normalize whitespace
        sql_clean = re.sub(r'--.*?\n', '\n', sql)  # Remove line comments
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)  # Remove block comments
        
        # Split by semicolons, but be careful about semicolons in functions
        statements = []
        current_statement = ""
        in_function = False
        
        for line in sql_clean.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Track if we're inside a function definition
            if 'CREATE OR REPLACE FUNCTION' in line.upper() or 'CREATE FUNCTION' in line.upper():
                in_function = True
            elif line.upper().startswith('$$') and in_function:
                in_function = False
            
            current_statement += line + '\n'
            
            # If we hit a semicolon and we're not in a function, this is a complete statement
            if line.endswith(';') and not in_function:
                statement = current_statement.strip()
                if statement and not statement.upper().startswith('BEGIN') and not statement.upper().startswith('COMMIT'):
                    statements.append(statement)
                current_statement = ""
        
        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        # Execute each statement via Supabase REST API
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {supabase_key}',
            'apikey': supabase_key
        }
        
        successful_statements = 0
        for i, statement in enumerate(statements):
            if not statement.strip():
                continue
                
            try:
                # Use the Supabase REST API for raw SQL execution
                url = f"{supabase_url}/rest/v1/rpc/exec_sql"
                payload = {"sql": statement}
                
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code in [200, 201, 204]:
                    successful_statements += 1
                    print(f"  ✅ Statement {i+1}/{len(statements)} executed")
                else:
                    print(f"  ❌ Statement {i+1} failed: {response.status_code} - {response.text}")
                    # For critical statements, we might want to fail fast
                    # For now, continue with other statements
                    
            except Exception as e:
                print(f"  ❌ Statement {i+1} error: {e}")
        
        if successful_statements == len(statements):
            print(f"✅ Successfully executed {description} ({successful_statements}/{len(statements)} statements)")
            return True
        else:
            print(f"⚠️  Partially executed {description} ({successful_statements}/{len(statements)} statements)")
            return False
            
    except Exception as e:
        print(f"❌ Error executing {description}: {e}")
        return False


def run_migration(migration_name: str = "001") -> bool:
    """
    Run a specific migration.
    
    Args:
        migration_name: Name/number of migration to run
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace("host.docker.internal", "localhost")
        
        client = get_supabase_client()
        
        # Construct migration file path
        migrations_dir = os.path.dirname(__file__)
        migration_file = os.path.join(migrations_dir, "migrations", f"{migration_name}_create_sources_table.sql")
        
        if not os.path.exists(migration_file):
            print(f"❌ Migration file not found: {migration_file}")
            return False
        
        # Read and execute migration
        with open(migration_file, 'r') as f:
            sql_content = f.read()
        
        success = run_sql_direct(client, sql_content, f"Migration {migration_name}")
        
        # Restore original URL
        if original_url:
            os.environ["SUPABASE_URL"] = original_url
            
        return success
        
    except Exception as e:
        print(f"❌ Error running migration {migration_name}: {e}")
        return False


def run_rollback(migration_name: str = "001") -> bool:
    """
    Run a rollback for a specific migration.
    
    Args:
        migration_name: Name/number of migration to rollback
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace("host.docker.internal", "localhost")
        
        client = get_supabase_client()
        
        # Construct rollback file path
        migrations_dir = os.path.dirname(__file__)
        rollback_file = os.path.join(migrations_dir, "migrations", f"{migration_name}_rollback_sources_table.sql")
        
        if not os.path.exists(rollback_file):
            print(f"❌ Rollback file not found: {rollback_file}")
            return False
        
        # Read and execute rollback
        with open(rollback_file, 'r') as f:
            sql_content = f.read()
        
        success = run_sql_direct(client, sql_content, f"Rollback {migration_name}")
        
        # Restore original URL
        if original_url:
            os.environ["SUPABASE_URL"] = original_url
            
        return success
        
    except Exception as e:
        print(f"❌ Error running rollback {migration_name}: {e}")
        return False


def validate_migration(migration_name: str = "001") -> bool:
    """
    Validate that a migration was applied successfully.
    
    Args:
        migration_name: Name/number of migration to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace("host.docker.internal", "localhost")
        
        client = get_supabase_client()
        
        if migration_name == "001":
            # Validate sources table exists and has data
            print("🔍 Validating migration 001...")
            
            # Check sources table exists
            result = client.table("sources").select("source_id", count="exact").execute()
            sources_count = result.count
            
            # Check crawled_pages has source_id column
            result = client.table("crawled_pages").select("source_id").limit(1).execute()
            
            # Count unique URLs in crawled_pages
            result = client.table("crawled_pages").select("url").execute()
            unique_urls = len(set(row['url'] for row in result.data if row['url']))
            
            if sources_count == unique_urls and sources_count > 0:
                print(f"✅ Migration 001 validated: {sources_count} sources created, source_id column added")
                return True
            else:
                print(f"❌ Migration 001 validation failed: {sources_count} sources != {unique_urls} unique URLs")
                return False
        
        # Restore original URL
        if original_url:
            os.environ["SUPABASE_URL"] = original_url
            
        return True
        
    except Exception as e:
        print(f"❌ Error validating migration {migration_name}: {e}")
        return False


def main():
    """Main CLI interface for migration runner."""
    parser = argparse.ArgumentParser(description="Database migration runner")
    parser.add_argument("action", choices=["migrate", "rollback", "validate"], 
                       help="Action to perform")
    parser.add_argument("--migration", "-m", default="001", 
                       help="Migration name/number (default: 001)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.action} for migration {args.migration}")
    
    success = False
    if args.action == "migrate":
        success = run_migration(args.migration)
    elif args.action == "rollback":
        success = run_rollback(args.migration)
    elif args.action == "validate":
        success = validate_migration(args.migration)
    
    if success:
        print(f"🎉 {args.action.title()} completed successfully!")
        sys.exit(0)
    else:
        print(f"💥 {args.action.title()} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()