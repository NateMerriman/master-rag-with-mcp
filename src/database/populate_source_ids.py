#!/usr/bin/env python3
"""
Populate source_id column in crawled_pages table by matching URLs to sources table.
Task 2.3.1: Foreign Key Constraints - Data Population Phase
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from ..utils import get_supabase_client


def run_sql_file(supabase, sql_file_path: str):
    """Execute SQL from file using Supabase raw SQL execution."""
    print(f"\n=== Executing {sql_file_path} ===")
    
    with open(sql_file_path, 'r') as f:
        sql_content = f.read()
    
    try:
        # Execute the SQL file content
        result = supabase.rpc('execute_sql', {'sql': sql_content}).execute()
        print("SQL execution completed successfully")
        return True
    except Exception as e:
        print(f"Error executing SQL file: {e}")
        return False


def validate_before_migration(supabase):
    """Validate database state before running migration."""
    print("\n=== Pre-Migration Validation ===")
    
    try:
        # Check crawled_pages count
        pages_result = supabase.table('crawled_pages').select('id', count='exact').execute()
        pages_count = pages_result.count
        print(f"Total crawled_pages: {pages_count}")
        
        # Check source_id population
        populated_result = supabase.table('crawled_pages').select('id', count='exact').not_.is_('source_id', 'null').execute()
        populated_count = populated_result.count
        print(f"Pages with source_id populated: {populated_count}")
        
        # Check sources count
        sources_result = supabase.table('sources').select('source_id', count='exact').execute()
        sources_count = sources_result.count
        print(f"Total sources: {sources_count}")
        
        # Basic validation
        if pages_count == 0:
            raise ValueError("No crawled_pages found in database")
        if sources_count == 0:
            raise ValueError("No sources found in database")
        if populated_count > 0:
            print(f"WARNING: {populated_count} pages already have source_id populated")
            return False
        
        print("✅ Pre-migration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Pre-migration validation failed: {e}")
        return False


def validate_after_migration(supabase):
    """Validate database state after running migration."""
    print("\n=== Post-Migration Validation ===")
    
    try:
        # Check final state
        pages_result = supabase.table('crawled_pages').select('id', count='exact').execute()
        pages_count = pages_result.count
        
        populated_result = supabase.table('crawled_pages').select('id', count='exact').not_.is_('source_id', 'null').execute()
        populated_count = populated_result.count
        
        unpopulated_count = pages_count - populated_count
        match_percentage = (populated_count / pages_count) * 100 if pages_count > 0 else 0
        
        print(f"Total pages: {pages_count}")
        print(f"Pages with source_id: {populated_count}")
        print(f"Pages without source_id: {unpopulated_count}")
        print(f"Match percentage: {match_percentage:.2f}%")
        
        # Validation criteria
        if match_percentage < 50:
            print("❌ CRITICAL: Very low match rate - migration may have failed")
            return False
        elif match_percentage < 90:
            print("⚠️  WARNING: Lower than expected match rate")
        else:
            print("✅ Good match rate achieved")
        
        return True
        
    except Exception as e:
        print(f"❌ Post-migration validation failed: {e}")
        return False


def main():
    """Main migration execution function."""
    print("=== Source ID Population Migration ===")
    print("Task 2.3.1: Populate source_id column in crawled_pages")
    
    # Override SUPABASE_URL for local testing if needed
    if os.getenv('SUPABASE_URL') == 'http://host.docker.internal:54321':
        print("Overriding SUPABASE_URL for local testing...")
        os.environ['SUPABASE_URL'] = 'http://localhost:54321'
    
    try:
        # Get database client
        supabase = get_supabase_client()
        print("✅ Database connection established")
        
        # Pre-migration validation
        if not validate_before_migration(supabase):
            print("❌ Pre-migration validation failed. Stopping.")
            return False
        
        # Get migration file path
        migrations_dir = Path(__file__).parent / "migrations"
        migration_file = migrations_dir / "003_populate_source_ids.sql"
        
        if not migration_file.exists():
            print(f"❌ Migration file not found: {migration_file}")
            return False
        
        # Ask for confirmation (skip if non-interactive)
        print(f"\nReady to execute migration: {migration_file}")
        try:
            confirmation = input("Continue? (y/N): ").strip().lower()
            if confirmation != 'y':
                print("Migration cancelled by user.")
                return False
        except EOFError:
            # Non-interactive mode, proceed automatically
            print("Non-interactive mode detected, proceeding with migration...")
        
        # Execute migration
        success = run_sql_file(supabase, str(migration_file))
        if not success:
            print("❌ Migration execution failed")
            return False
        
        # Post-migration validation
        if not validate_after_migration(supabase):
            print("❌ Post-migration validation failed")
            return False
        
        print("\n✅ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Review any unmatched URLs shown above")
        print("2. Run foreign key constraint addition (Task 2.3.2)")
        print("3. Test constraint enforcement")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)