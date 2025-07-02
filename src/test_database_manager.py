#!/usr/bin/env python3
"""
Test Database Manager for Task 14.7

Provides comprehensive database setup, teardown, and validation infrastructure
for integration and end-to-end testing of the complete pipeline system.

Features:
- Test database creation with Supabase-compatible schema
- Mock Supabase client for testing without external dependencies  
- Data validation and integrity checking
- Performance metrics and storage analytics
- Cleanup and teardown mechanisms
"""

import asyncio
import logging
import sqlite3
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock
import uuid

logger = logging.getLogger(__name__)


@dataclass
class StorageRecord:
    """Represents a record stored in the test database."""
    
    id: Optional[int] = None
    url: str = ""
    chunk_number: int = 0
    content: str = ""
    metadata: Dict[str, Any] = None
    embedding: Optional[bytes] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DatabaseMetrics:
    """Database performance and content metrics."""
    
    total_records: int = 0
    unique_urls: int = 0
    total_chunks: int = 0
    avg_content_length: float = 0.0
    avg_chunk_size: float = 0.0
    storage_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    index_performance_ms: float = 0.0


class MockSupabaseClient:
    """
    Mock Supabase client for testing without external database dependencies.
    
    Simulates key Supabase operations while storing data in SQLite for validation.
    """
    
    def __init__(self, db_connection: sqlite3.Connection):
        """Initialize mock client with database connection."""
        self.connection = db_connection
        self.table_name = "crawled_pages"
        
    def table(self, table_name: str):
        """Mock table method."""
        self.table_name = table_name
        return self
        
    def insert(self, records: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Mock insert operation."""
        return MockInsertQuery(self.connection, self.table_name, records)
        
    def select(self, columns: str = "*"):
        """Mock select operation."""
        return MockSelectQuery(self.connection, self.table_name, columns)
        
    def update(self, data: Dict[str, Any]):
        """Mock update operation.""" 
        return MockUpdateQuery(self.connection, self.table_name, data)
        
    def delete(self):
        """Mock delete operation."""
        return MockDeleteQuery(self.connection, self.table_name)


class MockInsertQuery:
    """Mock Supabase insert query."""
    
    def __init__(self, connection: sqlite3.Connection, table_name: str, records: Union[Dict[str, Any], List[Dict[str, Any]]]):
        self.connection = connection
        self.table_name = table_name
        self.records = records if isinstance(records, list) else [records]
        
    def execute(self):
        """Execute the insert operation."""
        cursor = self.connection.cursor()
        
        for record in self.records:
            # Convert metadata to JSON string if it's a dict
            metadata_json = json.dumps(record.get('metadata', {})) if isinstance(record.get('metadata'), dict) else record.get('metadata', '{}')
            
            cursor.execute("""
                INSERT INTO crawled_pages (url, chunk_number, content, metadata, embedding, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (
                record.get('url', ''),
                record.get('chunk_number', 0),
                record.get('content', ''),
                metadata_json,
                record.get('embedding'),
            ))
        
        self.connection.commit()
        return MockQueryResult(True, f"Inserted {len(self.records)} records")


class MockSelectQuery:
    """Mock Supabase select query."""
    
    def __init__(self, connection: sqlite3.Connection, table_name: str, columns: str):
        self.connection = connection
        self.table_name = table_name
        self.columns = columns
        self.where_conditions = []
        
    def eq(self, column: str, value: Any):
        """Add equality condition."""
        self.where_conditions.append((column, "=", value))
        return self
        
    def execute(self):
        """Execute the select operation."""
        cursor = self.connection.cursor()
        
        query = f"SELECT {self.columns} FROM {self.table_name}"
        params = []
        
        if self.where_conditions:
            where_clause = " AND ".join([f"{col} {op} ?" for col, op, val in self.where_conditions])
            query += f" WHERE {where_clause}"
            params = [val for col, op, val in self.where_conditions]
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        column_names = [description[0] for description in cursor.description]
        data = []
        for row in rows:
            record = dict(zip(column_names, row))
            # Parse metadata JSON
            if 'metadata' in record and record['metadata']:
                try:
                    record['metadata'] = json.loads(record['metadata'])
                except json.JSONDecodeError:
                    record['metadata'] = {}
            data.append(record)
        
        return MockQueryResult(True, "Query successful", data)


class MockUpdateQuery:
    """Mock Supabase update query."""
    
    def __init__(self, connection: sqlite3.Connection, table_name: str, data: Dict[str, Any]):
        self.connection = connection
        self.table_name = table_name
        self.data = data
        self.where_conditions = []
        
    def eq(self, column: str, value: Any):
        """Add equality condition."""
        self.where_conditions.append((column, "=", value))
        return self
        
    def execute(self):
        """Execute the update operation."""
        cursor = self.connection.cursor()
        
        # Convert metadata to JSON if needed
        if 'metadata' in self.data and isinstance(self.data['metadata'], dict):
            self.data['metadata'] = json.dumps(self.data['metadata'])
        
        set_clause = ", ".join([f"{key} = ?" for key in self.data.keys()])
        query = f"UPDATE {self.table_name} SET {set_clause}, updated_at = datetime('now')"
        params = list(self.data.values())
        
        if self.where_conditions:
            where_clause = " AND ".join([f"{col} {op} ?" for col, op, val in self.where_conditions])
            query += f" WHERE {where_clause}"
            params.extend([val for col, op, val in self.where_conditions])
        
        cursor.execute(query, params)
        self.connection.commit()
        
        return MockQueryResult(True, f"Updated {cursor.rowcount} records")


class MockDeleteQuery:
    """Mock Supabase delete query."""
    
    def __init__(self, connection: sqlite3.Connection, table_name: str):
        self.connection = connection
        self.table_name = table_name
        self.where_conditions = []
        
    def eq(self, column: str, value: Any):
        """Add equality condition."""
        self.where_conditions.append((column, "=", value))
        return self
        
    def execute(self):
        """Execute the delete operation."""
        cursor = self.connection.cursor()
        
        query = f"DELETE FROM {self.table_name}"
        params = []
        
        if self.where_conditions:
            where_clause = " AND ".join([f"{col} {op} ?" for col, op, val in self.where_conditions])
            query += f" WHERE {where_clause}"
            params = [val for col, op, val in self.where_conditions]
        
        cursor.execute(query, params)
        self.connection.commit()
        
        return MockQueryResult(True, f"Deleted {cursor.rowcount} records")


class MockQueryResult:
    """Mock query result object."""
    
    def __init__(self, success: bool, message: str, data: Optional[List[Dict[str, Any]]] = None):
        self.success = success
        self.message = message
        self.data = data or []
        
    @property
    def error(self):
        """Return error if query failed."""
        return None if self.success else self.message


class TestDatabaseManager:
    """
    Comprehensive test database manager for integration and E2E testing.
    
    Features:
    - SQLite database with Supabase-compatible schema
    - Mock Supabase client for seamless testing
    - Data validation and integrity checking
    - Performance metrics and analytics
    - Automated cleanup and teardown
    """
    
    def __init__(self, 
                 test_db_path: Optional[str] = None,
                 enable_performance_tracking: bool = True):
        """
        Initialize test database manager.
        
        Args:
            test_db_path: Path to test database file (None for in-memory)
            enable_performance_tracking: Track performance metrics
        """
        self.test_db_path = test_db_path or ":memory:"
        self.enable_performance_tracking = enable_performance_tracking
        self.connection: Optional[sqlite3.Connection] = None
        self.mock_client: Optional[MockSupabaseClient] = None
        self.metrics = DatabaseMetrics()
        self.setup_complete = False
        
    async def setup_test_database(self) -> bool:
        """
        Create and configure test database with complete schema.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info("🔧 Setting up test database infrastructure...")
            start_time = time.time()
            
            # Create database connection
            self.connection = sqlite3.connect(self.test_db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            # Create tables with Supabase-compatible schema
            await self._create_database_schema()
            
            # Create indexes for performance
            await self._create_database_indexes()
            
            # Initialize mock Supabase client
            self.mock_client = MockSupabaseClient(self.connection)
            
            # Validate schema integrity
            schema_valid = await self._validate_schema()
            if not schema_valid:
                logger.error("❌ Database schema validation failed")
                return False
            
            setup_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Test database setup complete in {setup_time:.1f}ms")
            
            self.setup_complete = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Test database setup failed: {str(e)}")
            return False
    
    async def _create_database_schema(self):
        """Create database tables matching Supabase schema."""
        cursor = self.connection.cursor()
        
        # Main crawled_pages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crawled_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,  -- JSON as TEXT for SQLite compatibility
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url, chunk_number)
            )
        """)
        
        # Code examples table for testing code extraction
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                language TEXT NOT NULL,
                code TEXT NOT NULL,
                context TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Test metrics table for performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                operation TEXT NOT NULL,
                execution_time_ms REAL NOT NULL,
                record_count INTEGER DEFAULT 0,
                success BOOLEAN DEFAULT TRUE,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        logger.info("✅ Database schema created")
    
    async def _create_database_indexes(self):
        """Create performance indexes."""
        cursor = self.connection.cursor()
        
        # Indexes for crawled_pages
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crawled_pages_url ON crawled_pages(url)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crawled_pages_chunk_number ON crawled_pages(chunk_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crawled_pages_created_at ON crawled_pages(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crawled_pages_url_chunk ON crawled_pages(url, chunk_number)")
        
        # Indexes for code_examples
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_examples_url ON code_examples(url)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_examples_language ON code_examples(language)")
        
        # Indexes for test_metrics
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_metrics_test_name ON test_metrics(test_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_metrics_operation ON test_metrics(operation)")
        
        self.connection.commit()
        logger.info("✅ Database indexes created")
    
    async def _validate_schema(self) -> bool:
        """Validate database schema integrity."""
        try:
            cursor = self.connection.cursor()
            
            # Check that main tables exist
            required_tables = ['crawled_pages', 'code_examples', 'test_metrics']
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            for table in required_tables:
                if table not in existing_tables:
                    logger.error(f"❌ Required table missing: {table}")
                    return False
            
            # Check crawled_pages table structure
            cursor.execute("PRAGMA table_info(crawled_pages)")
            columns = [row[1] for row in cursor.fetchall()]
            required_columns = ['id', 'url', 'chunk_number', 'content', 'metadata', 'embedding', 'created_at', 'updated_at']
            
            for column in required_columns:
                if column not in columns:
                    logger.error(f"❌ Required column missing in crawled_pages: {column}")
                    return False
            
            logger.info("✅ Database schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema validation error: {str(e)}")
            return False
    
    def get_mock_supabase_client(self):
        """Get mock Supabase client for testing."""
        if not self.setup_complete:
            raise RuntimeError("Database not set up. Call setup_test_database() first.")
        return self.mock_client
    
    async def add_test_documents(self, documents: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Add test documents to the database.
        
        Args:
            documents: List of document records to insert
            
        Returns:
            Tuple of (success, message)
        """
        if not self.setup_complete:
            return False, "Database not set up"
        
        try:
            start_time = time.time()
            
            # Validate document structure
            for doc in documents:
                if not self._validate_document_structure(doc):
                    return False, f"Invalid document structure: {doc}"
            
            # Insert documents using mock client
            result = self.mock_client.table('crawled_pages').insert(documents).execute()
            
            if result.success:
                storage_time = (time.time() - start_time) * 1000
                self.metrics.storage_time_ms += storage_time
                
                # Track performance if enabled
                if self.enable_performance_tracking:
                    await self._record_performance_metric(
                        'document_storage', 'insert', storage_time, len(documents)
                    )
                
                logger.info(f"✅ Stored {len(documents)} documents in {storage_time:.1f}ms")
                return True, f"Successfully stored {len(documents)} documents"
            else:
                return False, result.error or "Storage failed"
                
        except Exception as e:
            logger.error(f"❌ Document storage error: {str(e)}")
            return False, f"Storage error: {str(e)}"
    
    def _validate_document_structure(self, document: Dict[str, Any]) -> bool:
        """Validate document structure matches expected schema."""
        required_fields = ['url', 'chunk_number', 'content']
        
        for field in required_fields:
            if field not in document:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate data types
        if not isinstance(document['url'], str):
            logger.error("URL must be string")
            return False
            
        if not isinstance(document['chunk_number'], int):
            logger.error("Chunk number must be integer")
            return False
            
        if not isinstance(document['content'], str):
            logger.error("Content must be string")
            return False
        
        # Validate metadata if present
        if 'metadata' in document:
            metadata = document['metadata']
            if not isinstance(metadata, (dict, str)):
                logger.error("Metadata must be dict or JSON string")
                return False
        
        return True
    
    async def validate_stored_data(self, expected_count: Optional[int] = None) -> Tuple[bool, DatabaseMetrics]:
        """
        Validate stored data integrity and calculate metrics.
        
        Args:
            expected_count: Expected number of records (optional)
            
        Returns:
            Tuple of (validation_success, metrics)
        """
        if not self.setup_complete:
            return False, self.metrics
        
        try:
            start_time = time.time()
            
            # Get all stored records
            result = self.mock_client.table('crawled_pages').select('*').execute()
            
            if not result.success:
                return False, self.metrics
            
            records = result.data
            retrieval_time = (time.time() - start_time) * 1000
            self.metrics.retrieval_time_ms = retrieval_time
            
            # Calculate metrics
            self.metrics.total_records = len(records)
            self.metrics.unique_urls = len(set(record['url'] for record in records))
            
            if records:
                content_lengths = [len(record['content']) for record in records]
                self.metrics.avg_content_length = sum(content_lengths) / len(content_lengths)
                self.metrics.avg_chunk_size = self.metrics.avg_content_length
                
                # Calculate total chunks
                self.metrics.total_chunks = max((record['chunk_number'] for record in records), default=0)
            
            # Validate data integrity
            validation_success = True
            
            # Check for required fields
            for record in records:
                if not record.get('url') or not record.get('content'):
                    validation_success = False
                    logger.error(f"Invalid record: {record['id']}")
                    
                # Validate metadata JSON
                if record.get('metadata'):
                    try:
                        if isinstance(record['metadata'], str):
                            json.loads(record['metadata'])
                    except json.JSONDecodeError:
                        validation_success = False
                        logger.error(f"Invalid metadata JSON in record: {record['id']}")
            
            # Check expected count if provided
            if expected_count is not None and self.metrics.total_records != expected_count:
                validation_success = False
                logger.error(f"Expected {expected_count} records, found {self.metrics.total_records}")
            
            # Track validation performance
            if self.enable_performance_tracking:
                await self._record_performance_metric(
                    'data_validation', 'select_all', retrieval_time, len(records)
                )
            
            status = "✅" if validation_success else "❌"
            logger.info(f"{status} Data validation complete: {len(records)} records in {retrieval_time:.1f}ms")
            
            return validation_success, self.metrics
            
        except Exception as e:
            logger.error(f"❌ Data validation error: {str(e)}")
            return False, self.metrics
    
    async def _record_performance_metric(self, test_name: str, operation: str, 
                                       execution_time_ms: float, record_count: int = 0):
        """Record performance metric for analysis."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO test_metrics (test_name, operation, execution_time_ms, record_count, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                test_name,
                operation,
                execution_time_ms,
                record_count,
                json.dumps({'timestamp': time.time()})
            ))
            self.connection.commit()
        except Exception as e:
            logger.warning(f"Failed to record performance metric: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.setup_complete:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Get overall metrics
            cursor.execute("""
                SELECT 
                    operation,
                    COUNT(*) as operation_count,
                    AVG(execution_time_ms) as avg_time_ms,
                    MIN(execution_time_ms) as min_time_ms,
                    MAX(execution_time_ms) as max_time_ms,
                    SUM(record_count) as total_records
                FROM test_metrics 
                GROUP BY operation
            """)
            
            operation_metrics = {}
            for row in cursor.fetchall():
                operation_metrics[row[0]] = {
                    'count': row[1],
                    'avg_time_ms': round(row[2], 2),
                    'min_time_ms': round(row[3], 2),
                    'max_time_ms': round(row[4], 2),
                    'total_records': row[5]
                }
            
            return {
                'database_metrics': asdict(self.metrics),
                'operation_metrics': operation_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    async def cleanup_test_data(self):
        """Clean up test data while preserving schema."""
        if not self.setup_complete:
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Clear test data
            cursor.execute("DELETE FROM crawled_pages")
            cursor.execute("DELETE FROM code_examples") 
            cursor.execute("DELETE FROM test_metrics")
            
            self.connection.commit()
            logger.info("✅ Test data cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {str(e)}")
    
    async def teardown_test_database(self):
        """Complete database teardown and cleanup."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                
            self.mock_client = None
            self.setup_complete = False
            
            logger.info("✅ Test database teardown complete")
            
        except Exception as e:
            logger.error(f"❌ Teardown error: {str(e)}")


# Utility functions for testing

async def create_test_database_manager(in_memory: bool = True) -> TestDatabaseManager:
    """Create and setup a test database manager."""
    db_path = ":memory:" if in_memory else None
    manager = TestDatabaseManager(db_path)
    
    success = await manager.setup_test_database()
    if not success:
        raise RuntimeError("Failed to setup test database")
    
    return manager


def create_mock_documents(count: int = 5, base_url: str = "https://example.com") -> List[Dict[str, Any]]:
    """Create mock documents for testing."""
    documents = []
    
    for i in range(count):
        doc = {
            'url': f"{base_url}/page-{i+1}",
            'chunk_number': i + 1,
            'content': f"Mock content for chunk {i+1}. " * 20,  # ~400 chars
            'metadata': {
                'test_run': True,
                'chunk_index': i,
                'quality_score': 0.8 + (i * 0.02),
                'framework': 'mock_framework',
                'extraction_time_ms': 100 + (i * 10)
            }
        }
        documents.append(doc)
    
    return documents


if __name__ == "__main__":
    # Test the database manager
    async def test_database_manager():
        logger.info("Testing database manager...")
        
        manager = await create_test_database_manager()
        
        # Add test documents
        test_docs = create_mock_documents(3)
        success, message = await manager.add_test_documents(test_docs)
        print(f"Storage: {success} - {message}")
        
        # Validate data
        valid, metrics = await manager.validate_stored_data(expected_count=3)
        print(f"Validation: {valid}")
        print(f"Metrics: {metrics}")
        
        # Get performance metrics
        perf_metrics = manager.get_performance_metrics()
        print(f"Performance: {json.dumps(perf_metrics, indent=2)}")
        
        await manager.teardown_test_database()
    
    asyncio.run(test_database_manager())