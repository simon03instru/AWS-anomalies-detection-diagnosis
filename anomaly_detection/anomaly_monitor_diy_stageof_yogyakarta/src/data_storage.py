# data_storage.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os
import json

class WeatherDataStorage:
    """Handles persistent storage of incoming weather data"""
    
    def __init__(self, db_path: str = "weather_data.db", csv_path: str = "weather_data.csv"):
        self.db_path = db_path
        self.csv_path = csv_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with weather data table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table for weather data (NO raw_data column)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tt REAL,
                    rh REAL,
                    pp REAL,
                    ws REAL,
                    wd REAL,
                    sr REAL,
                    rr REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Check if id column exists (for migration from old schema)
            cursor.execute("PRAGMA table_info(weather_data)")
            columns = {row[1] for row in cursor.fetchall()}
            
            if 'id' not in columns:
                self.logger.warning("Migrating old table schema - adding id column...")
                try:
                    # Create new table with id column
                    cursor.execute('''
                        CREATE TABLE weather_data_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            tt REAL,
                            rh REAL,
                            pp REAL,
                            ws REAL,
                            wd REAL,
                            sr REAL,
                            rr REAL,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Copy data from old table
                    cursor.execute('''
                        INSERT INTO weather_data_new (timestamp, tt, rh, pp, ws, wd, sr, rr, created_at)
                        SELECT timestamp, tt, rh, pp, ws, wd, sr, rr, COALESCE(created_at, CURRENT_TIMESTAMP)
                        FROM weather_data
                    ''')
                    
                    # Drop old table and rename new one
                    cursor.execute('DROP TABLE weather_data')
                    cursor.execute('ALTER TABLE weather_data_new RENAME TO weather_data')
                    
                    self.logger.info("Migration completed successfully")
                except Exception as e:
                    self.logger.warning(f"Could not migrate old table: {e}. Recreating...")
                    cursor.execute('DROP TABLE IF EXISTS weather_data')
                    cursor.execute('''
                        CREATE TABLE weather_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            tt REAL,
                            rh REAL,
                            pp REAL,
                            ws REAL,
                            wd REAL,
                            sr REAL,
                            rr REAL,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
            
            # Create index on timestamp for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON weather_data(timestamp)
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def save_weather_data(self, data: Dict, timestamp: datetime) -> bool:
        """
        Save weather data to both SQLite and CSV
        
        Args:
            data: Weather data dictionary
            timestamp: Timestamp of the data
            
        Returns:
            Success status
        """
        try:
            # Save to SQLite
            self._save_to_sqlite(data, timestamp)
            
            # Save to CSV
            self._save_to_csv(data, timestamp)
            
            self.logger.debug(f"Saved weather data for {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving weather data: {e}")
            return False
    
    def _save_to_sqlite(self, data: Dict, timestamp: datetime):
        """Save data to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO weather_data (timestamp, tt, rh, pp, ws, wd, sr, rr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(),
            data.get('tt'),
            data.get('rh'),
            data.get('pp'),
            data.get('ws'),
            data.get('wd'),
            data.get('sr'),
            data.get('rr')
        ))
        
        conn.commit()
        conn.close()
    
    def _save_to_csv(self, data: Dict, timestamp: datetime):
        """Save data to CSV file"""
        # Prepare row data (matching CSV columns)
        row_data = {
            'timestamp': timestamp.isoformat(),
            'tt': data.get('tt'),
            'rh': data.get('rh'),
            'pp': data.get('pp'),
            'ws': data.get('ws'),
            'wd': data.get('wd'),
            'sr': data.get('sr'),
            'rr': data.get('rr')
        }
        
        # Create DataFrame
        df = pd.DataFrame([row_data])
        
        # Append to CSV (create if doesn't exist)
        if os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_path, mode='w', header=True, index=False)
    
    def get_recent_data(self, limit: int = 100, source: str = "sqlite") -> pd.DataFrame:
        """
        Get recent weather data
        
        Args:
            limit: Number of recent records to retrieve
            source: Data source ("sqlite" or "csv")
            
        Returns:
            DataFrame with recent weather data
        """
        if source == "sqlite":
            return self._get_from_sqlite(limit)
        else:
            return self._get_from_csv(limit)
    
    
    def _get_from_sqlite(self, limit: int) -> pd.DataFrame:
        """Get data from SQLite database - ordered by insertion order (ID), not timestamp"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, tt, rh, pp, ws, wd, sr, rr
            FROM weather_data
            ORDER BY id DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        if len(df) == 0:
            return df
        
        # Parse timestamps flexibly (handles mixed formats)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=False)
        
        # Reverse order so oldest is first (chronological order)
        df = df.iloc[::-1].reset_index(drop=True)
        
        return df

    
    def _get_from_csv(self, limit: int) -> pd.DataFrame:
        """Get data from CSV file"""
        if not os.path.exists(self.csv_path):
            return pd.DataFrame()
        
        df = pd.read_csv(self.csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get last 'limit' rows
        df = df.tail(limit).reset_index(drop=True)
        
        return df
    
    def get_data_count(self) -> int:
        """Get total number of records in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM weather_data")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove data older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM weather_data WHERE timestamp < ?", 
                (cutoff_date.isoformat(),)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} old records")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")