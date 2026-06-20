import sqlite3
import os
from datetime import datetime
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("database")

class DatabaseManager:
    """
    Handles local database operations using SQLite for storing session summaries and user metadata.
    """
    def __init__(self, db_path="data/app.db"):
        self.db_path = db_path
        # Ensure data folder exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        # Enable dictionary row factory for convenient retrieval
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes SQLite tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        avg_stress_score REAL,
                        peak_stress_score REAL,
                        max_stress_level TEXT,
                        total_frames INTEGER,
                        log_file_path TEXT,
                        landmark_file_path TEXT,
                        notes TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                conn.commit()
                logger.info(f"SQLite database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")

    def add_user(self, username) -> int:
        """Adds a user and returns their ID. If user exists, returns their current ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                    row = cursor.fetchone()
                    return row["id"] if row else None
            except Exception as e:
                logger.error(f"Error fetching user ID for {username}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error adding user {username}: {e}")
            return None

    def start_session(self, user_id=None, log_file_path=None, landmark_file_path=None) -> int:
        """Creates a session record with start time and returns the session ID."""
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (user_id, start_time, log_file_path, landmark_file_path)
                    VALUES (?, ?, ?, ?)
                """, (user_id, start_time, log_file_path, landmark_file_path))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error starting database session: {e}")
            return None

    def end_session(self, session_id, avg_stress, peak_stress, max_level, total_frames, notes=None) -> bool:
        """Updates the session record with final metrics at completion."""
        if session_id is None:
            return False
            
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions
                    SET end_time = ?,
                        avg_stress_score = ?,
                        peak_stress_score = ?,
                        max_stress_level = ?,
                        total_frames = ?,
                        notes = ?
                    WHERE id = ?
                """, (end_time, avg_stress, peak_stress, max_level, total_frames, notes, session_id))
                conn.commit()
                logger.info(f"Database session {session_id} updated successfully.")
                return True
        except Exception as e:
            logger.error(f"Error updating database session {session_id}: {e}")
            return False

    def get_all_sessions(self):
        """Retrieves all logged session histories."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.*, u.username
                    FROM sessions s
                    LEFT JOIN users u ON s.user_id = u.id
                    ORDER BY s.start_time DESC
                """)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching sessions: {e}")
            return []
