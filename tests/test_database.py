import pytest
import os
from src.utils.database import DatabaseManager

def test_database_initialization(tmp_path):
    db_file = tmp_path / "test_init.db"
    db_manager = DatabaseManager(db_path=str(db_file))
    
    assert db_file.exists()
    assert db_manager.get_all_sessions() == []

def test_add_user(tmp_path):
    db_file = tmp_path / "test_user.db"
    db_manager = DatabaseManager(db_path=str(db_file))
    
    user_id = db_manager.add_user("test_user")
    assert user_id == 1
    
    # Re-adding the same user should return the same ID
    user_id_again = db_manager.add_user("test_user")
    assert user_id_again == 1

def test_session_lifecycle(tmp_path):
    db_file = tmp_path / "test_session.db"
    db_manager = DatabaseManager(db_path=str(db_file))
    
    user_id = db_manager.add_user("runner")
    
    # Start session
    session_id = db_manager.start_session(
        user_id=user_id,
        log_file_path="logs/session_1.csv",
        landmark_file_path="data/session_1.npy"
    )
    assert session_id == 1
    
    # End session
    success = db_manager.end_session(
        session_id=session_id,
        avg_stress=0.25,
        peak_stress=0.68,
        max_level="Slight Stress",
        total_frames=1500,
        notes="All good"
    )
    assert success is True
    
    # Retrieve sessions
    sessions = db_manager.get_all_sessions()
    assert len(sessions) == 1
    assert sessions[0]["username"] == "runner"
    assert sessions[0]["avg_stress_score"] == 0.25
    assert sessions[0]["total_frames"] == 1500
