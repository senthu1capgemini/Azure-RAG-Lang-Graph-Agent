import sqlite3
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

SQLITE_DB_PATH ="chat_history.db"

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    "Get SQLite chat history for a session."
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=f"sqlite:///{SQLITE_DB_PATH}"
    )

def clear_session_history(session_id: str = None):
    "Clear chat history for a specific session or all sessions."
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor() # Needed to execute the SQL commands
        if session_id:
            # Clear the session
            cursor.execute("Delete from message_store WHERE session_id = ? ", (session_id,))
            rows_deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            if rows_deleted > 0:
                return f"Cleared {rows_deleted} messages for session: {session_id}"
            return f"No history for session: {session_id}"
        else:
            # Clear all sessions
            cursor.execute("Delete from message_store")
            rows_deleted = cursor.rowcount
            conn.commit()
            conn.close()
            return f"Cleared all chat histories ({rows_deleted} messages)"
    except Exception as e:
        return f"Error clearing history: {str(e)}"

def list_sessions():
    "List all available sessions in the database."
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, COUNT(*) as message_count, 
                   MIN(id) as first_message_id, MAX(id) as last_message_id
            FROM message_store 
            GROUP BY session_id 
            ORDER BY last_message_id DESC
        """)
        
        sessions = cursor.fetchall()
        conn.close()
        
        if not sessions:
            return "No sessions found in database."
        result = "\nAvailable sessions:\n" + "="*50 + "\n"
        for session_id, count, first_id, last_id in sessions:
            result += f"  - {session_id}: {count} messages\n"
        return result
        
    except Exception as e:
        return f"Error listing sessions: {str(e)}"
