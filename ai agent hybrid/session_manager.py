"""
Session Manager - Handle temporal conversation storage
Each chat session has isolated temporal memory
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional


class ChatSession:
    """Represents a single chat session with temporal storage"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.started_at = datetime.now().isoformat()
        self.temporal_messages: List[Dict[str, str]] = []
        self.temporal_context: List[str] = []  # Facts extracted during session
        self.is_active = True
    
    def add_message(self, role: str, content: str):
        """Add message to temporal storage"""
        self.temporal_messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_temporal_fact(self, fact: str):
        """Add fact to temporal context (not yet in main memory)"""
        if fact not in self.temporal_context:
            self.temporal_context.append(fact)
    
    def get_context(self) -> str:
        """Get temporal context as string"""
        if not self.temporal_messages:
            return ""
        
        context_parts = ["=== Current Session Context ==="]
        
        # Add recent messages (last 6)
        for msg in self.temporal_messages[-6:]:
            context_parts.append(f"{msg['role']}: {msg['content']}")
        
        # Add temporal facts if any
        if self.temporal_context:
            context_parts.append("\n=== Session Facts (Temporary) ===")
            for fact in self.temporal_context:
                context_parts.append(f"  • {fact}")
        
        return "\n".join(context_parts)
    
    def end_session(self):
        """Mark session as ended"""
        self.is_active = False
        self.ended_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session"""
        return {
            'session_id': self.session_id,
            'started_at': self.started_at,
            'ended_at': getattr(self, 'ended_at', None),
            'is_active': self.is_active,
            'messages': self.temporal_messages,
            'temporal_context': self.temporal_context
        }


class SessionManager:
    """Manage chat sessions and temporal memory"""
    
    def __init__(self):
        self.current_session: Optional[ChatSession] = None
        self.past_sessions: List[ChatSession] = []
    
    def start_new_session(self) -> ChatSession:
        """Start a new chat session"""
        if self.current_session and self.current_session.is_active:
            # End previous session
            self.end_current_session()
        
        self.current_session = ChatSession()
        return self.current_session
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Get current active session"""
        if not self.current_session or not self.current_session.is_active:
            self.start_new_session()
        return self.current_session
    
    def end_current_session(self) -> Optional[ChatSession]:
        """End current session and archive it"""
        if not self.current_session:
            return None
        
        self.current_session.end_session()
        self.past_sessions.append(self.current_session)
        
        ended_session = self.current_session
        self.current_session = None
        
        return ended_session
    
    def get_temporal_context(self) -> str:
        """Get temporal context from current session"""
        if self.current_session:
            return self.current_session.get_context()
        return ""
    
    def add_message_to_session(self, role: str, content: str):
        """Add message to current session"""
        session = self.get_current_session()
        session.add_message(role, content)
    
    def add_temporal_fact(self, fact: str):
        """Add fact to current session's temporal storage"""
        session = self.get_current_session()
        session.add_temporal_fact(fact)
    
    def get_session_facts(self) -> List[str]:
        """Get all temporal facts from current session"""
        if self.current_session:
            return self.current_session.temporal_context
        return []
