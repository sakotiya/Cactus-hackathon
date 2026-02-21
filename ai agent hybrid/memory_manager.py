"""
Core Memory Manager - Central memory operations
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from time_decay import (calculate_relevance_score, should_decay_to_longterm,
                       should_archive, update_access_stats)


class MemoryManager:
    """Manage short-term and long-term memory with hybrid storage"""
    
    def __init__(self, data_dir: str = "./data", gemma_model: str = "gemma2:2b"):
        self.data_dir = data_dir
        self.memories_file = os.path.join(data_dir, "memories.json")
        self.archived_dir = os.path.join(data_dir, "archived")
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.archived_dir, exist_ok=True)
        
        # Initialize vector store with Gemma
        self.vector_store = VectorStore(
            persist_directory=os.path.join(data_dir, "chroma_db"),
            gemma_model=gemma_model
        )
        
        # Load memories from JSON
        self.memories = self._load_memories()
    
    def _load_memories(self) -> Dict[str, Dict[str, Any]]:
        """Load memories from JSON file"""
        if os.path.exists(self.memories_file):
            try:
                with open(self.memories_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_memories(self) -> None:
        """Save memories to JSON file"""
        with open(self.memories_file, 'w') as f:
            json.dump(self.memories, f, indent=2)
    
    def add_memory(self, content: str, category: str, 
                  memory_type: str = 'short', 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new memory
        
        Args:
            content: Memory content
            category: Category name
            memory_type: 'short' or 'long'
            metadata: Additional metadata
        
        Returns:
            memory_id
        """
        memory_id = str(uuid.uuid4())
        
        # Create memory object
        memory = {
            'id': memory_id,
            'content': content,
            'category': category,
            'memory_type': memory_type,
            'timestamp': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        
        # Store in JSON
        self.memories[memory_id] = memory
        self._save_memories()
        
        # Store in vector database
        vector_metadata = {
            'category': category,
            'memory_type': memory_type,
            'timestamp': memory['timestamp']
        }
        
        if memory_type == 'short':
            self.vector_store.add_to_shortterm(memory_id, content, vector_metadata)
        else:
            self.vector_store.add_to_longterm(memory_id, content, vector_metadata)
        
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        memory = self.memories.get(memory_id)
        if memory:
            # Update access stats
            memory = update_access_stats(memory)
            self.memories[memory_id] = memory
            self._save_memories()
        return memory
    
    def get_memories_by_category(self, category: str, 
                                 memory_type: str = 'both') -> List[Dict[str, Any]]:
        """Get all memories in a category"""
        results = []
        
        for memory_id, memory in self.memories.items():
            if memory['category'] == category:
                if memory_type == 'both' or memory['memory_type'] == memory_type:
                    results.append(memory)
        
        return results
    
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory's metadata"""
        if memory_id in self.memories:
            self.memories[memory_id].update(updates)
            self._save_memories()
            return True
        return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # Remove from vector store
            if memory['memory_type'] == 'short':
                self.vector_store.delete_from_shortterm(memory_id)
            else:
                self.vector_store.delete_from_longterm(memory_id)
            
            # Remove from JSON
            del self.memories[memory_id]
            self._save_memories()
            return True
        return False
    
    def promote_to_longterm(self, memory_id: str) -> bool:
        """Move memory from short-term to long-term"""
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        if memory['memory_type'] != 'short':
            return False  # Already long-term
        
        # Update memory type
        memory['memory_type'] = 'long'
        memory['promoted_at'] = datetime.now().isoformat()
        self.memories[memory_id] = memory
        self._save_memories()
        
        # Move in vector store
        vector_metadata = {
            'category': memory['category'],
            'memory_type': 'long',
            'timestamp': memory['timestamp']
        }
        self.vector_store.move_to_longterm(memory_id, memory['content'], vector_metadata)
        
        return True
    
    def archive_memory(self, memory_id: str) -> bool:
        """Archive (forget) a memory"""
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        
        # Save to archived folder
        archived_file = os.path.join(self.archived_dir, f"{memory_id}.json")
        with open(archived_file, 'w') as f:
            json.dump(memory, f, indent=2)
        
        # Delete from active memory
        return self.delete_memory(memory_id)
    
    def run_decay_process(self) -> Dict[str, int]:
        """
        Run time decay process
        
        Returns:
            Statistics about what was processed
        """
        stats = {
            'promoted': 0,
            'archived': 0,
            'updated': 0
        }
        
        memory_ids = list(self.memories.keys())
        
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
            
            # Check if should archive
            if should_archive(memory):
                if self.archive_memory(memory_id):
                    stats['archived'] += 1
                continue
            
            # Check if short-term should be promoted
            if memory['memory_type'] == 'short':
                if should_decay_to_longterm(memory):
                    if self.promote_to_longterm(memory_id):
                        stats['promoted'] += 1
            
            stats['updated'] += 1
        
        return stats
    
    def search_memories(self, query: str, category: Optional[str] = None,
                       memory_type: str = 'both', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories using vector similarity
        
        Args:
            query: Search query
            category: Optional category filter
            memory_type: 'short', 'long', or 'both'
            top_k: Number of results
        
        Returns:
            List of memories with relevance scores
        """
        results = []
        
        # Search short-term
        if memory_type in ['short', 'both']:
            short_results = self.vector_store.search_shortterm(query, category, top_k)
            for result in short_results:
                memory_id = result['id']
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    memory['similarity'] = result['similarity']
                    memory['relevance_score'] = calculate_relevance_score(
                        memory, result['similarity'], is_shortterm=True
                    )
                    results.append(memory)
        
        # Search long-term
        if memory_type in ['long', 'both']:
            long_results = self.vector_store.search_longterm(query, category, top_k)
            for result in long_results:
                memory_id = result['id']
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    memory['similarity'] = result['similarity']
                    memory['relevance_score'] = calculate_relevance_score(
                        memory, result['similarity'], is_shortterm=False
                    )
                    results.append(memory)
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update access stats for retrieved memories
        for memory in results[:top_k]:
            self.update_memory(memory['id'], {
                'access_count': memory['access_count'] + 1,
                'last_accessed': datetime.now().isoformat()
            })
        
        return results[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            'total': len(self.memories),
            'shortterm': 0,
            'longterm': 0,
            'by_category': {}
        }
        
        for memory in self.memories.values():
            if memory['memory_type'] == 'short':
                stats['shortterm'] += 1
            else:
                stats['longterm'] += 1
            
            category = memory['category']
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
        
        return stats
    
    def forget_pattern(self, pattern: str) -> List[str]:
        """Archive all memories matching a pattern"""
        pattern_lower = pattern.lower()
        archived = []
        
        memory_ids = list(self.memories.keys())
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if memory and pattern_lower in memory['content'].lower():
                if self.archive_memory(memory_id):
                    archived.append(memory['content'])
        
        return archived
