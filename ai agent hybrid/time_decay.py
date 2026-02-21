"""
Time Decay Algorithms for Memory Relevance
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Any


def calculate_recency_score(timestamp: str, half_life_days: int = 7) -> float:
    """
    Calculate recency score using exponential decay
    
    Args:
        timestamp: ISO format timestamp
        half_life_days: Days until score reaches 0.5
    
    Returns:
        Score between 0 and 1
    """
    try:
        memory_time = datetime.fromisoformat(timestamp)
        now = datetime.now()
        days_ago = (now - memory_time).total_seconds() / 86400  # Convert to days
        
        # Exponential decay: score = 0.5 ^ (days_ago / half_life)
        score = math.pow(0.5, days_ago / half_life_days)
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    except:
        return 0.5  # Default if parsing fails


def calculate_relevance_score(memory: Dict[str, Any], 
                              query_similarity: float,
                              is_shortterm: bool = True) -> float:
    """
    Calculate overall relevance score combining multiple factors
    
    Args:
        memory: Memory object with metadata
        query_similarity: Vector similarity score (0-1)
        is_shortterm: Whether this is from short-term memory
    
    Returns:
        Combined relevance score (0-1)
    """
    # Get timestamp
    timestamp = memory.get('timestamp', datetime.now().isoformat())
    
    # Recency score (different half-life for short vs long term)
    half_life = 7 if is_shortterm else 90
    recency = calculate_recency_score(timestamp, half_life)
    
    # Access count (normalized)
    access_count = memory.get('access_count', 0)
    # Logarithmic normalization (frequent access matters but with diminishing returns)
    access_score = math.log(access_count + 1) / math.log(100)  # Normalize to ~0-1 range
    access_score = min(1.0, access_score)
    
    # Combine scores with weights
    score = (
        query_similarity * 0.4 +    # Similarity is most important
        recency * 0.3 +              # Recent memories matter
        access_score * 0.2 +         # Frequently accessed memories matter
        (0.1 if is_shortterm else 0.05)  # Slight boost for short-term
    )
    
    return max(0.0, min(1.0, score))


def should_decay_to_longterm(memory: Dict[str, Any], 
                             threshold_days: int = 30,
                             min_access_count: int = 3) -> bool:
    """
    Determine if a short-term memory should be moved to long-term
    
    Args:
        memory: Memory object
        threshold_days: Age threshold in days
        min_access_count: Minimum accesses to be considered important
    
    Returns:
        True if should move to long-term
    """
    try:
        timestamp = memory.get('timestamp', datetime.now().isoformat())
        memory_time = datetime.fromisoformat(timestamp)
        days_old = (datetime.now() - memory_time).days
        
        access_count = memory.get('access_count', 0)
        
        # Move to long-term if:
        # 1. Old enough AND has been accessed multiple times (important)
        # 2. OR very old regardless of access
        if days_old >= threshold_days and access_count >= min_access_count:
            return True
        elif days_old >= threshold_days * 2:  # Much older
            return True
        
        return False
    except:
        return False


def should_archive(memory: Dict[str, Any], 
                  relevance_threshold: float = 0.1,
                  max_age_days: int = 365) -> bool:
    """
    Determine if a memory should be archived (forgotten)
    
    Args:
        memory: Memory object
        relevance_threshold: Minimum relevance score to keep
        max_age_days: Maximum age before auto-archiving
    
    Returns:
        True if should be archived
    """
    try:
        # Calculate current relevance
        timestamp = memory.get('timestamp', datetime.now().isoformat())
        recency = calculate_recency_score(timestamp, half_life_days=90)
        
        access_count = memory.get('access_count', 0)
        access_score = math.log(access_count + 1) / math.log(100)
        
        relevance = (recency * 0.6 + access_score * 0.4)
        
        # Check age
        memory_time = datetime.fromisoformat(timestamp)
        days_old = (datetime.now() - memory_time).days
        
        # Archive if:
        # 1. Relevance too low
        # 2. Too old
        # 3. Never accessed and moderately old
        if relevance < relevance_threshold:
            return True
        elif days_old > max_age_days:
            return True
        elif access_count == 0 and days_old > 180:
            return True
        
        return False
    except:
        return False


def update_access_stats(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update access statistics for a memory
    
    Args:
        memory: Memory object
    
    Returns:
        Updated memory object
    """
    memory['access_count'] = memory.get('access_count', 0) + 1
    memory['last_accessed'] = datetime.now().isoformat()
    return memory
