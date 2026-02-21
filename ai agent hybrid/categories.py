"""
Category Definitions for Multi-Category Memory System
"""

CATEGORIES = {
    'careers': {
        'keywords': ['job', 'work', 'career', 'company', 'salary', 'interview', 
                    'promotion', 'coworker', 'boss', 'employment', 'profession',
                    'doordash', 'amazon', 'microsoft', 'software engineer'],
        'description': 'Career, work, and professional information'
    },
    'sports': {
        'keywords': ['football', 'basketball', 'tennis', 'athlete', 'soccer',
                    'cricket', 'baseball', 'player', 'team', 'game', 'match',
                    'messi', 'ronaldo', 'sport', 'fitness', 'exercise'],
        'description': 'Sports, athletes, and physical activities'
    },
    'interests': {
        'keywords': ['hobby', 'like', 'enjoy', 'favorite', 'passion', 'love',
                    'preference', 'interest', 'dislike', 'hate'],
        'description': 'Personal interests, hobbies, and preferences'
    },
    'pop_culture': {
        'keywords': ['movie', 'music', 'celebrity', 'show', 'series', 'film',
                    'actor', 'singer', 'band', 'tv', 'entertainment', 'netflix'],
        'description': 'Movies, music, celebrities, and entertainment'
    },
    'personal': {
        'keywords': ['name', 'age', 'location', 'family', 'live', 'born',
                    'parent', 'sibling', 'friend', 'relationship', 'address'],
        'description': 'Personal information and relationships'
    },
    'food': {
        'keywords': ['food', 'restaurant', 'eat', 'drink', 'pizza', 'cuisine',
                    'meal', 'breakfast', 'lunch', 'dinner', 'favorite dish'],
        'description': 'Food preferences and dining'
    },
    'tech': {
        'keywords': ['technology', 'computer', 'software', 'programming', 'code',
                    'python', 'java', 'react', 'aws', 'api', 'database', 'llm'],
        'description': 'Technology and programming'
    },
    'general': {
        'keywords': [],
        'description': 'General information that doesn\'t fit other categories'
    }
}

def get_all_categories():
    """Return list of all category names"""
    return list(CATEGORIES.keys())

def get_category_info(category):
    """Get information about a specific category"""
    return CATEGORIES.get(category, CATEGORIES['general'])

def keyword_match_category(text):
    """Simple keyword-based category matching (fallback)"""
    text_lower = text.lower()
    scores = {}
    
    for category, info in CATEGORIES.items():
        if category == 'general':
            continue
        score = sum(1 for keyword in info['keywords'] if keyword in text_lower)
        if score > 0:
            scores[category] = score
    
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    return 'general'
