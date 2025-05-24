import numpy as np

# Global configurations
RANDOM_STATE = 42
N_JOBS = -1  # Use all available cores

# Features from audio_features.csv to use for modeling
AUDIO_FEATURES = [
    'danceability', 'energy', 'valence', 
    'acousticness', 'tempo'
]

# Data type optimization for CSV loading
DATA_TYPES = {
    'audio': {
        'track_id': 'category',
        'danceability': np.float32,
        'energy': np.float32,
        'valence': np.float32,
        'acousticness': np.float32,
        'tempo': np.float32,
        'duration_ms': np.float32
    },
    'interactions': {
        'spotify_track_uri': 'category',
        'ms_played': np.float32,
        'skipped': np.int8
    }
}