import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from config import *
from time import time

def load_data():
    """Load and merge datasets with proper column handling"""
    audio_df = pd.read_csv('data/audio_features.csv', dtype=DATA_TYPES['audio'])
    interactions_df = pd.read_csv('data/user_interactions.csv', dtype=DATA_TYPES['interactions'])
    
    merged = pd.merge(
        audio_df,
        interactions_df,
        left_on='track_id',
        right_on='spotify_track_uri',
        how='inner',
        suffixes=('_audio', '_interaction')
    )
    
    # Create unified columns
    merged['track_name'] = merged['track_name_audio'].combine_first(merged['track_name_interaction'])
    merged['album_name'] = merged['album_name_audio'].combine_first(merged['album_name_interaction'])
    merged['artist'] = merged['artists'].combine_first(merged['artist_name'])
    
    # Feature engineering
    merged['weight'] = merged['ms_played'] * (1 - merged['skipped'])
    merged['liked'] = (merged['ms_played'] > 30000).astype(np.int8)
    
    # Cleanup and final typing
    return merged.astype({
        'ms_played': np.float32,
        'weight': np.float32,
        'liked': np.int8
    }).drop([
        'track_name_audio', 'track_name_interaction',
        'album_name_audio', 'album_name_interaction',
        'artists', 'artist_name'
    ], axis=1)

def train_and_save_models(X, y):
    """Train models with cross-validation and save artifacts"""
    if not os.path.exists('models'):
        os.makedirs('models')
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')

    models = {
        'logreg': LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            random_state=RANDOM_STATE
        ),
        'randomforest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced_subsample',
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        ),
        'svm': LinearSVC(
            class_weight='balanced',
            dual=False,
            tol=1e-3,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"Training {name}...")
        start = time()
        scores = cross_val_score(
            model, X_scaled, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1 if name == 'svm' else N_JOBS
        )
        model.fit(X_scaled, y)
        joblib.dump(model, f'models/{name}.pkl')
        print(f"{name} trained in {time()-start:.2f}s | CV AUC: {np.mean(scores):.3f}")

if __name__ == "__main__":
    data = load_data()
    X, y = data[AUDIO_FEATURES], data['liked']
    train_and_save_models(X, y)
    joblib.dump(data, 'models/processed_data.pkl')