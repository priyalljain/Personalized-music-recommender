import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from config import AUDIO_FEATURES
import datetime
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    data = joblib.load('models/processed_data.pkl')
    metrics = pd.read_csv('reports/metrics/performance.csv', index_col=0)
    # Load listening history data
    listening_history = pd.read_csv('data/user_interactions.csv')
    return data, metrics, listening_history

@st.cache_resource
def load_models():
    return {
        'Logistic Regression': joblib.load('models/logreg.pkl'),
        'Random Forest': joblib.load('models/randomforest.pkl'),
        'SVM': joblib.load('models/svm.pkl')
    }

def analyze_listening_patterns(history_data):
    """Analyze user listening patterns to provide insights"""
    # Convert timestamp to datetime
    history_data['ts'] = pd.to_datetime(history_data['ts'])
    
    # Calculate listening duration in minutes
    history_data['minutes_played'] = history_data['ms_played'] / 60000
    
    # Calculate skip ratio
    total_tracks = len(history_data)
    skipped_tracks = history_data['skipped'].sum()
    skip_ratio = skipped_tracks / total_tracks if total_tracks > 0 else 0
    
    # Calculate completion ratio (tracks played more than 70% of their duration)
    merged_data = history_data.merge(
        load_data()[0][['track_name', 'duration_ms']], 
        on='track_name', how='left'
    )
    merged_data['completion_ratio'] = merged_data['ms_played'] / merged_data['duration_ms']
    completion_ratio = (merged_data['completion_ratio'] > 0.7).mean()
    
    # Time of day analysis
    history_data['hour'] = history_data['ts'].dt.hour
    hourly_listening = history_data.groupby('hour')['minutes_played'].sum()
    peak_hour = hourly_listening.idxmax()
    
    # Day of week analysis
    history_data['day_of_week'] = history_data['ts'].dt.dayofweek
    daily_listening = history_data.groupby('day_of_week')['minutes_played'].sum()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    peak_day = day_names[daily_listening.idxmax()]
    
    # Top artists and tracks
    top_artists = history_data['artist_name'].value_counts().head(5).to_dict()
    top_tracks = history_data['track_name'].value_counts().head(5).to_dict()
    
    # Mood analysis based on reasons for ending tracks
    mood_map = {
        'trackdone': 'Satisfied',
        'endplay': 'Neutral',
        'fwdbtn': 'Impatient',
        'backbtn': 'Nostalgic',
        'unexpected': 'Distracted'
    }
    history_data['mood'] = history_data['reason_end'].map(lambda x: mood_map.get(x, 'Unknown'))
    mood_distribution = history_data['mood'].value_counts().to_dict()
    primary_mood = max(mood_distribution.items(), key=lambda x: x[1])[0]
    
    # Create visualizations
    hourly_fig = px.bar(
        x=hourly_listening.index, 
        y=hourly_listening.values,
        labels={'x': 'Hour of Day', 'y': 'Minutes Played'},
        title='Listening Activity by Hour'
    )
    
    daily_fig = px.bar(
        x=[day_names[i] for i in daily_listening.index], 
        y=daily_listening.values,
        labels={'x': 'Day of Week', 'y': 'Minutes Played'},
        title='Listening Activity by Day'
    )
    
    # Calculate "Music DNA" - your unique listening profile
    genre_counts = history_data.merge(
        load_data()[0][['track_name', 'track_genre']], 
        on='track_name', how='left'
    )['track_genre'].value_counts().head(6)
    
    genre_fig = px.pie(
        values=genre_counts.values,
        names=genre_counts.index,
        title='Your Music DNA - Genre Distribution'
    )
    
    # Calculate listening consistency score (0-100)
    days_active = history_data['ts'].dt.date.nunique()
    days_total = (history_data['ts'].max() - history_data['ts'].min()).days + 1
    consistency_score = min(100, int((days_active / max(days_total, 1)) * 100))
    
    return {
        'skip_ratio': skip_ratio,
        'completion_ratio': completion_ratio,
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'top_artists': top_artists,
        'top_tracks': top_tracks,
        'primary_mood': primary_mood,
        'hourly_fig': hourly_fig,
        'daily_fig': daily_fig,
        'genre_fig': genre_fig,
        'consistency_score': consistency_score,
        'total_minutes': history_data['minutes_played'].sum(),
    }

def generate_music_personality(listening_data, track_data):
    """Generate a music personality profile based on listening patterns"""
    # Merge listening data with track audio features
    merged = listening_data.merge(
        track_data[['track_name', 'danceability', 'energy', 'valence', 'tempo', 'acousticness']], 
        on='track_name', how='left'
    )
    
    # Weight the audio features by play count and time spent
    play_counts = listening_data['track_name'].value_counts()
    weighted_merged = merged.copy()
    weighted_merged['weight'] = weighted_merged['track_name'].map(lambda x: play_counts.get(x, 0))
    
    # Calculate weighted averages of audio features
    weighted_features = {}
    for feature in ['danceability', 'energy', 'valence', 'acousticness']:
        weighted_avg = (weighted_merged[feature] * weighted_merged['weight']).sum() / weighted_merged['weight'].sum()
        weighted_features[feature] = weighted_avg
    
    # Determine music personality type based on feature combinations
    personality_type = ""
    
    # Energy and Danceability define the first part of personality
    if weighted_features['energy'] > 0.7 and weighted_features['danceability'] > 0.7:
        personality_type += "Party Animal"
    elif weighted_features['energy'] > 0.7 and weighted_features['danceability'] <= 0.7:
        personality_type += "Energetic Contemplator"
    elif weighted_features['energy'] <= 0.7 and weighted_features['danceability'] > 0.7:
        personality_type += "Smooth Groover"
    else:
        personality_type += "Thoughtful Listener"
    
    # Valence and Acousticness define the second part
    if weighted_features['valence'] > 0.6 and weighted_features['acousticness'] < 0.4:
        personality_type += ", Upbeat Electronic"
    elif weighted_features['valence'] > 0.6 and weighted_features['acousticness'] >= 0.4:
        personality_type += ", Happy Acoustic"
    elif weighted_features['valence'] <= 0.6 and weighted_features['acousticness'] < 0.4:
        personality_type += ", Moody Electronic"
    else:
        personality_type += ", Introspective Acoustic"
    
    # Create radar chart of musical preferences
    categories = ['Danceability', 'Energy', 'Positivity', 'Acousticness']
    values = [
        weighted_features['danceability'],
        weighted_features['energy'],
        weighted_features['valence'],
        weighted_features['acousticness']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Music Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Your Musical Preference Profile"
    )
    
    return {
        'personality_type': personality_type,
        'radar_chart': fig,
        'feature_values': weighted_features
    }

def main():
    st.set_page_config(page_title="Music Recommender", layout="wide")
    data, metrics, listening_history = load_data()
    models = load_models()
    scaler = joblib.load('models/scaler.pkl')

    # Sidebar controls
    st.sidebar.header("Controls")
    track = st.sidebar.selectbox("Select Track", data['track_name'].unique())
    model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))

    # Main interface
    st.title("Music Recommendation System")

    # Generate Recommendations button moved above the Personal Music Analysis section
    if st.sidebar.button("Generate Recommendations"):
        model = models[model_name]
        X = scaler.transform(data[AUDIO_FEATURES])
        
        # Handle different model types
        if hasattr(model, "predict_proba"):
            similarities = model.predict_proba(X)[:,1]
        else:  # For LinearSVC
            similarities = model.decision_function(X)
            similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        
        rec_indices = np.argsort(-similarities)
        rec_df = data.iloc[rec_indices].head(10)
        
        st.subheader("Top Recommendations")
        st.dataframe(
            rec_df[['track_name', 'artist', 'album_name', 'track_genre']],
            height=400
        )

    # Add new feature in sidebar (removed "Musical Twins" option)
    st.sidebar.header("Personal Music Analysis")
    analysis_options = ["Listening Patterns", "Music Personality"]
    selected_analysis = st.sidebar.radio("Choose Analysis", analysis_options)
    
    # New unique feature: Listening Pattern Analysis
    if st.sidebar.button("Analyze My Music"):
        st.header(f"{selected_analysis}")
        
        if selected_analysis == "Listening Patterns":
            with st.spinner("Analyzing your listening patterns..."):
                insights = analyze_listening_patterns(listening_history)
                
                # Create dashboard layout with multiple columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Listening Time", f"{int(insights['total_minutes'])} mins")
                    st.metric("Skip Rate", f"{int(insights['skip_ratio']*100)}%")
                    
                with col2:
                    st.metric("Completion Rate", f"{int(insights['completion_ratio']*100)}%") 
                    st.metric("Consistency Score", f"{insights['consistency_score']}/100")
                    
                with col3:
                    st.metric("Peak Listening Hour", f"{insights['peak_hour']}:00")
                    st.metric("Favorite Day", insights['peak_day'])
                
                st.subheader("Your Primary Listening Mood")
                st.info(f"Based on your listening patterns, you're a **{insights['primary_mood']}** listener")
                
                # Display charts in tabs
                tabs = st.tabs(["Hourly Patterns", "Weekly Patterns", "Music DNA"])
                
                with tabs[0]:
                    st.plotly_chart(insights['hourly_fig'], use_container_width=True)
                
                with tabs[1]:
                    st.plotly_chart(insights['daily_fig'], use_container_width=True)
                    
                with tabs[2]:
                    st.plotly_chart(insights['genre_fig'], use_container_width=True)
                
                # Display top artists and tracks
                st.subheader("Your Top Artists")
                for i, (artist, count) in enumerate(insights['top_artists'].items(), 1):
                    st.write(f"{i}. {artist} ({count} plays)")
                
                st.subheader("Your Top Tracks")
                for i, (track, count) in enumerate(insights['top_tracks'].items(), 1):
                    st.write(f"{i}. {track} ({count} plays)")
        
        elif selected_analysis == "Music Personality":
            with st.spinner("Generating your music personality profile..."):
                personality = generate_music_personality(listening_history, data)
                
                st.subheader("Your Music Personality Type")
                st.info(f"### {personality['personality_type']}")
                
                # Explanation of personality
                personality_explanations = {
                    "Party Animal": "You gravitate toward high-energy, danceable music that keeps the vibe going.",
                    "Energetic Contemplator": "You enjoy energetic music but prefer to experience it rather than dance to it.",
                    "Smooth Groover": "You love rhythmic, groovy tracks but don't need them to be at maximum energy.",
                    "Thoughtful Listener": "You prefer music that allows for contemplation and emotional connection."
                }
                
                # Extract the first part of personality for the explanation
                base_personality = personality['personality_type'].split(',')[0].strip()
                st.write(personality_explanations.get(base_personality, "Your listening style is unique!"))
                
                # Display radar chart
                st.plotly_chart(personality['radar_chart'], use_container_width=True)
                
                # Feature breakdown with explanations
                st.subheader("Your Musical Preferences Breakdown")
                
                feature_cols = st.columns(4)
                
                with feature_cols[0]:
                    dance_val = personality['feature_values']['danceability']
                    st.metric("Danceability", f"{int(dance_val*100)}%")
                    if dance_val > 0.7:
                        st.caption("You prefer rhythmic, danceable tracks")
                    else:
                        st.caption("You enjoy music regardless of its danceability")
                
                with feature_cols[1]:
                    energy_val = personality['feature_values']['energy']
                    st.metric("Energy", f"{int(energy_val*100)}%")
                    if energy_val > 0.7:
                        st.caption("You gravitate toward high-energy music")
                    else:
                        st.caption("You appreciate more relaxed, low-key tracks")
                
                with feature_cols[2]:
                    valence_val = personality['feature_values']['valence']
                    st.metric("Positivity", f"{int(valence_val*100)}%")
                    if valence_val > 0.6:
                        st.caption("You tend to listen to upbeat, positive music")
                    else:
                        st.caption("You connect with more contemplative or melancholic sounds")
                
                with feature_cols[3]:
                    acoustic_val = personality['feature_values']['acousticness']
                    st.metric("Acousticness", f"{int(acoustic_val*100)}%")
                    if acoustic_val > 0.5:
                        st.caption("You prefer organic, acoustic instrumentation")
                    else:
                        st.caption("You enjoy electronic and produced sounds")
                
                # Instead of showing personalized recommendations that were causing errors, 
                # just display a message
                st.subheader("Personalized Recommendations")
                st.info("Explore your music profile above to discover new music that matches your taste.")

    # Performance metrics (original code)
    st.header("Model Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evaluation Metrics")
        st.dataframe(metrics.style.format("{:.2f}"))
        
    with col2:
        st.subheader("ROC Curves")
        st.image('reports/metrics/roc_curves.png')
    
    st.subheader("Feature Importance")
    st.image('reports/metrics/feature_importance.png')

if __name__ == "__main__":
    main()