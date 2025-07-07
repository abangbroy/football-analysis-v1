import streamlit as st
import cv2
import numpy as np
from tracker import FootballTracker
import tempfile
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import psutil  # For memory monitoring
import os

# Configuration
st.set_page_config(
    page_title="MY Football AI Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .stVideo {
        border-radius: 10px;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .player-card {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .team-A { background-color: rgba(0, 120, 255, 0.1); border-left: 4px solid #0078FF; }
    .team-B { background-color: rgba(255, 50, 50, 0.1); border-left: 4px solid #FF3232; }
</style>
""", unsafe_allow_html=True)

# Initialize tracker with caching
@st.cache_resource
def get_tracker():
    return FootballTracker()

tracker = get_tracker()

# UI Header
st.title("⚽ Malaysian Football AI Pro")
st.subheader("Advanced Player Tracking & Performance Analytics")

# Processing options
with st.sidebar:
    st.header("Analysis Settings")
    process_mode = st.radio(
        "Processing Mode",
        ["Fast (recommended)", "Full Analysis"],
        index=0
    )
    show_every_n = st.slider("Display every N frames", 1, 5, 2)
    
# Initialize player options
if 'player_options' not in st.session_state:
    st.session_state.player_options = []

# File upload
uploaded_file = st.file_uploader(
    "Upload match video (MP4/AVI)",
    type=["mp4", "avi"],
    key="file_uploader"
)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Bytes to MB

if uploaded_file:
    # Clear previous runs
    tracker.reset()
    
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    
    # Video configuration
    cap = cv2.VideoCapture(tfile.name)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 15 if process_mode == "Fast" else original_fps
    skip_frames = max(1, int(original_fps / target_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = total_frames // skip_frames
    
    # UI Elements
    status_text = st.empty()
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()
    player_stats_container = st.empty()
    
    # Processing loop
    start_time = time.time()
    frame_pos = 0
    last_update = 0
    
    try:
        with tqdm(total=processed_frames, desc="Processing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_pos += 1
                if frame_pos % skip_frames != 0:
                    continue
                    
                # Process frame
                annotated_frame, player_stats = tracker.process_frame(frame)
                
                # Update player selection
                player_ids = sorted(player_stats.keys())
                if player_ids:
                    st.session_state.player_options = player_ids
                
                # Visual update throttling
                current_time = time.time()
                if current_time - last_update > 0.1:  # 10 FPS UI update
                    if tracker.frame_count % show_every_n == 0:
                        video_placeholder.image(
                            annotated_frame,
                            channels="BGR",
                            use_container_width=True,
                            caption=f"Frame: {frame_pos}/{total_frames}"
                        )
                    
                    # Update metrics
                    fps = tracker.frame_count / (current_time - start_time)
                    metrics = {
                        "Frames Processed": f"{tracker.frame_count}/{processed_frames}",
                        "Processing Speed": f"{fps:.1f} FPS",
                        "Players Tracked": len(player_stats),
                        "Memory Usage": f"{get_memory_usage():.1f} MB"
                    }
                    metrics_placeholder.table(pd.DataFrame([metrics]))
                    
                    # Player stats cards
                    if player_stats:
                        player_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
                        for pid, stats in player_stats.items():
                            team = stats.get('team', 0)
                            team_class = f"team-{'A' if team == 0 else 'B'}"
                            player_html += f"""
                            <div class="player-card {team_class}">
                                <b>Player {pid}</b>
                                <div>Distance: {stats['distance']:.1f} px</div>
                                <div>Speed: {stats['speed']:.1f} px/frame</div>
                                <div>Frames: {stats['frames']}</div>
                                <div>Team: {'A' if team == 0 else 'B'}</div>
                            </div>
                            """
                        player_html += "</div>"
                        player_stats_container.markdown(player_html, unsafe_allow_html=True)
                    
                    progress_bar.progress(min(tracker.frame_count/processed_frames, 1.0))
                    last_update = current_time
                
                pbar.update(1)
                
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    finally:
        cap.release()
        tfile.close()
        
    # Final results
    st.success("✅ Analysis Complete!")
    
    # Player statistics
    if tracker.player_stats:
        st.subheader("Player Performance Analysis")
        
        # Distance covered visualization
        st.markdown("### Distance Covered by Players")
        distance_data = {
            f"Player {pid}": stats['distance'] 
            for pid, stats in tracker.player_stats.items()
        }
        st.bar_chart(distance_data)
        
        # Player trajectories
        st.markdown("### Player Movement Patterns")
        fig = go.Figure()
        
        for pid, trail in tracker.track_history.items():
            if len(trail) > 10:  # Only show players with significant movement
                team = tracker.player_stats[pid].get('team', 0)
                color = "blue" if team == 0 else "red"
                
                x, y = zip(*trail)
                fig.add_trace(go.Scatter(
                    x=x, 
                    y=y,
                    mode='lines',
                    name=f"Player {pid}",
                    line=dict(color=color, width=2),
                    hoverinfo='name'
                ))
        
        fig.update_layout(
            title='Player Trajectories',
            xaxis_title='Field Position (X)',
            yaxis_title='Field Position (Y)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Team analysis
        if tracker.team_colors:
            st.subheader("Team Performance")
            team_a_players = [pid for pid, stats in tracker.player_stats.items() if stats.get('team') == 0]
            team_b_players = [pid for pid, stats in tracker.player_stats.items() if stats.get('team') == 1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Team A**")
                st.metric("Players", len(team_a_players))
                st.metric("Total Distance", f"{sum(tracker.player_stats[pid]['distance'] for pid in team_a_players):.1f} px")
                
            with col2:
                st.markdown("**Team B**")
                st.metric("Players", len(team_b_players))
                st.metric("Total Distance", f"{sum(tracker.player_stats[pid]['distance'] for pid in team_b_players):.1f} px")

# Run with: streamlit run app.py