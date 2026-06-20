import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

from src.inference.face_landmark_detector import FaceLandmarkDetector
from src.inference.feature_engineering import FeatureEngineer
from src.inference.stress_model import StressModel
from src.utils.signal_processing import SignalSmoother

# Page configuration
st.set_page_config(
    page_title="AI Micro-Expression & Stress Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .calm-text { color: #4CAF50; font-weight: bold; }
    .slight-text { color: #FFEB3B; font-weight: bold; }
    .high-text { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔬 Real-Time Micro-Expression & Stress Analyzer</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Live Camera Analyzer", "Image File Analyzer", "Session Logs Viewer"])

# Initialize models cached so they don't load on every streamlit rerun
@st.cache_resource
def load_analysis_models():
    detector = FaceLandmarkDetector()
    feature_engineer = FeatureEngineer()
    stress_model = StressModel(window_size=15)
    return detector, feature_engineer, stress_model

try:
    detector, feature_engineer, stress_model = load_analysis_models()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load models: {e}")
    model_loaded = False

if model_loaded:
    if app_mode == "Live Camera Analyzer":
        st.subheader("📹 Live Camera Stress Analyzer")
        st.write("This mode captures frames from your local webcam and processes them in real-time.")

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            run_live = st.checkbox("Start Live Webcam", value=False)
        with col2:
            smoothing_alpha = st.slider("Smoothing Factor (alpha)", 0.05, 1.0, 0.3, 0.05)

        # Output canvas and metrics side-by-side
        video_col, metrics_col = st.columns([2, 1])

        with video_col:
            st.write("Webcam Feed")
            image_placeholder = st.empty()
            
        with metrics_col:
            st.write("Real-Time Stress Metrics")
            stress_level_placeholder = st.empty()
            stress_score_placeholder = st.empty()
            feature_chart_placeholder = st.empty()

        # History for live plotting
        if 'live_score_history' not in st.session_state:
            st.session_state.live_score_history = []

        plot_placeholder = st.empty()

        if run_live:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            smoother = SignalSmoother(alpha=smoothing_alpha)
            prev_landmarks = None
            start_time = time.time()
            
            # Read loop
            while run_live:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from webcam. Ensure your camera is available.")
                    break
                    
                # Mirror frame
                frame = cv2.flip(frame, 1)
                timestamp_ms = int((time.time() - start_time) * 1000)
                
                # Inference
                landmarks, result = detector.extract_landmarks(frame, timestamp_ms)
                
                if landmarks is not None:
                    # Feature Extraction & Smoothing
                    raw_features = feature_engineer.get_features(landmarks, prev_landmarks)
                    features = smoother.smooth_features(raw_features)
                    
                    # Predict Stress
                    stress_info = stress_model.predict(features)
                    
                    # Update State
                    prev_landmarks = landmarks
                    st.session_state.live_score_history.append(stress_info["stress_score"])
                    # Keep last 100 entries for plotting
                    if len(st.session_state.live_score_history) > 100:
                        st.session_state.live_score_history.pop(0)
                    
                    # Draw landmarks on frame
                    frame = detector.draw_landmarks(frame, result)
                    
                    # Display variables
                    level = stress_info["stress_level"]
                    score = stress_info["stress_score"]
                    
                    # Color formatting
                    if level == "Calm":
                        class_style = "calm-text"
                    elif level == "Slight Stress":
                        class_style = "slight-text"
                    else:
                        class_style = "high-text"
                        
                    stress_level_placeholder.markdown(
                        f'<div class="metric-card"><h4>Stress Level</h4><span class="{class_style}" style="font-size: 1.5rem;">{level}</span></div>',
                        unsafe_allow_html=True
                    )
                    stress_score_placeholder.markdown(
                        f'<div class="metric-card"><h4>Stress Score</h4><span style="font-size: 1.5rem; font-weight: bold;">{score:.2f} / 1.0</span></div>',
                        unsafe_allow_html=True
                    )
                    
                    # Draw Feature bar chart
                    feat_df = pd.DataFrame({
                        "Metric": ["Eyebrow Raise", "Lip Tension", "Blink Intensity", "Head Nod", "Symmetry Delta"],
                        "Intensity": [
                            features.get("eyebrow_raise", 0.0),
                            features.get("lip_tension", 0.0),
                            features.get("blink_intensity", 0.0),
                            features.get("head_nod", 0.0),
                            features.get("symmetry_delta", 0.0)
                        ]
                    })
                    feature_chart_placeholder.bar_chart(feat_df.set_index("Metric"))
                else:
                    stress_level_placeholder.markdown(
                        '<div class="metric-card"><h4>Stress Level</h4><span style="font-size: 1.3rem;">No Face Detected</span></div>',
                        unsafe_allow_html=True
                    )
                    
                # Render frame in Streamlit
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(img_rgb, channels="RGB", use_column_width=True)
                
                # Display temporal live plot
                if len(st.session_state.live_score_history) > 0:
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(st.session_state.live_score_history, color='red', linewidth=2)
                    ax.fill_between(range(len(st.session_state.live_score_history)), st.session_state.live_score_history, color='red', alpha=0.1)
                    ax.set_ylim(0, 1)
                    ax.set_title("Stress Trend (Last 100 Frames)")
                    ax.set_ylabel("Score")
                    ax.set_facecolor('#1E1E1E')
                    fig.patch.set_facecolor('#1E1E1E')
                    ax.tick_params(colors='white')
                    ax.yaxis.label.set_color('white')
                    ax.title.set_color('white')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                
                time.sleep(0.01)
                
            cap.release()

    elif app_mode == "Image File Analyzer":
        st.subheader("🖼️ Static Image Stress Analyzer")
        st.write("Upload a static photo/selfie to extract and inspect micro-expression features.")
        
        uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert uploaded file to OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            # Analyze
            timestamp_ms = int(time.time() * 1000)
            landmarks, result = detector.extract_landmarks(frame, timestamp_ms)
            
            img_col, chart_col = st.columns([1, 1])
            
            with img_col:
                st.write("Processed Frame")
                if landmarks is not None:
                    # Draw landmarks on a copy of the frame
                    vis_frame = frame.copy()
                    vis_frame = detector.draw_landmarks(vis_frame, result)
                    img_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                st.image(img_rgb, use_column_width=True)
                
            with chart_col:
                if landmarks is not None:
                    raw_features = feature_engineer.get_features(landmarks)
                    stress_info = stress_model.predict(raw_features)
                    
                    level = stress_info["stress_level"]
                    score = stress_info["stress_score"]
                    
                    st.markdown(f"### Stress Level: **{level}**")
                    st.markdown(f"### Stress Score: **{score:.2f}**")
                    
                    st.progress(score)
                    
                    st.write("Extracted Facial Feature Vectors:")
                    feat_df = pd.DataFrame({
                        "Metric": ["Eyebrow Raise", "Lip Tension", "Blink Intensity", "Head Nod", "Symmetry Delta"],
                        "Intensity": [
                            raw_features.get("eyebrow_raise", 0.0),
                            raw_features.get("lip_tension", 0.0),
                            raw_features.get("blink_intensity", 0.0),
                            raw_features.get("head_nod", 0.0),
                            raw_features.get("symmetry_delta", 0.0)
                        ]
                    })
                    st.bar_chart(feat_df.set_index("Metric"))
                else:
                    st.error("No face detected in the uploaded image. Please try another photo.")

    elif app_mode == "Session Logs Viewer":
        st.subheader("📊 Session Log Statistics Viewer")
        st.write("Browse and plot statistical trends from logged `.csv` sessions.")
        
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        csv_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
        
        if not csv_files:
            st.info("No session log files (.csv) found. Run the analyzer first to record sessions!")
        else:
            selected_file = st.selectbox("Select Session Log File", csv_files)
            file_path = os.path.join(log_dir, selected_file)
            
            df = pd.read_csv(file_path)
            st.write(f"Session data shape: {df.shape[0]} frames logged.")
            
            # Show summary stats
            st.markdown("### Session Metrics Overview")
            avg_score = df["stress_score"].mean()
            max_score = df["stress_score"].max()
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Average Stress Score", f"{avg_score:.2f}")
            with c2:
                st.metric("Peak Stress Score", f"{max_score:.2f}")
                
            # Line plot of stress score
            st.markdown("### Stress Score Profile")
            st.line_chart(df["stress_score"])
            
            # Multi-line plot of metrics
            st.markdown("### Feature Evolution")
            metric_cols = ["eyebrow_raise", "lip_tension", "blink_rate", "head_nod", "symmetry"]
            available_cols = [c for c in metric_cols if c in df.columns]
            if available_cols:
                st.line_chart(df[available_cols])
            
            # Raw data
            if st.checkbox("Show Raw Log Table"):
                st.dataframe(df)
