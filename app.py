import streamlit as st
import cv2
import numpy as np
from PIL import Image
from analyzer import BiometricAnalyzer

# Page config
st.set_page_config(
    page_title="Biometric Face Analysis",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    h1 {
        color: #1f2937;
    }
    h2, h3 {
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_analyzer():
    return BiometricAnalyzer()

def main():
    st.title("👤 Biometric Face Analysis Dashboard")
    st.markdown("Upload a face image to analyze facial features, ratios, and estimating biometrics.")

    # Sidebar
    st.sidebar.header("Configuration")
    st.sidebar.info("This tool uses Google Mediapipe for high-fidelity 468-point face mesh and DeepFace for age/gender estimation.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        # Initialize analyzer
        with st.spinner("Loading models..."):
            analyzer = get_analyzer()
            
        # Analysis
        with st.spinner("Analyzing face..."):
            landmarks, rect = analyzer.get_landmarks(image_np)
            
            if landmarks is None:
                st.error("No face detected! Please use a clearer image.")
            else:
                # DeepFace
                # Note: DeepFace might take a moment
                deepface_result = analyzer.analyze_deepface(image_np)
                
                # Ratios
                ratios = analyzer.calculate_ratios(landmarks)
                score = analyzer.calculate_beauty_score(ratios)
                eval_text = analyzer.evaluate_beauty_text(score)
                
                # Visualization
                viz_image = analyzer.visualize(image_np, landmarks)
                
                # Display Results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Visualized Landmarks")
                    st.image(viz_image, use_container_width=True, caption="Analysis Visualization")
                
                with col2:
                    st.subheader("Biometric Metrics")
                    
                    # Row 1: Key Stats
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Age", f"{deepface_result.get('age', 'N/A')}")
                    m2.metric("Gender", f"{deepface_result.get('gender', 'N/A')}")
                    m3.metric("Beauty Score", f"{score:.1f}/10", eval_text)
                    
                    st.markdown("---")
                    st.subheader("Golden Ratio Analysis")
                    
                    # Row 2: Ratios
                    r1, r2 = st.columns(2)
                    r1.metric("Eye Dist / Nose Length", f"{ratios.get('eye_distance_nose_length', 0):.2f}")
                    r2.metric("Nose Len / Mouth Width", f"{ratios.get('nose_length_mouth_width', 0):.2f}")
                    
                    st.info("""
                    **Metric Explanations:**
                    - **Beauty Score**: Calculated based on deviation from the Golden Ratio (1.618).
                    - **Ratios**: Key facial proportions used in aesthetic analysis.
                    """)

if __name__ == "__main__":
    main()
