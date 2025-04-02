import os
import sys
import subprocess
import json
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# ========== PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND ==========
import streamlit as st
st.set_page_config(
    page_title="Botanicare",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CONSTANTS
# =============================================
MODEL_FILE = "trained_model.h5"
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    # ... rest of your class names ...
]

GROWTH_FACTORS = {
    # ... your growth factors dictionary ...
}

PLANT_CATEGORIES = {
    # ... your plant categories dictionary ...
}

# =============================================
# DEPENDENCY VERIFICATION
# =============================================
def check_dependencies():
    with st.sidebar.expander("⚙️ Dependency Status", expanded=False):
        if os.path.exists('requirements.txt'):
            st.code(open('requirements.txt').read(), language='text')
        else:
            st.error("requirements.txt not found!")
        
        def check_package(package_name, pip_name=None):
            try:
                __import__(package_name)
                st.success(f"✅ {package_name} installed")
                return True
            except ImportError:
                pip_name = pip_name or package_name
                st.warning(f"⚠️ {package_name} missing! Attempting install...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                    st.rerun()
                except:
                    st.error(f"❌ Failed to install {package_name}")
                    return False
        
        check_package("tensorflow", "tensorflow-cpu")
        check_package("PIL", "pillow")
        check_package("google.generativeai")
        check_package("h5py")
        check_package("numpy")

# =============================================
# GEMINI INITIALIZATION
# =============================================
def init_gemini():
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GOOGLE_GEMINI_API_KEY"])
        return genai, True
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini: {str(e)}")
        return None, False

# =============================================
# MODEL LOADING
# =============================================
@st.cache_resource
def load_model():
    possible_paths = [
        MODEL_FILE,
        "trained_model.keras",
        os.path.join(os.path.dirname(__file__), MODEL_FILE)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                import h5py
                import tensorflow as tf
                with h5py.File(path, 'r') as f:
                    if 'model_weights' not in f.keys():
                        continue
                
                model = tf.keras.models.load_model(path, compile=False)
                model.compile(optimizer='adam', loss='categorical_crossentropy')
                st.success(f"✅ Model loaded from: {os.path.abspath(path)}")
                return model
                
            except Exception as e:
                st.warning(f"⚠️ Failed to load {path}: {str(e)}")
                continue
    
    st.error(f"""
    ❌ Model loading failed. Please ensure:
    1. The file '{MODEL_FILE}' exists in: {os.path.abspath('.')}
    2. You have proper read permissions
    3. TensorFlow version is compatible
    """)
    st.stop()

# =============================================
# SIDEBAR NAVIGATION
# =============================================
def create_sidebar():
    with st.sidebar:
        params = st.query_params
        is_dark = params.get("theme", [""])[0] == "dark"

        st.markdown(f"""
        <style>
            .sidebar .sidebar-content {{ background-color: {"#1a1a1a" if is_dark else "#f0f7f4"}; }}
            .nav-item {{ padding: 12px 15px; margin: 8px 0; border-radius: 10px; }}
            .nav-item:hover {{ background-color: {"#2a2a2a" if is_dark else "#d4e8e0"}; }}
            .nav-item.active {{ background-color: {"#3a7d44" if is_dark else "#2e8b57"}; color: white; }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-bottom:20px;">
            <h1 style="color:#2e8b57; font-size:28px;">🌿 Botanicare</h1>
        </div>
        """, unsafe_allow_html=True)

        nav_options = ["Home", "Disease Detection", "Plant Identification", 
                      "Tree Age Calculator", "Plant Age Detection", "About"]
        nav_icons = {
            "Home": "🏠",
            "Disease Detection": "🔍",
            "Plant Identification": "🌱",
            "Tree Age Calculator": "🌳",
            "Plant Age Detection": "📅",
            "About": "📚"
        }

        app_mode = st.radio(
            "Navigate to",
            nav_options,
            format_func=lambda x: f"{nav_icons[x]} {x}",
            label_visibility="collapsed"
        )

        dark_mode = st.toggle("Dark Mode", value=is_dark)
        if dark_mode != is_dark:
            params["theme"] = "dark" if dark_mode else "light"
            st.query_params = params
            st.rerun()

        return app_mode

# =============================================
# IMAGE PROCESSING
# =============================================
def preprocess_image(uploaded_file):
    try:
        from PIL import Image
        import tensorflow as tf
        import numpy as np
        
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        st.error(f"❌ Image processing error: {str(e)}")
        return None

# =============================================
# DISEASE PREDICTION
# =============================================
def predict_disease(model, image_array):
    try:
        import numpy as np
        predictions = model.predict(image_array)
        pred_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        return CLASS_NAMES[pred_idx], confidence
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None

# =============================================
# PLANT IDENTIFICATION
# =============================================
def identify_plant(uploaded_file, gemini_available, genai):
    if not gemini_available:
        return {"error": "Gemini API not configured"}
    
    try:
        from PIL import Image
        model = genai.GenerativeModel('gemini-1.5-flash')
        img = Image.open(uploaded_file)
        
        prompt = """Provide plant information in this exact JSON format:
        {
            "taxonomy": {
                "common_name": "",
                "scientific_name": "",
                "kingdom": "Plantae",
                "family": "",
                "genus": "",
                "species": ""
            },
            "description": "",
            "growth_conditions": {
                "sunlight": [],
                "water_needs": "",
                "soil_type": [],
                "hardiness_zones": "",
                "temperature_range": ""
            },
            "morphology": {
                "leaves": "",
                "flowers": "",
                "fruit": "",
                "height": "",
                "special_features": []
            },
            "uses": {
                "culinary": [],
                "medicinal": [],
                "other": []
            },
            "propagation_methods": [],
            "conservation_status": "",
            "interesting_facts": []
        }"""
        
        with st.spinner("🔍 Analyzing plant..."):
            response = model.generate_content([prompt, img])
            try:
                json_str = response.text[response.text.find('{'):response.text.rfind('}')+1]
                return json.loads(json_str)
            except Exception as e:
                return {"error": f"Response parsing failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Identification error: {str(e)}"}

# =============================================
# MAIN APP
# =============================================
def main():
    check_dependencies()
    genai, gemini_available = init_gemini()
    
    if 'model' not in st.session_state:
        st.session_state.model = load_model()
    
    app_mode = create_sidebar()
    
    if app_mode == "Home":
        st.title("🌿 Botanicare")
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image("home.png", use_column_width=True)
            except FileNotFoundError:
                st.warning("Home image not found")
        with col2:
            st.markdown("""
            **Features:**
            - 🌱 Plant identification
            - 🔍 Disease detection
            - 🌳 Tree age calculator
            - 📅 Growth stage analysis
            """)
    
    elif app_mode == "Disease Detection":
        st.title("🔍 Disease Detection")
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            cols = st.columns([1, 1.5])
            with cols[0]:
                st.image(uploaded_file)
                if st.button("Analyze"):
                    with st.spinner("Analyzing..."):
                        img_array = preprocess_image(uploaded_file)
                        if img_array is not None:
                            disease, confidence = predict_disease(st.session_state.model, img_array)
                            if disease:
                                plant, condition = disease.split("___")
                                st.session_state.diagnosis = {
                                    "plant": plant,
                                    "condition": condition.replace('_', ' '),
                                    "confidence": f"{confidence*100:.1f}%"
                                }
            
            if 'diagnosis' in st.session_state:
                with cols[1]:
                    diagnosis = st.session_state.diagnosis
                    if "healthy" in diagnosis["condition"].lower():
                        st.success(f"✅ Healthy {diagnosis['plant']}")
                    else:
                        st.error(f"⚠️ {diagnosis['plant']} has: {diagnosis['condition']}")
                    st.metric("Confidence", diagnosis["confidence"])
    
    elif app_mode == "Plant Identification":
        st.title("🌱 Plant Identification")
        uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, use_column_width=True)
                if st.button("Identify"):
                    with st.spinner("Identifying..."):
                        identification = identify_plant(uploaded_file, gemini_available, genai)
                        st.session_state.identification = identification
            
            if 'identification' in st.session_state:
                with col2:
                    if "error" in st.session_state.identification:
                        st.error(st.session_state.identification["error"])
                    else:
                        st.subheader("Taxonomy")
                        st.write(st.session_state.identification["taxonomy"])
                        st.subheader("Description")
                        st.write(st.session_state.identification["description"])
    
    elif app_mode == "Tree Age Calculator":
        st.title("🌳 Tree Age Calculator")
        # ... implement tree_age_calculator function ...
    
    elif app_mode == "Plant Age Detection":
        st.title("📅 Plant Age Detection")
        # ... implement plant_age_detection function ...
    
    elif app_mode == "About":
        st.title("📚 About")
        st.markdown("""
        **PlantAI Analyzer v1.4**
        Comprehensive plant analysis tool combining:
        - Computer vision
        - Growth factor calculations
        - AI-powered analysis
        Developed for botanists, gardeners, and plant enthusiasts.
        """)

if __name__ == "__main__":
    main()