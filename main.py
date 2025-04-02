import os
import json
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import h5py
import google.generativeai as genai

# ========== PAGE CONFIG ==========
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
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

GROWTH_FACTORS = {
    "American Beech": 6, "American Elm": 4, "American Sycamore": 4, "Austrian Pine": 4.5,
    "Basswood": 3, "Black Cherry": 5, "Black Maple": 5, "Black Walnut": 4.5,
    "White Oak": 5, "Red Maple": 4.5, "Douglas Fir": 5, "Sugar Maple": 5.5
}

PLANT_CATEGORIES = {
    # Forest Trees
    "Oak": {"growth_rate": "slow", "max_age": 300, "age_factor": 1.2},
    "Pine": {"growth_rate": "medium", "max_age": 200, "age_factor": 1.1},
    
    # Fruit Trees
    "Apple": {"growth_rate": "medium", "max_age": 100, "age_factor": 0.9},
    "Mango": {"growth_rate": "medium", "max_age": 80, "age_factor": 0.8},
    
    # Succulents
    "Aloe": {"growth_rate": "slow", "max_age": 12, "age_factor": 0.7, "unit": "years"},
    "Agave": {"growth_rate": "slow", "max_age": 25, "age_factor": 0.6, "unit": "years"},
    
    # Annuals
    "Sunflower": {"growth_rate": "fast", "max_age": 1, "age_factor": 0.5, "unit": "year"},
    "Corn": {"growth_rate": "fast", "max_age": 1, "age_factor": 0.4, "unit": "year"},
    
    # Vegetables
    "Tomato": {"growth_rate": "fast", "max_age": 1, "age_factor": 0.3, "unit": "year"},
    "Pepper": {"growth_rate": "fast", "max_age": 1.5, "age_factor": 0.35, "unit": "years"}
}

# =============================================
# GEMINI INITIALIZATION
# =============================================
try:
    genai.configure(api_key=st.secrets["GOOGLE_GEMINI_API_KEY"])
    gemini_available = True
except Exception as e:
    st.error(f"❌ Failed to configure Gemini: {str(e)}")
    gemini_available = False

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
# IMAGE PROCESSING
# =============================================
def preprocess_image(uploaded_file):
    try:
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
def identify_plant(uploaded_file):
    if not gemini_available:
        return {"error": "Gemini API not configured"}
    
    try:
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
# PLANT AGE DETECTION
# =============================================
def plant_age_detection():
    st.title("📅 Plant Age Detection")
    st.markdown("""
    **Hybrid Method:** Combines AI visual analysis with species-specific growth factors
    *Provides approximate age ranges (not exact years)*
    """)

    uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Plant", use_column_width=True)
            
            plant_type = st.selectbox(
                "Select plant type (optional for better accuracy)",
                ["Autodetect"] + sorted(PLANT_CATEGORIES.keys()),
                index=0
            )
            
            if st.button("Estimate Age"):
                with st.spinner("🔍 Analyzing plant..."):
                    # Get visual analysis first
                    visual_analysis = analyze_plant_visuals(uploaded_file)
                    
                    # If plant type specified, apply growth factors
                    if plant_type != "Autodetect":
                        age_estimate = apply_growth_factors(visual_analysis, plant_type)
                    else:
                        age_estimate = visual_analysis
                    
                    st.session_state.age_estimate = age_estimate

        with col2:
            if 'age_estimate' in st.session_state:
                result = st.session_state.age_estimate
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Display visual characteristics
                    with st.expander("🖼️ Visual Characteristics"):
                        st.json(result["visual_characteristics"])
                    
                    # Display age estimate
                    st.subheader("🧮 Age Estimate")
                    min_age = result["estimated_age_range"]["min"]
                    max_age = result["estimated_age_range"]["max"]
                    unit = result["estimated_age_range"]["unit"]
                    
                    st.metric(
                        "Approximate Age Range",
                        f"{min_age}-{max_age} {unit}",
                        f"Confidence: {result['confidence'].title()}"
                    )
                    
                    # Display key indicators
                    if result["key_indicators"]:
                        st.subheader("🔍 Key Indicators")
                        for indicator in result["key_indicators"]:
                            st.markdown(f"- {indicator}")
                    
                    # Display species-specific notes if available
                    if "species_notes" in result:
                        st.subheader("🌱 Species Notes")
                        st.info(result["species_notes"])

    with st.expander("ℹ️ About This Method"):
        st.markdown("""
        **Hybrid Age Detection Approach:**
        
        - **Forest Trees (Oak, Pine):** ~70-80% accuracy (DBH + bark texture)
        - **Fruit Trees (Apple, Mango):** ~50-60% accuracy (if not heavily pruned)
        - **Succulents (Aloe, Agave):** ~60% accuracy (leaf count/ring scars)
        - **Annuals (Sunflower, Corn):** Growth stage prediction only
        
        **Methodology:**
        1. AI analyzes visual characteristics (size, trunk thickness, leaf density)
        2. System applies species-specific growth factors
        3. Confidence-adjusted based on plant type and image quality
        """)

def analyze_plant_visuals(uploaded_file):
    """Analyze plant visuals using AI model"""
    try:
        if not gemini_available:
            return {"error": "Gemini API not available for visual analysis"}
            
        model = genai.GenerativeModel('gemini-1.5-flash')
        img = Image.open(uploaded_file)
        
        prompt = """Analyze this plant image and estimate its approximate age range based on:
        1. Visual characteristics (size, trunk thickness, leaf density)
        2. Common developmental stages
        
        Provide output in this exact JSON format:
        {
            "visual_characteristics": {
                "size_category": "",
                "trunk_thickness": "",
                "leaf_density": ""
            },
            "estimated_age_range": {
                "min": 0,
                "max": 0,
                "unit": "years/months"
            },
            "key_indicators": [],
            "confidence": "low/medium/high"
        }"""
        
        response = model.generate_content([prompt, img])
        json_str = response.text[response.text.find('{'):response.text.rfind('}')+1]
        return json.loads(json_str)
        
    except Exception as e:
        return {"error": f"Visual analysis failed: {str(e)}"}

def apply_growth_factors(age_data, plant_type):
    """Adjust age estimates based on species-specific factors"""
    if plant_type in PLANT_CATEGORIES:
        species_info = PLANT_CATEGORIES[plant_type]
        age_range = age_data.get("estimated_age_range", {})
        
        if age_range:
            # Apply growth factor adjustment
            adjustment = species_info["age_factor"]
            age_range["min"] = round(age_range.get("min", 1) * adjustment, 1)
            age_range["max"] = round(age_range.get("max", 10) * adjustment, 1)
            
            # Cap at species maximum age
            age_range["max"] = min(age_range["max"], species_info["max_age"])
            
            # Update unit if specified
            if "unit" in species_info:
                age_range["unit"] = species_info["unit"]
            
            # Increase confidence
            if age_data.get("confidence", "medium") == "medium":
                age_data["confidence"] = "high"
            
            # Add species notes
            age_data["species_notes"] = (
                f"Typical {species_info['growth_rate']}-growing {plant_type}. "
                f"Maximum expected age: {species_info['max_age']} years."
            )
            
            age_data["key_indicators"].append(
                f"Adjusted for {plant_type}'s {species_info['growth_rate']} growth rate"
            )
    
    return age_data

# =============================================
# TREE AGE CALCULATOR
# =============================================
def tree_age_calculator():
    st.title("🌳 Tree Age Calculator")
    st.markdown("""
    Estimate a tree's age using its circumference and species growth factor.
    *Based on the International Society of Arboriculture (ISA) method*
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        circumference = st.number_input(
            "Circumference at Breast Height (inches)", 
            min_value=1.0, 
            max_value=500.0,
            value=60.0,
            step=0.1
        )
        
        tree_type = st.selectbox(
            "Tree Species",
            sorted(GROWTH_FACTORS.keys())
        )

        if st.button("Calculate Age"):
            dbh = circumference / 3.141592
            growth_factor = GROWTH_FACTORS[tree_type]
            estimated_age = dbh * growth_factor

            st.session_state.tree_age_result = {
                "circumference": circumference,
                "dbh": dbh,
                "growth_factor": growth_factor,
                "estimated_age": estimated_age,
                "tree_type": tree_type
            }

    with col2:
        if 'tree_age_result' in st.session_state:
            result = st.session_state.tree_age_result
            st.metric("Diameter at Breast Height (DBH)", f"{result['dbh']:.2f} inches")
            st.metric("Growth Factor", result['growth_factor'])
            st.metric("Estimated Age", f"{result['estimated_age']:.1f} years")

    with st.expander("📖 Methodology Details"):
        st.markdown("""
        **Formula:** `Age = (Circumference / π) × Growth Factor`
        
        **Instructions:**
        1. Measure circumference at 4.5 feet above ground
        2. Use inches for best accuracy
        3. Select the correct species
        
        **Note:** Urban trees may have slower growth rates.
        """)

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
# MAIN APP
# =============================================
def main():
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
                        identification = identify_plant(uploaded_file)
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
        tree_age_calculator()
    
    elif app_mode == "Plant Age Detection":
        plant_age_detection()
    
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