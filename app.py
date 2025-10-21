import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="Flower Recognition",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #FF6B6B;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🌸 Flower Recognition Model")
st.markdown("### Upload an image of a flower to identify its type")
st.markdown("---")

# Load the model with caching
@st.cache_resource
def load_model():
    try:
        # Try importing tensorflow first
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model('flower_recognition_model.keras')
            backend = "TensorFlow"
        except ImportError:
            # Fall back to keras
            import keras
            model = keras.models.load_model('flower_recognition_model.keras')
            backend = "Keras"
        
        # Try to detect the correct input size from model
        try:
            input_shape = model.input_shape
            if input_shape and len(input_shape) >= 3:
                detected_height = input_shape[1] if input_shape[1] else 180
                detected_width = input_shape[2] if input_shape[2] else 180
                return model, backend, (detected_height, detected_width)
        except:
            pass
        
        return model, backend, (180, 180)  # Default fallback
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure 'flower_recognition_model.keras' is in the same directory as this app.")
        return None, None, (180, 180)

# Load the model
model, backend, detected_size = load_model()

# Use detected size or allow override
default_height, default_width = detected_size

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Model parameters (can be adjusted based on actual model)
    st.subheader("Image Settings")
    
    # Quick size presets
    st.write("**Quick Presets:**")
    preset_cols = st.columns(4)
    preset_clicked = None
    with preset_cols[0]:
        if st.button("128x128", use_container_width=True):
            preset_clicked = 128
    with preset_cols[1]:
        if st.button("144x144", use_container_width=True):
            preset_clicked = 144
    with preset_cols[2]:
        if st.button("160x160", use_container_width=True):
            preset_clicked = 160
    with preset_cols[3]:
        if st.button("224x224", use_container_width=True):
            preset_clicked = 224
    
    if preset_clicked:
        st.session_state.img_size = preset_clicked
    
    # Calculate the correct size based on the error pattern
    # Error shows 21600 vs 7776, which suggests 144x144 input
    # 144x144 -> after conv/pooling -> ~7776 features
    recommended_size = 144
    
    # Get the size from session state or use recommended
    if 'img_size' not in st.session_state:
        st.session_state.img_size = recommended_size
    
    current_size = st.session_state.img_size
    
    st.warning(f"⚠️ **IMPORTANT**: This model requires **144x144** input size!")
    st.info(f"Current: {current_size}x{current_size} | Recommended: 144x144")
    st.caption("📏 Any uploaded image will be automatically resized to your selected dimensions")
    
    # Force same height and width
    img_size = st.number_input("Image Size (Height & Width)", value=144, min_value=32, max_value=512, step=16, 
                                help="Both height and width will use this value",
                                key="size_input")
    img_height = img_size
    img_width = img_size
    
    st.success(f"✅ Will resize all images to: {img_size}x{img_size}")
    
    st.subheader("Flower Classes")
    use_custom_classes = st.checkbox("Use custom class names", value=False)
    
    if use_custom_classes:
        num_classes = st.number_input("Number of classes", value=5, min_value=2, max_value=20)
        default_classes = [f"Class_{i}" for i in range(num_classes)]
        class_names_input = st.text_area(
            "Enter class names (one per line)", 
            value="\n".join(default_classes),
            height=150
        )
        FLOWER_CLASSES = [name.strip() for name in class_names_input.split('\n') if name.strip()]
    else:
        # Common flower datasets: Oxford Flowers, TF Flowers
        FLOWER_CLASSES = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    
    st.info(f"Current classes: {len(FLOWER_CLASSES)}")
    
    st.subheader("About")
    st.markdown("""
    This app uses a trained Keras deep learning model for flower classification.
    
    **Tips for best results:**
    - Use clear, well-lit images
    - Center the flower in the frame
    - Avoid busy backgrounds
    - Supported formats: JPG, JPEG, PNG
    """)

# Display model status
if model is not None:
    st.success(f"✅ Model loaded successfully (Backend: {backend})")
    
    # Try to display model info
    with st.expander("📊 Model Information"):
        try:
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Output Shape:** {model.output_shape}")
            st.write(f"**Number of Parameters:** {model.count_params():,}")
            
            # Show layer count
            st.write(f"**Number of Layers:** {len(model.layers)}")
        except:
            st.write("Model information not available")

# Image preprocessing function
def preprocess_image(image, target_size):
    """
    Preprocess the uploaded image for model prediction.
    Handles any input size (e.g., 1200x1200) and resizes to target size.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to exact target size using high-quality resampling
    # This handles any input size (1200x1200, 800x600, etc.) -> target_size
    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image_resized, dtype=np.float32)
    
    # Normalize pixel values to [0, 1] range
    img_array = img_array / 255.0
    
    # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, image_resized

# Main content area
st.markdown("### 📤 Upload Your Image")

# File uploader with more options
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a flower image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a flower",
        label_visibility="collapsed"
    )

with col2:
    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.metric("File Size", f"{file_size:.1f} KB")

# Process uploaded image
if uploaded_file is not None and model is not None:
    try:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Display original image info
        with st.expander("📷 Original Image Info"):
            st.write(f"**Original Format:** {image.format}")
            st.write(f"**Original Mode:** {image.mode}")
            st.write(f"**Original Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Will be resized to:** {img_width} x {img_height} pixels")
            if image.size[0] != img_width or image.size[1] != img_height:
                st.caption("✅ Automatic resizing will be applied for model compatibility")
        
        # Create two columns for display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, use_container_width=True, caption="Original Image")
        
        # Make prediction
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner('🌸 Analyzing the flower...'):
                try:
                    # Preprocess the image
                    target_size = (img_width, img_height)
                    processed_image, resized_img = preprocess_image(image, target_size)
                    
                    # Get predictions
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx] * 100
                    
                    # Display the top prediction with styling
                    st.markdown(f"""
                    <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;'>
                        <h3 style='color: #FF6B6B; margin: 0;'>🌺 {FLOWER_CLASSES[predicted_class_idx].title()}</h3>
                        <p style='font-size: 24px; margin: 10px 0;'><strong>{confidence:.2f}%</strong> confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show processed image
                    with st.expander("View Processed Image"):
                        st.image(resized_img, caption=f"Resized to {img_width}x{img_height}", use_container_width=True)
                
                except Exception as e:
                    error_message = str(e)
                    st.error(f"❌ Error making prediction: {error_message}")
                    
                    # Provide helpful suggestions based on error
                    if "incompatible" in error_message.lower() or "shape" in error_message.lower():
                        st.warning("🔧 **Shape Mismatch Detected!**")
                        st.info("""
                        **Try these image sizes in the sidebar:**
                        - **160 x 160** (most common)
                        - **144 x 144**
                        - **128 x 128**
                        - **224 x 224**
                        
                        The model expects a specific input size. Adjust both height and width to the same value.
                        """)
                        
                        # Suggest automatic sizes to try
                        with st.expander("🤖 Auto-detect recommended sizes"):
                            st.write("Based on the error, try these dimensions:")
                            common_sizes = [128, 144, 160, 180, 224]
                            cols = st.columns(5)
                            for i, size in enumerate(common_sizes):
                                with cols[i]:
                                    if st.button(f"{size}x{size}", key=f"size_{size}"):
                                        st.info(f"Please manually set both Height and Width to {size} in the sidebar, then reupload the image.")
                    else:
                        st.info("💡 Try adjusting the image size settings in the sidebar or use a different image.")
        
        # Full width section for all predictions
        st.markdown("---")
        st.subheader("📊 Detailed Predictions")
        
        # Create a more detailed view
        if predictions is not None:
            # Sort predictions by probability
            pred_indices = np.argsort(predictions[0])[::-1]
            
            for idx in pred_indices:
                flower_name = FLOWER_CLASSES[idx] if idx < len(FLOWER_CLASSES) else f"Class {idx}"
                prob = predictions[0][idx]
                
                # Color code based on probability
                if prob > 0.5:
                    color = "#4CAF50"  # Green
                elif prob > 0.2:
                    color = "#FF9800"  # Orange
                else:
                    color = "#9E9E9E"  # Grey
                
                col_a, col_b, col_c = st.columns([2, 3, 1])
                with col_a:
                    st.write(f"**{flower_name.title()}**")
                with col_b:
                    st.progress(float(prob))
                with col_c:
                    st.write(f"{prob*100:.2f}%")
        
        # Download section
        st.markdown("---")
        st.subheader("💾 Save Results")
        
        # Prepare result text
        result_text = f"""Flower Recognition Results
================================
Predicted Flower: {FLOWER_CLASSES[predicted_class_idx].title()}
Confidence: {confidence:.2f}%

All Predictions:
"""
        for idx in pred_indices:
            flower_name = FLOWER_CLASSES[idx] if idx < len(FLOWER_CLASSES) else f"Class {idx}"
            prob = predictions[0][idx]
            result_text += f"  - {flower_name.title()}: {prob*100:.2f}%\n"
        
        st.download_button(
            label="📥 Download Results as Text",
            data=result_text,
            file_name="flower_prediction_results.txt",
            mime="text/plain"
        )
                
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try uploading a different image.")

elif uploaded_file is None:
    # Show welcome message and instructions
    st.info("👆 Please upload a flower image to get started!")
    
    # Create attractive info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='padding: 20px; background-color: #e3f2fd; border-radius: 10px; text-align: center;'>
            <h3>📤</h3>
            <h4>Upload</h4>
            <p>Select a flower image from your device</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 20px; background-color: #f3e5f5; border-radius: 10px; text-align: center;'>
            <h3>🔍</h3>
            <h4>Analyze</h4>
            <p>AI model processes the image instantly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='padding: 20px; background-color: #e8f5e9; border-radius: 10px; text-align: center;'>
            <h3>🌸</h3>
            <h4>Identify</h4>
            <p>Get flower type with confidence score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sample images section
    st.markdown("### 💡 Tips for Best Results")
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **✅ Do:**
        - Use clear, focused images
        - Ensure good lighting
        - Center the flower in frame
        - Use high-resolution images
        """)
    
    with tips_col2:
        st.markdown("""
        **❌ Avoid:**
        - Blurry or dark images
        - Multiple flowers in one image
        - Heavy filters or edits
        - Extreme angles
        """)

elif model is None:
    st.warning("⚠️ Model not loaded. Please check that the model file exists in the correct location.")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown("""
    <div style='text-align: center;'>
        <p>Built with <strong>Streamlit</strong> 🎈 and <strong>Keras</strong> 🧠</p>
        <p style='font-size: 12px; color: #666;'>Deep Learning Flower Recognition System</p>
    </div>
    """, unsafe_allow_html=True)
