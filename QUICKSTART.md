# Quick Start Guide

## ⚠️ IMPORTANT: Model Image Size

**Your model requires exactly 144x144 pixel input!**

The app is now configured to use 144x144 by default. When you upload any image (even 1200x1200), it will be automatically resized to 144x144 before prediction.

**Why 144x144?**
- Based on the error messages, your model architecture expects this size
- The dense layer expects 7,776 features after convolution
- This comes from 144x144 input after pooling/convolution layers

**No action needed** - just upload your images and the app handles the rest!

---

## Getting Started in 3 Steps

### Step 1: Setup Environment
```powershell
# Navigate to project directory
cd "f:\Temp\New folder"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the App
```powershell
streamlit run app.py
```

### Step 3: Use the App
1. Open browser at `http://localhost:8501`
2. Upload a flower image
3. View predictions and confidence scores!

---

## Customizing for Your Model

### If Your Model Has Different Classes

1. **Option A: Use the Sidebar** (Easy)
   - Run the app
   - Check "Use custom class names" in sidebar
   - Enter your class names (one per line)

2. **Option B: Edit Code** (Permanent)
   - Open `app.py`
   - Find line: `FLOWER_CLASSES = ['Orchid', 'Sunflower', 'Tulip', 'Lotus', 'Lilly']`
   - Replace with your classes: `FLOWER_CLASSES = ['class1', 'class2', 'class3']`

### If Your Model Has Different Input Size

1. **Option A: Use the Sidebar** (Easy)
   - Adjust "Image Height" and "Image Width" in sidebar
   - Common sizes: 160, 144, 128, 224, 180
   - **Start with 160x160** (this is the most common for your model)

2. **Option B: Edit Code** (Permanent)
   - Open `app.py`
   - Find: `img_height = st.number_input("Image Height", value=180, ...)`
   - Change `value=180` to your size

### If Your Model Uses Different Preprocessing

Edit the `preprocess_image()` function in `app.py`:

```python
def preprocess_image(image, target_size):
    # Your preprocessing here
    
    # Example: Different normalization
    img_array = img_array / 127.5 - 1.0  # For [-1, 1] range
    
    # Example: No normalization
    # img_array = img_array  # Keep as [0, 255]
    
    return img_array, image_resized
```

---

## Checking Your Model Specifications

To find your model's requirements:

```python
import keras
model = keras.models.load_model('flower_recognition_model.keras')

print("Input shape:", model.input_shape)  # e.g., (None, 180, 180, 3)
print("Output shape:", model.output_shape)  # e.g., (None, 5)
print("Number of classes:", model.output_shape[-1])
```

The input shape tells you: `(batch_size, height, width, channels)`
- Height and Width: Use these values in sidebar settings
- Channels: Usually 3 for RGB images

---

## Common Issues and Solutions

### Issue: "Input shape incompatible" or "expected axis -1 to have value X"
**Solution:** 
- **The app automatically resizes images** - your 1200x1200 image will be resized to match model requirements
- Change the **Image Height** and **Image Width** in the sidebar
- Start with **160x160** (default), then try: 144, 128, or 224
- Make sure both height and width are the **same value**
- The error means your current size setting doesn't match what the model expects

**Quick Fix:**
1. Go to sidebar → Image Settings
2. Set both Height and Width to **160**
3. Reupload your image (any size will work - it gets auto-resized!)

### Issue: "Model not loading"
**Solution:** 
- Check file name is exactly `flower_recognition_model.keras`
- Ensure it's in the same folder as `app.py`
- Try: `ls` or `dir` to verify file exists

### Issue: "Wrong number of predictions"
**Solution:**
- Update class count in sidebar
- Or edit `FLOWER_CLASSES` list in code

### Issue: "Poor prediction accuracy"
**Solution:**
- Adjust image size to match training
- Check if preprocessing matches training
- Use similar images to training data

### Issue: "App runs but no predictions appear"
**Solution:**
- Check browser console for errors (F12)
- Verify model file is not corrupted
- Try with different image

---

## Advanced Configuration

### Change Theme Colors

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"      # Change to your color
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Increase Upload Size Limit

Edit `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 20  # Increase from 10MB to 20MB
```

### Add Multiple Models

Modify `load_model()` function to support model selection:

```python
@st.cache_resource
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model, "Keras"

# In sidebar
model_choice = st.selectbox("Select Model", ["Model 1", "Model 2"])
model_path = "model1.keras" if model_choice == "Model 1" else "model2.keras"
model, backend = load_model(model_path)
```

---

## Performance Tips

1. **Use caching**: Already implemented with `@st.cache_resource`
2. **Optimize images**: Compress before uploading
3. **Batch processing**: For multiple images, modify code to accept lists
4. **GPU acceleration**: TensorFlow will use GPU if available

---

## Deployment Checklist

Before deploying to production:

- [ ] Test with various image types and sizes
- [ ] Verify all class names are correct
- [ ] Check error handling works properly
- [ ] Test on different browsers
- [ ] Optimize model size if needed
- [ ] Set appropriate upload limits
- [ ] Add authentication if required
- [ ] Monitor performance metrics

---

## Need Help?

1. Check the main `README.md` for detailed documentation
2. Review Streamlit docs: https://docs.streamlit.io
3. Check TensorFlow/Keras docs: https://www.tensorflow.org
4. Open an issue on GitHub (if applicable)

---

**Happy Flower Recognition! 🌸**
