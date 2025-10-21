# 🌸 Flower Recognition Streamlit App

A professional web application for recognizing flower types using a trained Keras deep learning model with an interactive and modern UI.

## ✨ Features

### Core Functionality
- **Image Upload**: Support for JPG, JPEG, and PNG formats
- **Real-time Classification**: Instant flower type prediction
- **Confidence Scores**: Detailed probability breakdown for all classes
- **Visual Results**: Color-coded predictions with progress bars

### Advanced Features
- **Configurable Settings**: Adjust image dimensions and class names via sidebar
- **Model Information**: View detailed model architecture and parameters
- **Results Export**: Download prediction results as text file
- **Responsive Design**: Custom CSS for enhanced user experience
- **Image Preprocessing**: Automatic resizing and normalization
- **Error Handling**: Graceful fallbacks and helpful error messages

## 🚀 Installation

### Option 1: Using Virtual Environment (Recommended)

1. Create a virtual environment:
```powershell
python -m venv .venv
```

2. Activate the virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

### Option 2: Direct Installation

```powershell
pip install -r requirements.txt
```

## 📦 Requirements

- Python 3.8 or higher
- TensorFlow 2.x or Keras 3.x
- Streamlit
- Pillow (PIL)
- NumPy

All dependencies are listed in `requirements.txt`

## 🎯 Running the App

1. Ensure the model file `flower_recognition_model.keras` is in the same directory

2. Run the Streamlit app:
```powershell
streamlit run app.py
```

3. The app will automatically open in your default web browser at `http://localhost:8501`

## 🌺 Default Flower Types

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

*These can be customized in the sidebar settings*

## ⚙️ Configuration

### Sidebar Settings

The app includes a configuration sidebar where you can:

1. **Adjust Image Dimensions**
   - Default: 180x180 pixels
   - Range: 32-512 pixels
   - The model will automatically resize uploaded images

2. **Customize Class Names**
   - Toggle custom class names
   - Add your own flower categories
   - Supports up to 20 classes

3. **View Model Information**
   - Input/output shapes
   - Number of parameters
   - Layer count

### Model Customization

If your model has different specifications:

- **Input Size**: Adjust via sidebar or modify default values in code
- **Number of Classes**: Update in sidebar settings
- **Class Names**: Use the custom class names feature in sidebar
- **Preprocessing**: The app uses standard normalization (0-1 range)

## 📊 Features Breakdown

### User Interface
- ✅ Modern, responsive design
- ✅ Custom CSS styling
- ✅ Progress indicators
- ✅ Expandable sections
- ✅ File size display
- ✅ Image information viewer

### Predictions Display
- ✅ Top prediction with confidence
- ✅ Sorted probability list
- ✅ Color-coded results
- ✅ Visual progress bars
- ✅ Percentage values

### Additional Tools
- ✅ Download results as text
- ✅ View processed images
- ✅ Tips and instructions
- ✅ Error messages with solutions

## 🌐 Deployment

### Deploy on Streamlit Cloud

1. Push your code to a GitHub repository (include `app.py`, `requirements.txt`, and your `.keras` model file)

2. Visit [share.streamlit.io](https://share.streamlit.io)

3. Sign in with GitHub

4. Click "New app" and select your repository

5. Set the main file path to `app.py`

6. Click "Deploy"

### Deploy on Other Platforms

The app can also be deployed on:
- **Heroku**: Add a `Procfile` with `web: streamlit run app.py --server.port=$PORT`
- **Google Cloud Run**: Create a `Dockerfile` with Streamlit
- **AWS EC2**: Run directly on an instance
- **Azure App Service**: Configure Python web app

## 🔧 Troubleshooting

### Model Loading Issues
- Ensure `flower_recognition_model.keras` is in the same directory as `app.py`
- Check that TensorFlow or Keras is properly installed
- Try adjusting the backend in the model loading function

### Image Processing Errors
- Verify image format (JPG, JPEG, PNG only)
- Check image dimensions in sidebar settings
- Ensure image is not corrupted

### Dependency Conflicts
- Use a virtual environment
- Update packages: `pip install --upgrade -r requirements.txt`
- Check Python version compatibility (3.8+)

## 📝 File Structure

```
flower-recognition-app/
│
├── app.py                           # Main Streamlit application
├── flower_recognition_model.keras   # Trained Keras model
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
│
└── .venv/                           # Virtual environment (optional)
```

## 💡 Tips for Best Results

### Image Quality
- Use high-resolution images (at least 224x224)
- Ensure good lighting
- Center the flower in the frame
- Avoid busy backgrounds

### Model Performance
- Test with various image sizes
- Adjust preprocessing if needed
- Use images similar to training data
- Consider model retraining for better accuracy

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share feedback

## 📄 License

This project is open source and available for educational and commercial use.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [TensorFlow/Keras](https://www.tensorflow.org/)
- Image processing with [Pillow](https://python-pillow.org/)

---

**Made with ❤️ for flower enthusiasts and ML practitioners**
