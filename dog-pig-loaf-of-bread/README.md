# 🍞🐶🐷 Image Classification System

A complete machine learning system for classifying images of bread, dogs, pigs, and pugs using TensorFlow/Keras.

## 🚀 Quick Start

### 1. Setup
```bash
# Run setup script to create folder structure
python setup.py

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Training Data
Add your labeled images to the `training_images/` folder using this naming convention:
- `bread-01.jpg`, `bread-02.png`, `bread-03.jpeg`
- `dog-01.jpg`, `dog-02.png`, `dog-03.jpeg`
- `pig-01.jpg`, `pig-02.png`, `pig-03.jpeg`
- `pug-01.jpg`, `pug-02.png`, `pug-03.jpeg`

**Tip:** Use at least 20-50 images per class for better results!

### 3. Train the Model
```bash
# Basic training
python main.py train

# Custom parameters
python main.py train --epochs 100 --batch-size 16

# Or use convenience script
./train.sh
```

### 4. Make Predictions
```bash
# Place images in upload/ folder, then:
python main.py predict

# Or predict specific image:
python main.py predict --input path/to/image.jpg

# Or use convenience script
./predict.sh
```

## 📁 Project Structure

```
image-classifier/
├── main.py                    # Main script for training/prediction
├── setup.py                   # Setup script
├── requirements.txt           # Python dependencies
├── train.sh                   # Training convenience script
├── predict.sh                 # Prediction convenience script
│
├── utils/                     # Utility modules
│   ├── __init__.py           # Package init
│   ├── model_trainer.py      # Model creation and training
│   ├── data_preprocessor.py  # Data loading and preprocessing
│   └── image_predictor.py    # Image prediction and visualization
│
├── training_images/          # Your labeled training data
│   ├── README.md
│   ├── bread-01.jpg
│   ├── dog-01.jpg
│   ├── pig-01.jpg
│   └── pug-01.jpg
│
├── upload/                   # Images to classify
│   └── README.md
│
├── models/                   # Saved models and training artifacts
│   ├── classifier_model.h5   # Trained model
│   ├── classifier_model_classes.txt # Class names
│   └── training_history.png  # Training plots
│
└── results/                  # Prediction results and visualizations
    └── prediction_*.png
```

## 🔧 Advanced Usage

### Custom Model Training
```bash
# Train with specific parameters
python main.py train \
    --input custom_training_folder \
    --model models/custom_model.h5 \
    --epochs 75 \
    --batch-size 64
```

### Batch Predictions
```python
from utils import ImagePredictor

predictor = ImagePredictor()
predictor.load_model('models/classifier_model.h5')
results = predictor.predict_batch('upload/')
```

### Model Architecture
The system uses a Convolutional Neural Network (CNN) with:
- Data augmentation layers (rotation, flip, zoom)
- 4 convolutional blocks with max pooling
- Dropout layers for regularization
- Dense layers for classification
- Softmax output for multi-class classification

## 📊 Features

- **Automatic Data Organization**: Organizes images by filename prefixes
- **Data Augmentation**: Improves model robustness with image transformations
- **Training Visualization**: Plots training history and saves results
- **Prediction Visualization**: Shows predictions with confidence scores
- **Batch Processing**: Predict multiple images at once
- **Model Checkpointing**: Saves best model during training
- **Early Stopping**: Prevents overfitting

## 🎯 Supported Classes

- **Bread** 🍞
- **Dog** 🐶
- **Pig** 🐷
- **Pug** 🐾

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.12+
- PIL/Pillow for image processing
- Matplotlib for visualization
- NumPy for array operations
- Scikit-learn for data splitting

## 💡 Tips for Best Results

1. **More Data = Better Results**: Use 50+ images per class if possible
2. **Diverse Images**: Include various angles, lighting, backgrounds
3. **Image Quality**: Use clear, well-lit images
4. **Balanced Dataset**: Try to have similar numbers of images per class
5. **Validation**: The system automatically sets aside 20% for validation

## 🐛 Troubleshooting

### Common Issues:

**"No training data found"**
- Make sure images are in `training_images/` folder
- Check file naming convention (class-number.extension)
- Verify image formats are supported (.jpg, .png, .jpeg, .bmp, .gif)

**"Model not found"**
- Train a model first: `python main.py train`
- Check model path in `models/` folder

**Low accuracy**
- Add more training images per class
- Increase training epochs: `--epochs 100`
- Check if images are properly labeled

**Memory errors**
- Reduce batch size: `--batch-size 16`
- Use smaller images (current: 224x224)

## 🚀 Next Steps

- Add more classes by including more image types
- Experiment with different model architectures
- Try transfer learning with pre-trained models
- Deploy the model as a web service
- Create a mobile app interface

## 📄 License

This project is open source and available under the MIT License.