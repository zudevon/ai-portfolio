#!/usr/bin/env python3
"""
Setup Script for Image Classification System
Creates necessary folders and provides setup instructions
"""

import os
from pathlib import Path

def create_folder_structure():
    """Create the required folder structure"""
    folders = [
        'training_images',
        'upload',
        'models',
        'results',
        'utils'
    ]
    
    print("📁 Creating folder structure...")
    
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f"   ✅ Created: {folder}/")
    
    # Create sample images folder structure
    training_path = Path('training_images')
    
    # Create README for training images
    readme_content = """# Training Images

Place your labeled training images here using this naming convention:

## File naming format:
- bread-01.jpg, bread-02.png, bread-03.jpeg
- dog-01.jpg, dog-02.png, dog-03.jpeg  
- pig-01.jpg, pig-02.png, pig-03.jpeg
- pug-01.jpg, pug-02.png, pug-03.jpeg

## Requirements:
- At least 20-50 images per class recommended
- Supported formats: .jpg, .jpeg, .png, .bmp, .gif
- Images will be automatically resized to 224x224 pixels

## Example:
training_images/
├── bread-01.jpg
├── bread-02.png
├── dog-01.jpg
├── dog-02.jpeg
├── pig-01.png
├── pug-01.jpg
└── ...
"""
    
    with open(training_path / 'README.md', 'w') as f:
        f.write(readme_content)
    
    # Create README for upload folder
    upload_readme = """# Upload Folder

Place new images here that you want to classify.

The prediction script will automatically find and process images in this folder.

Supported formats: .jpg, .jpeg, .png, .bmp, .gif
"""
    
    with open(Path('upload') / 'README.md', 'w') as f:
        f.write(upload_readme)

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """tensorflow>=2.12.0
pillow>=9.0.0
matplotlib>=3.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("📦 Created requirements.txt")

def create_run_scripts():
    """Create convenient run scripts"""
    
    # Training script
    train_script = """#!/bin/bash
# Train the image classifier
echo "🚀 Starting model training..."
python main.py train --epochs 50 --batch-size 32
"""
    
    with open('train.sh', 'w') as f:
        f.write(train_script)
    os.chmod('train.sh', 0o755)
    
    # Prediction script
    predict_script = """#!/bin/bash
# Predict images in upload folder
echo "🔮 Making predictions..."
python main.py predict
"""
    
    with open('predict.sh', 'w') as f:
        f.write(predict_script)
    os.chmod('predict.sh', 0o755)
    
    print("📜 Created convenience scripts: train.sh, predict.sh")

def print_setup_instructions():
    """Print setup and usage instructions"""
    instructions = """
🎯 IMAGE CLASSIFICATION SYSTEM SETUP COMPLETE!

📋 NEXT STEPS:

1️⃣  INSTALL DEPENDENCIES:
    pip install -r requirements.txt

2️⃣  PREPARE TRAINING DATA:
    - Add labeled images to training_images/ folder
    - Use naming convention: bread-01.jpg, dog-01.png, pig-01.jpeg, pug-01.jpg
    - Minimum 20-50 images per class recommended

3️⃣  TRAIN THE MODEL:
    python main.py train
    # OR use the convenience script:
    ./train.sh

4️⃣  MAKE PREDICTIONS:
    - Place new images in upload/ folder
    - Run: python main.py predict
    # OR use the convenience script:
    ./predict.sh

📂 FOLDER STRUCTURE:
    ├── main.py                 # Main script
    ├── utils/                  # Utility modules
    │   ├── model_trainer.py    # Model training
    │   ├── data_preprocessor.py # Data preparation
    │   └── image_predictor.py  # Image prediction
    ├── training_images/        # Your labeled training images
    ├── upload/                 # Images to classify
    ├── models/                 # Saved models
    ├── results/               # Prediction results
    └── requirements.txt       # Dependencies

🔧 ADVANCED USAGE:
    # Custom training parameters
    python main.py train --epochs 100 --batch-size 16
    
    # Predict specific image
    python main.py predict --input path/to/image.jpg
    
    # Use custom model path
    python main.py predict --model models/my_model.h5

💡 TIPS:
    - More diverse training images = better accuracy
    - Images are automatically resized to 224x224 pixels
    - Training progress and results are saved automatically
    - Check results/ folder for prediction visualizations

❓ NEED HELP?
    - Check README.md files in each folder
    - Run: python main.py --help
"""
    
    print(instructions)

def main():
    """Main setup function"""
    print("🔧 Setting up Image Classification System...")
    print("=" * 50)
    
    create_folder_structure()
    create_requirements_file()
    create_run_scripts()
    
    print("=" * 50)
    print_setup_instructions()

if __name__ == "__main__":
    main()