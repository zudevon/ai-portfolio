"""
Data Preprocessor Module
Handles data loading, preprocessing, and organization for training
"""

import os
import shutil
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

class DataPreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.class_names = []
    
    def organize_data_from_filenames(self, source_folder, organized_folder='organized_data'):
        """
        Organize images into class folders based on filename prefixes
        
        Args:
            source_folder: Folder containing images with names like 'bread-01.jpg'
            organized_folder: Output folder with class subfolders
        
        Returns:
            Path to organized folder
        """
        print(f"📁 Organizing data from {source_folder}...")
        
        # Create organized folder structure
        organized_path = Path(organized_folder)
        if organized_path.exists():
            shutil.rmtree(organized_path)
        organized_path.mkdir(exist_ok=True)
        
        # Get all image files
        source_path = Path(source_folder)
        if not source_path.exists():
            print(f"❌ Source folder not found: {source_folder}")
            return None
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in {source_folder}")
            return None
        
        # Extract class names from filenames and organize
        class_counts = {}
        
        for image_file in image_files:
            # Extract class name (everything before the first dash or number)
            filename = image_file.stem  # filename without extension
            
            # Try to extract class name
            class_name = None
            if '-' in filename:
                class_name = filename.split('-')[0]
            elif '_' in filename:
                class_name = filename.split('_')[0]
            else:
                # If no separator, look for first digit
                for i, char in enumerate(filename):
                    if char.isdigit():
                        class_name = filename[:i]
                        break
                if not class_name:
                    class_name = filename  # Use entire filename as class
            
            class_name = class_name.lower().strip()
            
            # Create class folder if it doesn't exist
            class_folder = organized_path / class_name
            class_folder.mkdir(exist_ok=True)
            
            # Copy image to class folder
            destination = class_folder / image_file.name
            shutil.copy2(image_file, destination)
            
            # Count images per class
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Print organization summary
        print("📊 Data organization summary:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   {class_name}: {count} images")
        
        self.class_names = sorted(class_counts.keys())
        return str(organized_path)
    
    def prepare_data(self, data_folder, batch_size=32, validation_split=0.2):
        """
        Prepare data generators for training
        
        Args:
            data_folder: Folder containing training images
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        
        Returns:
            train_generator, validation_generator, class_names
        """
        # First, try to organize data if it's not already organized
        data_path = Path(data_folder)
        
        # Check if data is already organized (has class subfolders)
        subfolders = [f for f in data_path.iterdir() if f.is_dir()]
        image_files = [f for f in data_path.iterdir() if f.is_file() and 
                      f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}]
        
        if image_files and not subfolders:
            # Data needs to be organized
            print("🔄 Data not organized, organizing by filename...")
            organized_folder = self.organize_data_from_filenames(data_folder)
            if organized_folder is None:
                return None, None, None
            data_folder = organized_folder
        elif not subfolders:
            print(f"❌ No data found in {data_folder}")
            return None, None, None
        
        # Create data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            data_folder,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Create validation generator
        validation_generator = validation_datagen.flow_from_directory(
            data_folder,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        # Get class names
        class_names = list(train_generator.class_indices.keys())
        
        print(f"✅ Data prepared successfully!")
        print(f"   Classes: {class_names}")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {validation_generator.samples}")
        
        return train_generator, validation_generator, class_names
    
    def create_sample_data_structure(self, base_folder='training_images'):
        """
        Create sample folder structure and instructions
        """
        base_path = Path(base_folder)
        base_path.mkdir(exist_ok=True)
        
        # Create README file with instructions
        readme_content = """# Training Images Folder

Place your labeled training images in this folder using the following naming convention:

## Naming Convention:
- bread-01.jpg, bread-02.png, bread-03.jpeg, ...
- dog-01.jpg, dog-02.png, dog-03.jpeg, ...
- pig-01.jpg, pig-02.png, pig-03.jpeg, ...
- pug-01.jpg, pug-02.png, pug-03.jpeg, ...

## Supported formats:
- .jpg, .jpeg, .png, .bmp, .gif

## Example structure:
training_images/
├── bread-01.jpg
├── bread-02.png
├── dog-01.jpg
├── dog-02.jpeg
├── pig-01.png
├── pug-01.jpg
└── ...

## Tips:
- Use at least 20-50 images per class for better results
- More diverse images = better model performance
- Images will be automatically resized to 224x224 pixels
"""
        
        with open(base_path / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"📁 Created training folder structure at: {base_folder}")
        print(f"📝 Instructions saved to: {base_folder}/README.md")
        
        return str(base_path)