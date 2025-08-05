#!/usr/bin/env python3
"""
Image Classification System
Main script for training and using the bread/pug/pig/dog classifier
"""

import os
import sys
import argparse
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from model_trainer import ModelTrainer
from image_predictor import ImagePredictor
from data_preprocessor import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Image Classification System')
    parser.add_argument('mode', choices=['train', 'predict'], 
                       help='Mode: train a new model or predict on new images')
    parser.add_argument('--input', '-i', required=False,
                       help='Input path: training data folder for train mode, image file for predict mode')
    parser.add_argument('--model', '-m', default='models/classifier_model.h5',
                       help='Path to save/load the model')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for training (default: 32)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🔄 Starting model training...")
        
        # Default training data path
        training_data_path = args.input or 'training_images'
        
        if not os.path.exists(training_data_path):
            print(f"❌ Training data folder not found: {training_data_path}")
            print("Please create the folder and add labeled images like:")
            print("  training_images/bread-01.jpg")
            print("  training_images/dog-01.png")
            print("  training_images/pig-01.jpg")
            print("  training_images/pug-01.png")
            return
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        train_generator, validation_generator, class_names = preprocessor.prepare_data(
            training_data_path, batch_size=args.batch_size
        )
        
        if train_generator is None:
            print("❌ Failed to prepare training data")
            return
        
        # Train model
        trainer = ModelTrainer()
        model = trainer.create_model(num_classes=len(class_names))
        
        # Create models directory
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        
        trainer.train_model(
            model, 
            train_generator, 
            validation_generator,
            epochs=args.epochs,
            model_path=args.model
        )
        
        # Save class names
        class_names_path = args.model.replace('.h5', '_classes.txt')
        with open(class_names_path, 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        print(f"✅ Model training completed! Saved to: {args.model}")
        print(f"📝 Class names saved to: {class_names_path}")
    
    elif args.mode == 'predict':
        if not args.input:
            # Look for images in upload folder
            upload_folder = 'upload'
            if os.path.exists(upload_folder):
                images = [f for f in os.listdir(upload_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                if images:
                    args.input = os.path.join(upload_folder, images[0])
                    print(f"🔍 Found image to predict: {args.input}")
        
        if not args.input:
            print("❌ No input image specified. Use --input or place an image in the 'upload' folder")
            return
        
        if not os.path.exists(args.model):
            print(f"❌ Model not found: {args.model}")
            print("Please train a model first using: python main.py train")
            return
        
        print(f"🔮 Predicting image: {args.input}")
        
        # Load predictor
        predictor = ImagePredictor()
        predictor.load_model(args.model)
        
        # Make prediction
        prediction, confidence = predictor.predict_image(args.input)
        
        if prediction:
            print(f"🎯 Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.2%}")
        else:
            print("❌ Failed to make prediction")

if __name__ == "__main__":
    main()