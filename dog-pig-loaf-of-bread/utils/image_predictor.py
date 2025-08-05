"""
Image Predictor Module
Handles loading trained models and making predictions on new images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

class ImagePredictor:
    def __init__(self, img_size=(224, 224)):
        self.model = None
        self.class_names = []
        self.img_size = img_size
    
    def load_model(self, model_path):
        """
        Load a trained model and its class names
        
        Args:
            model_path: Path to the saved model (.h5 file)
        """
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        try:
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")
            
            # Try to load class names
            class_names_path = model_path.replace('.h5', '_classes.txt')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                print(f"📝 Class names loaded: {self.class_names}")
            else:
                # Default class names if file not found
                self.class_names = ['bread', 'dog', 'pig', 'pug']
                print(f"⚠️  Class names file not found, using default: {self.class_names}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed image array
        """
        try:
            # Load and resize image
            img = image.load_img(image_path, target_size=self.img_size)
            
            # Convert to array and normalize
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize to [0, 1] range
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {str(e)}")
            return None
    
    def predict_image(self, image_path, show_image=True):
        """
        Predict the class of an image
        
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            print("❌ No model loaded. Please load a model first.")
            return None, None
        
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None, None
        
        # Preprocess the image
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None, None
        
        try:
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]
            
            # Get class name
            if predicted_class_index < len(self.class_names):
                predicted_class = self.class_names[predicted_class_index]
            else:
                predicted_class = f"class_{predicted_class_index}"
            
            # Show image if requested
            if show_image:
                self.display_prediction(image_path, predicted_class, confidence, predictions[0])
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None, None
    
    def display_prediction(self, image_path, predicted_class, confidence, all_predictions):
        """
        Display the image with prediction results
        
        Args:
            image_path: Path to the image
            predicted_class: Predicted class name
            confidence: Confidence score
            all_predictions: Array of all class probabilities
        """
        plt.figure(figsize=(12, 6))
        
        # Display image
        plt.subplot(1, 2, 1)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}', 
                 fontsize=14, fontweight='bold')
        
        # Display prediction probabilities
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(self.class_names))
        colors = ['green' if i == np.argmax(all_predictions) else 'skyblue' 
                 for i in range(len(self.class_names))]
        
        bars = plt.barh(y_pos, all_predictions * 100, color=colors)
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Confidence (%)')
        plt.title('Class Probabilities')
        plt.xlim(0, 100)
        
        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, all_predictions)):
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Save the result
        result_path = f"results/prediction_{os.path.basename(image_path)}.png"
        os.makedirs('results', exist_ok=True)
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Prediction visualization saved to: {result_path}")
    
    def predict_batch(self, image_folder):
        """
        Predict classes for all images in a folder
        
        Args:
            image_folder: Folder containing images to predict
        
        Returns:
            List of prediction results
        """
        if not os.path.exists(image_folder):
            print(f"❌ Image folder not found: {image_folder}")
            return []
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [f for f in os.listdir(image_folder) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in {image_folder}")
            return []
        
        results = []
        print(f"🔮 Predicting {len(image_files)} images...")
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            predicted_class, confidence = self.predict_image(image_path, show_image=False)
            
            result = {
                'filename': image_file,
                'predicted_class': predicted_class,
                'confidence': confidence
            }
            results.append(result)
            
            if predicted_class:
                print(f"   {image_file}: {predicted_class} ({confidence:.2%})")
        
        return results