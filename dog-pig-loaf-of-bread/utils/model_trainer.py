"""
Model Trainer Module
Handles the creation and training of the CNN model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
    
    def create_model(self, num_classes=4, input_shape=(224, 224, 3)):
        """
        Create a CNN model for image classification
        
        Args:
            num_classes: Number of classes to classify
            input_shape: Input image shape (height, width, channels)
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Data augmentation layers
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Rescaling layer
            layers.Rescaling(1./255),
            
            # Convolutional layers
            layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            layers.Conv2D(256, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, model, train_generator, validation_generator, 
                   epochs=50, model_path='models/classifier_model.h5'):
        """
        Train the model
        
        Args:
            model: Keras model to train
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            model_path: Path to save the trained model
        """
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📊 Training samples: {train_generator.samples}")
        print(f"📊 Validation samples: {validation_generator.samples}")
        print(f"🏷️  Classes: {list(train_generator.class_indices.keys())}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation accuracy/loss"""
        if self.history is None:
            print("⚠️  No training history to plot")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📈 Training history plot saved to: models/training_history.png")
    
    def evaluate_model(self, model, test_generator):
        """Evaluate model performance"""
        print("🔍 Evaluating model...")
        
        # Get predictions
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        
        # Calculate accuracy
        accuracy = np.mean(predicted_classes == true_classes)
        print(f"📊 Test Accuracy: {accuracy:.4f}")
        
        return accuracy, predicted_classes, true_classes