"""
Model training and evaluation module.

Handles:
- Training with early stopping and checkpointing
- Model evaluation
- Prediction generation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Using tf.keras for consistency
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class IDSModelTrainer:
    """Handles training and evaluation of the IDS CNN model."""
    
    def __init__(self, model, model_save_path='saved_models/ids_cnn.keras'):
        """
        Initialize the trainer.
        
        Args:
            model: Compiled Keras model
            model_save_path: Path to save the trained model
        """
        self.model = model
        self.model_save_path = model_save_path
        self.history = None
        
        # Create save directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    def train(self, X_train, y_train, X_test, y_test, 
              epochs=50, batch_size=128, validation_split=0.2):
        """
        Train the model with early stopping and checkpointing.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (used for validation)
            y_test: Test labels (used for validation)
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            validation_split: Not used (kept for compatibility)
            
        Returns:
            Training history
        """
        print("\n=== Training IDS CNN Model ===")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_test.shape[0]}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n=== Training Complete ===")
        
        # Evaluate on test set
        self.evaluate(X_test, y_test)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n=== Evaluating Model ===")
        
        # Get predictions
        test_loss, test_acc = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return {
            'loss': test_loss,
            'accuracy': test_acc
        }
    
    def predict(self, X, return_probabilities=True):
        """
        Generate predictions for input data.
        
        Args:
            X: Input features
            return_probabilities: If True, return softmax probabilities
            
        Returns:
            Predictions (probabilities or class indices)
        """
        predictions = self.model.predict(X, verbose=0)
        
        if return_probabilities:
            return predictions  # Softmax probabilities
        else:
            return np.argmax(predictions, axis=1)  # Class indices
    
    def get_detailed_report(self, X_test, y_test, label_mapping):
        """
        Generate detailed classification report.
        
        Args:
            X_test: Test features
            y_test: Test labels
            label_mapping: Dictionary mapping indices to class names
            
        Returns:
            str: Classification report
        """
        y_pred = self.predict(X_test, return_probabilities=False)
        
        # Get class names
        target_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
        
        print("\n=== Detailed Classification Report ===")
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)
        
        return report
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
        plt.close()
    
    def save_model(self, path=None):
        """Save the trained model."""
        save_path = path or self.model_save_path
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")
    
    @staticmethod
    def load_model(path):
        """
        Load a trained model.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded Keras model
        """
        print(f"Loading model from: {path}")
        # Import custom layers to ensure they're registered
        from models.cnn_model import CNNBlock, ResidualCNNBlock
        model = keras.models.load_model(path)
        return model


if __name__ == "__main__":
    # This would be run as part of the full pipeline
    print("Model trainer module loaded successfully")
