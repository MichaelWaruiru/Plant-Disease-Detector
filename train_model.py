import argparse
import os
import logging
import tensorflow as tf # Required for GPU configuration
from ml_model import PlantDiseaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU Setup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Set memory growth to avoid OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Using GPU: {gpus}")
    except RuntimeError as e:
        logging.error(f"Error setting GPU memory growth: {e}")
else:
        logging.info("No GPU found. Training will run on CPU.")
        
# Checkpoint path
CHECKPOINT_PATH = "checkpoints/model_checkpoint.h5"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--dataset_path', type=str, default="dataset/PlantVillage/Plant Diseases Dataset/New Plant Diseases Dataset(Augmented)/train", help='Path to PlantVillage dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data to use for testing')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        logging.error(f"Dataset path {args.dataset_path} does not exist!")
        logging.info("Please download the PlantVillage dataset and extract it to the dataset folder")
        return
    
    # Create model
    logging.info("Initializing model...")
    model = PlantDiseaseModel()
    
    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        logging.info(f"Found checkpoint at {CHECKPOINT_PATH}, resuming training...")
        model.model.load_weights(CHECKPOINT_PATH)
    else:
        logging.info("No checkpoint found, starting fresh training.")
        
        # Setup checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True
        )
    
    # Train model
    logging.info(f"Starting training: epochs={args.epochs}, batch_size={args.batch_size}")
    
    result = model.train_with_real_data(
        train_dir=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if result and all(result):
        history, val_loss, val_accuracy = result
        logging.info("Training completed successfully!")
        logging.info(f"Validation Loss: {val_loss:.4f}")
        logging.info(f"Final Validation Accuracy: {val_accuracy:.4f}")
        logging.info(f"Model saved to {model.model_path}")
        
        # Save final model
        model.model.save(model.model_path)
        logging.info(f"Final model saved to {model.model_path}")
        
        # Plot training history
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            logging.info("Training history saved to training_history.png")
            
        except ImportError:
            logging.info("Matplotlib not available, skipping plot generation")
    
    else:
        logging.error("Training failed!")

if __name__ == "__main__":
    main()
