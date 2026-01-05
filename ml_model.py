import os
import logging
import numpy as np
from PIL import Image
import cv2
from tensorflow import keras
from keras import layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import json
from disease_data import DISEASE_INFO, TREATMENT_RECOMMENDATIONS

# For faster GPU training
from keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")


class PlantDiseaseModel:
  """Plant Disease Detection Model"""
  def __init__(self):
    self.model = None
    self.dataset_path = "dataset/PlantVillage/Plant Diseases Dataset"
    self.img_size = (224, 224)
   
    self.class_names_path = "models/class_names.json"
    self.model_path = "models/plant_disease_cnn_model.keras" 
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    self.class_names = self.load_class_names()
    self.num_classes = len(self.class_names)
    
    self.load_or_create_model()
    
    
  def load_class_names(self):
    """Load class names from dataset directory"""
    if os.path.exists(self.class_names_path):
      with open(self.class_names_path, "r") as f:
        return json.load(f)
      
    # If dataset exists, load class names from directory
    if os.path.exists(self.dataset_path):
      class_names = sorted([
        folder for folder in os.listdir(self.dataset_path)
        if os.path.isdir(os.path.join(self.dataset_path, folder))
      ])
      with open(self.class_names_path, "w") as f:
        json.dump(class_names, f)
      return class_names
    # Fallback to DISEASE_INFO keys if no dataset or class names file
    logging.warning("Dataset not found, using default class names")
    return sorted(list(DISEASE_INFO.keys()))
    
  def create_model(self):
    """Create CNN Model"""
    model = keras.Sequential([
      keras.Input(shape=(224, 224, 3)),
      
     # Data augmentation layers
      # layers.RandomFlip("horizontal"),
      # layers.RandomRotation(0.1),
      # layers.RandomZoom(0.1),
      # layers.RandomContrast(0.1),
      
      # Rescaling
      layers.Rescaling(1./255),
      
      # First Conv Block
      layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2, 2)),
      layers.SpatialDropout2D(0.3),
      
      # Second Conv Block
      # layers.Conv2D(64, (3, 3), activation="relu"),
      # layers.BatchNormalization(),
      # layers.MaxPooling2D((2, 2)),
      # layers.Dropout(0.25),
      
      # Third Conv Block
      layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2, 2)),
      layers.SpatialDropout2D(0.3),
      
      # Fourth Conv Block
      layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2, 2)),
      layers.SpatialDropout2D(0.3),
      
      # Global Average Pooling
      layers.GlobalAveragePooling2D(),
      
      # Dense layers
      layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
      layers.BatchNormalization(),
      layers.Dropout(0.5),
      
      # layers.Dense(256, activation='relu'),
      # layers.BatchNormalization(),
      # layers.Dropout(0.5),
      
      # Output layer
      layers.Dense(self.num_classes, activation='softmax')
      
    ])
    
    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.001),
      loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
      metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
      ]
    )
    
    return model
  
  def load_or_create_model(self):
    """Load existing model or create new one"""
    try:
      if os.path.exists(self.model_path) and os.path.exists(self.class_names_path):
          self.model = keras.models.load_model(self.model_path)
          with open(self.class_names_path, 'r') as f:
              self.class_names = json.load(f)
          logging.info("Loaded existing CNN model")
      else:
        self.model = self.create_model()
        # self.train_model_with_synthetic_data()
        logging.info("Created new CNN model")
    except Exception as e:
      logging.error(f"Error loading model: {str(e)}")
      self.model = self.create_model()
      logging.info("Created new model due to loading error")
  
  def preprocess_image(self, image_path):
    """Preprocess image for prediction"""
    try:
      # Load and preprocess image
      image = Image.open(image_path).convert('RGB')
      image = image.resize(self.img_size)
      
      # Convert to numpy array and normalize
      img_array = np.array(image).astype("float32") / 255
      img_array = np.expand_dims(img_array, axis=0)
      
      return img_array
    except Exception as e:
      logging.error(f"Error preprocessing image: {str(e)}")
      return None
  
  def train_model_with_synthetic_data(self):
    """Train model with synthetic data if real dataset is not available"""
    try:
      # Create synthetic training data
      X_train = np.random.rand(1000, 224, 224, 3) * 255
      y_train = np.random.randint(0, self.num_classes, 1000)
      
      X_val = np.random.rand(200, 224, 224, 3) * 255
      y_val = np.random.randint(0, self.num_classes, 200)
      
      early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
      )
      
      # Train model with reduced epochs for synthetic data
      history = self.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,  # Reduced for synthetic data
        batch_size=32,
        verbose=1,
        callbacks=[early_stop]
      )
      
      # Save model and class names
      self.model.save(self.model_path)
      with open(self.class_names_path, "w") as f:
        json.dump(self.class_names, f)
      
      logging.info("Model trained with synthetic data")
      return history
        
    except Exception as e:
      logging.error(f"Error training model: {str(e)}")
  
  def train_with_real_data(self, train_dir, epochs=30, batch_size=64):
    """Train model with real PlantVillage dataset"""
    try:
      import tensorflow as tf
      from keras.applications import MobileNetV2
      
      AUTOTUNE = tf.data.AUTOTUNE
      
      # Dataset loading for training generator (tf.data, GPU-friendly)
      train_dataset = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=self.img_size,
        batch_size=batch_size,
        label_mode="categorical"
      )
      
      # Validation dataset generator
      validation_dataset = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=self.img_size,
        batch_size=batch_size,
        label_mode="categorical"
      )
      
      # Create data generators with augmentation
      # train_datagen = ImageDataGenerator(
      #   rescale=1./255,
      #   rotation_range=30,
      #   width_shift_range=0.2,
      #   brightness_range=[0.8, 1.2],
      #   height_shift_range=0.2,
      #   horizontal_flip=True,
      #   zoom_range=0.3,
      #   shear_range=0.2,
      #   fill_mode="nearest",
      #   validation_split=0.2
      # )
      
      # # Training generator
      # train_generator = train_datagen.flow_from_directory(
      #   train_dir,
      #   target_size=self.img_size,
      #   batch_size=batch_size,
      #   class_mode="categorical",
      #   subset="training",
      #   shuffle=True
      # )
      
      # Validation generator
      # validation_generator = train_datagen.flow_from_directory(
      #   train_dir,
      #   target_size=self.img_size,
      #   batch_size=batch_size,
      #   class_mode="categorical",
      #   subset="validation",
      #   shuffle=False
      # )
      
      # Update class names from generator
      self.class_names = train_dataset.class_names
      self.num_classes = len(self.class_names)
      
      with open(self.class_names_path, "w") as f:
        json.dump(self.class_names, f)
      
      # class distribution + weights
      y_train = np.concatenate([y.numpy().argmax(axis=1) for _, y in train_dataset])
      counts = np.bincount(y_train)
      # Visulaize and print class balance
      # counts = np.bincount(train_generator.classes)
      for idx, count in enumerate(counts):
        logging.info(f"Class {self.class_names[idx]}: {count} images")
        
      # Save class distrubition plot
      plt.figure(figsize=(8, 3))
      plt.bar(self.class_names, counts)
      plt.xticks(rotation=90)
      plt.tight_layout()
      plt.savefig("class_distribution.png")
      plt.close()
      
      # Compute class weights
      class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
      )
      class_weights = dict(enumerate(class_weights))
      
      # Data augmentation (runs on GPU)
      normalization = layers.Rescaling(1. / 255)
      
      augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
      ])
      
      train_dataset = (
        train_dataset.map(lambda x, y: (augmentation(normalization(x), training=True), y),
                          num_parallel_calls=AUTOTUNE).cache().shuffle(256).prefetch(AUTOTUNE)
      )
      
      validation_dataset = (
        validation_dataset.map(lambda x, y: (normalization(x), y),
                               num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
      )
      
      # User MobileNetV2 for transfer training + L2
      base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
      base_model.trainable = False #Initial freezes base model
      
      reg = regularizers.l2(0.001)
      inputs = keras.Input(shape=(224, 224, 3))
      x= base_model(inputs, training=False)
      x = layers.GlobalAveragePooling2D()(x)
      x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
      x= layers.BatchNormalization()(x)
      x = layers.Dropout(0.5)(x)
      outputs = layers.Dense(self.num_classes, activation="softmax", dtype="float32")(x)
      
      self.model = keras.Model(inputs, outputs)
      self.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
      )
      
      # Callbacks for better training
      callbacks = [
          keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True
          ),
          keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6
          ),
          keras.callbacks.ModelCheckpoint(
            self.model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
          ),
          keras.callbacks.TensorBoard(log_dir="logs")
      ]
      
      # Split epochs
      initial_epochs = max(5, int(epochs * 0.4))
      fine_tune_epochs = epochs - initial_epochs
      
      # Phase 1: Train top layers only with frozen base
      history1 = self.model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
      )
      
      # Unfreeze base model for fine-tuning
      # base_model.trainable = True
      # for layer in base_model.layers[:-40]:
      #   layer.trainable = False
      for layer in base_model.layers[-40:]:
        layer.trainable = True
        
      self.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
      )
      
      history2 = self.model.fit(
        train_dataset,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=history1.epoch[-1] + 1,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
      )
      
      # Save class names
      with open(self.class_names_path, "w") as f:
        json.dump(self.class_names, f)
      
      # Evaluate final model
      val_loss, val_accuracy = self.model.evaluate(validation_dataset, verbose=0)
      logging.info(f"Final validation accuracy: {val_accuracy:.4f}")
      
      # Combine histories
      for key in history2.history:
        history1.history[key].extend(history2.history[key])
      
      return history1, val_loss, val_accuracy
        
    except Exception as e:
      logging.error(f"Error training with real data: {str(e)}")
      return None, None, None
  
  def predict(self, image_path):
    """Predict disease from image using CNN"""
    try:
      # Preprocess image
      img_array = self.preprocess_image(image_path)
      if img_array is None:
        return self.analyze_image_heuristics(image_path)
      
      # Reload class_names to ensure consistency with model ouput
      if os.path.exists(self.class_names_path):
        with open(self.class_names_path, "r") as f:
          self.class_names = json.load(f)
          
      # Make prediction
      predictions = self.model.predict(img_array, verbose=0)[0]
      
      # Get top 3 predictions
      top_indices = np.argsort(predictions)[-3:][::-1]
      
      results = []
      for idx in top_indices:
        confidence = float(predictions[idx])
        class_name = self.class_names[idx]
        
        # Only include predictions with reasonable confidence
        if confidence > 0.5:  # Lower threshold for CNN
          info = DISEASE_INFO.get(class_name, {})
          treatment = TREATMENT_RECOMMENDATIONS.get(class_name, {})
          results.append({
            "class": class_name,
            "confidence": confidence * 100,
            "symptoms": info.get("symptoms", "Not available"),
            "causes": info.get("causes", "Not available"),
            "severity": info.get("severity", "Unknown"),
            "treatment": treatment
          })
      
      # If no confident predictions, return top prediction
      if not results:
        top_idx = np.argmax(predictions)
        results.append({
          "class": self.class_names[top_idx],
          "confidence": float(predictions[top_idx]) * 100
        })
      
      return results
        
    except Exception as e:
      logging.error(f"Error making prediction: {str(e)}")
      return self.analyze_image_heuristics(image_path)

  def analyze_image_heuristics(self, image_path):
    """Enhanced heuristic analysis when model fails"""
    try:
      image = Image.open(image_path).convert('RGB')
      img_array = np.array(image)
      
      # Advanced color analysis
      hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
      
      # Calculate color statistics
      green_pixels = np.sum((hsv[:,:,0] >= 40) & (hsv[:,:,0] <= 80) & (hsv[:,:,1] > 50))
      total_pixels = img_array.shape[0] * img_array.shape[1]
      green_ratio = green_pixels / total_pixels
      
      # Calculate average brightness
      brightness = np.mean(hsv[:,:,2])
      
      # Calculate color variance (indicator of spots/disease)
      color_variance = np.var(img_array)
      
      # Enhanced heuristics
      if green_ratio > 0.6 and brightness > 100 and color_variance < 1000:
        fallback_class = "Healthy" if "Healthy" in self.class_names else self.class_names[0]
        return [{
          "class": fallback_class,
          "confidence": 80.0
        }]
        
      # Choose a fallback severe disease if variance is high
      elif color_variance > 2000:  # High variance indicates spots/disease
        severe_candidates = [c for c in self.class_names if "Late_Blight" in c or "Virus" in c or "Severe" in c]
        fallback_class = severe_candidates[0] if severe_candidates else self.class_names[0]
        return [{
          "class": fallback_class,
          "confidence": 65.0
        }]
        
      else:
        fallback_class = self.class_names[0]
        return [{
          "class": fallback_class,
          "confidence": 60.0
        }]
            
    except Exception as e:
      logging.error(f"Error in heuristic analysis: {str(e)}")
      return [{
        "class": "Unable to detect disease",
        "confidence": 0.0
      }]
  
  def evaluate_model(self, test_dir):
    """Evaluate model on test dataset"""
    try:
      test_datagen = ImageDataGenerator(rescale=1./255)
      
      test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=self.img_size,
        batch_size=32,
        class_mode="categorical",
        shuffle=False
      )
      
      # Evaluate
      test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
      
      # Detailed predictions for classification report
      predictions = self.model.predict(test_generator, verbose=1)
      predicted_classes = np.argmax(predictions, axis=1)
      
      true_classes = test_generator.classes
      # class_labels = list(test_generator.class_indices.keys())
      
      # Predict on test set
      predictions = self.model.predict(test_generator, verbose=1)
      predicted_classes = np.argmax(predictions, axis=1)
      
      """This for evaluate_model.py to run with the provided test data avoiding errors"""
      # Build classification report using training class names
      from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
      from sklearn.utils.multiclass import unique_labels
      import seaborn as sns
      
      # Macro metrics
      macro_precision = precision_score(true_classes, predicted_classes, average="macro", zero_division=0)
      macro_recall = recall_score(true_classes, predicted_classes, average="macro", zero_division=0)
      macro_f1 = f1_score(true_classes, predicted_classes, average="macro", zero_division=0)
      
      # Classification report
      # report = classification_report(true_classes, predicted_classes, target_names=class_labels)
      labels_in_test = sorted(list(unique_labels(true_classes, predicted_classes)))
      label_names_in_test = [self.class_names[i] for i in labels_in_test]
      
      report = classification_report(
        true_classes,
        predicted_classes,
        labels=labels_in_test,
        target_names=label_names_in_test,
        zero_division=0
      )
      
      # Confusion matrix
      cm = confusion_matrix(true_classes, predicted_classes)
      plt.figure(figsize=(14, 12))
      sns.heatmap(cm, cmap="Blues")
      plt.title("Confusion Matrix")
      plt.xlabel("Predicted")
      plt.ylabel("Actual")
      plt.tight_layout()
      plt.savefig("confusion_matrix.png")
      plt.close()
      
      logging.info(f"Test Accuracy: {test_accuracy:.4f}")
      logging.info(f"Macro Precision: {macro_precision:.4f}")
      logging.info(f"Macro Recall: {macro_recall:.4f}")
      logging.info(f"Macro F1-score: {macro_f1:.4f}")
      logging.info(f"Classification Report:\n{report}")
      
      return {
              "accuracy": test_accuracy,
              "macro_precision": macro_precision,
              "macro_recall": macro_recall,
              "macro_f1": macro_f1,
              "report": report,
            }
      
    except Exception as e:
      logging.error(f"Error evaluating model: {str(e)}")
      return None, None
