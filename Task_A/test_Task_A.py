
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model(model_path, val_path):
    # Load trained Keras model
    model = load_model(model_path)

    # Prepare the validation data generator
    datagen = ImageDataGenerator(rescale=1./255)
    val_gen = datagen.flow_from_directory(
        val_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    # Predict
    predictions = model.predict(val_gen, verbose=1)
    predicted_labels = (predictions > 0.5).astype(int).reshape(-1)
    true_labels = val_gen.classes

    # Metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Print metrics
    print("\n✅ Evaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # Optional: classification report and confusion matrix
    print("\n📊 Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=list(val_gen.class_indices.keys())))

    print("\n🧮 Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <model_path> <val_data_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    val_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        sys.exit(1)

    if not os.path.isdir(val_path):
        print(f"❌ Error: Validation directory not found at {val_path}")
        sys.exit(1)

    evaluate_model(model_path, val_path)
