# REF https://www.kaggle.com/code/mo7amedsabry/ecg-model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

folder_path = 'datasets'

from tensorflow.keras import layers, models, callbacks
for dirname, _, filenames in os.walk('datasets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def load_data():
    normal = pd.read_csv(f"{folder_path}/ptbdb_normal.csv", header=None)
    abnormal = pd.read_csv(f"{folder_path}/ptbdb_abnormal.csv", header=None)
    
    mitbih_train = pd.read_csv(f"{folder_path}/mitbih_train.csv", header=None)
    mitbih_test = pd.read_csv(f"{folder_path}//mitbih_test.csv", header=None)
    
    return normal, abnormal, mitbih_train, mitbih_test

normal, abnormal, mitbih_train, mitbih_test = load_data()

def plot_sample_ecgs(normal, abnormal, num_samples=3):
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    for i in range(num_samples):
        plt.plot(normal.iloc[i, :-1], label=f'Sample {i+1}')
    plt.title('Normal ECG Samples')
    plt.xlabel('Time steps')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i in range(num_samples):
        plt.plot(abnormal.iloc[i, :-1], label=f'Sample {i+1}')
    plt.title('Abnormal ECG Samples')
    plt.xlabel('Time steps')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_sample_ecgs(normal, abnormal)
def load_and_balance_data(normal_path, abnormal_path, train_size=0.8, val_test_split=0.5, random_state=42):
    """
    Load and balance ECG data with train/val/test splits (80/10/10)
    
    Args:
        normal_path: Path to normal ECG data
        abnormal_path: Path to abnormal ECG data
        train_size: Ratio of data for training (default 0.8)
        val_test_split: Ratio of remaining data to allocate to validation (default 0.5)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, class_weights
    """
    normal = pd.read_csv(normal_path, header=None)
    abnormal = pd.read_csv(abnormal_path, header=None)
    
    normal['label'] = 0
    abnormal['label'] = 1
    
    data = pd.concat([normal, abnormal], axis=0)
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    X = X.reshape(X.shape[0], X.shape[1], 1)
        class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = {
        0: total_samples / (2 * class_counts[0]),  # Normal
        1: total_samples / (2 * class_counts[1])   # Abnormal
    }
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=train_size, 
        random_state=random_state, 
        stratify=y
    )
        # Second split: val vs test from remaining data
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_test_split,  
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)} ({len(X_train)/len(X):.1%})")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X):.1%})")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(X):.1%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights

X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_and_balance_data(
    f"{folder_path}/ptbdb_normal.csv",
    f"{folder_path}/ptbdb_abnormal.csv"
)

print(f"Class weights: {class_weights}")
print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

def plot_class_distribution(y_train, y_val, y_test):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.countplot(x=y_train, ax=ax[0])
    ax[0].set_title(f'Train Set\n({len(y_train)} samples)')
    ax[0].set_xticklabels(['Normal', 'Abnormal'])
    
    sns.countplot(x=y_val, ax=ax[1])
    ax[1].set_title(f'Validation Set\n({len(y_val)} samples)')
    ax[1].set_xticklabels(['Normal', 'Abnormal'])
    
    sns.countplot(x=y_test, ax=ax[2])
    ax[2].set_title(f'Test Set\n({len(y_test)} samples)')
    ax[2].set_xticklabels(['Normal', 'Abnormal'])
    
    plt.tight_layout()
    plt.show()

plot_class_distribution(y_train, y_val, y_test)

from tensorflow.keras import regularizers

def build_model(input_shape):
    model = models.Sequential([
        # First Conv Block
        layers.Conv1D(64, kernel_size=15, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
        
        # Second Conv Block
        layers.Conv1D(128, kernel_size=11, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
                # Third Conv Block
        layers.Conv1D(256, kernel_size=7, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling1D(),
        
        layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    
    return model
print(X_train.shape[1])
print(X_train.shape[2])

input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape)
model.summary()

early_stopping = callbacks.EarlyStopping(
    monitor='val_auc',  
    patience=15,
    mode='max',
    restore_best_weights=True)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_auc',
    save_best_only=True,
    mode='max')

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  
    batch_size=128,
    class_weight=class_weights,  
    callbacks=[early_stopping, model_checkpoint, reduce_lr],shuffle=True)

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_plot(model, X_test, y_test):
    # 1. Model Evaluation
    print("\nEvaluating model on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")
    
    # 2. Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # 3. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['Normal', 'Abnormal']))
    
    # 4. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    plt.figure(figsize=(15, 10))
    for i in range(6):
        idx = np.random.randint(0, len(X_test))
        ecg = X_test[idx].flatten()
        true_label = y_test[idx]
        pred_prob = y_pred[idx][0]
        
        plt.subplot(3, 2, i+1)
        plt.plot(ecg)
        plt.title(f"True: {'Abnormal' if true_label else 'Normal'}\n"
                 f"Predicted: {'Abnormal' if pred_prob > 0.5 else 'Normal'} "
                 f"({pred_prob:.2f})")
        plt.xlabel('Time steps')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

evaluate_and_plot(model, X_test, y_test)
model.save('ecg_model.h5')  
from tensorflow.keras.models import load_model
model = load_model('best_model.keras') 
model.summary()
sample_input = X_test[0:1]  
prediction = model.predict(sample_input)
print(f"Prediction: {'Abnormal' if prediction > 0.5 else 'Normal'} ({prediction[0][0]:.2f})")

evaluate_and_plot(model, X_test, y_test)
from tensorflow.keras.models import load_model
model = load_model('ecg_model.h5') 
model.summary()
sample_input = X_test[0:1]  
prediction = model.predict(sample_input)
print(f"Prediction: {'Abnormal' if prediction > 0.5 else 'Normal'} ({prediction[0][0]:.2f})")