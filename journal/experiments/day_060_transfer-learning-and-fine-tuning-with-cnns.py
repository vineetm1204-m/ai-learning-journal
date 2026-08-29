import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. Load & preprocess CIFAR-10 (resize to 96x96 for MobileNetV2)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Resize to 96x96 (MobileNetV2 expects >= 32x32, 96 is a good trade-off)
def resize_images(images, size=(96, 96)):
    return tf.image.resize(images, size).numpy()

x_train = resize_images(x_train)
x_test = resize_images(x_test)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Validation split
val_split = 0.1
val_size = int(len(x_train) * val_split)
x_val, y_val = x_train[:val_size], y_train[:val_size]
x_train, y_train = x_train[val_size:], y_train[val_size:]

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# 2. Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# 3. Build transfer learning model
def build_model(trainable_base=False, fine_tune_at=100):
    """Build model with optional fine-tuning."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96, 96, 3)
    )
    base_model.trainable = trainable_base
    
    if trainable_base:
        # Freeze all layers before `fine_tune_at`
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    
    inputs = layers.Input(shape=(96, 96, 3))
    x = data_augmentation(inputs)
    x = layers.Lambda(lambda img: tf.keras.applications.mobilenet_v2.preprocess_input(img * 255.0))(x)
    x = base_model(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

# 4. Phase 1: Transfer learning (frozen base)
print("\n" + "="*60)
print("PHASE 1: TRANSFER LEARNING (frozen base)")
print("="*60)

model_tl, base_model_tl = build_model(trainable_base=False)
model_tl.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_tl = model_tl.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=64,
    callbacks=[
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ],
    verbose=1
)

# 5. Phase 2: Fine-tuning (unfreeze top layers)
print("\n" + "="*60)
print("PHASE 2: FINE-TUNING (unfreeze top layers)")
print("="*60)

model_ft, base_model_ft = build_model(trainable_base=True, fine_tune_at=100)
# Copy weights from transfer learning phase
model_ft.set_weights(model_tl.get_weights())

model_ft.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model_ft.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ],
    verbose=1
)

# 6. Evaluation
print("\n" + "="*60)
print("EVALUATION ON TEST SET")
print("="*60)

tl_loss, tl_acc = model_tl.evaluate(x_test, y_test, verbose=0)
ft_loss, ft_acc = model_ft.evaluate(x_test, y_test, verbose=0)

print(f"Transfer Learning - Test Acc: {tl_acc:.4f}, Loss: {tl_loss:.4f}")
print(f"Fine-Tuned        - Test Acc: {ft_acc:.4f}, Loss: {ft_loss:.4f}")
print(f"Improvement: {ft_acc - tl_acc:+.4f} accuracy points")

# 7. Plot training history
def plot_history(hist1, hist2, title="Training History"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(hist1.history['accuracy'], label='TL Train')
    axes[0].plot(hist1.history['val_accuracy'], label='TL Val')
    axes[0].plot([len(hist1.history['accuracy']) + i for i in range(len(hist2.history['accuracy']))], 
                 hist2.history['accuracy'], label='FT Train')
    axes[0].plot([len(hist1.history['val_accuracy']) + i for i in range(len(hist2.history['val_accuracy']))], 
                 hist2.history['val_accuracy'], label='FT Val')
    axes[0].axvline(len(hist1.history['accuracy']) - 1, color='gray', linestyle='--', label='Fine-tune start')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(hist1.history['loss'], label='TL Train')
    axes[1].plot(hist1.history['val_loss'], label='TL Val')
    axes[1].plot([len(hist1.history['loss']) + i for i in range(len(hist2.history['loss']))], 
                 hist2.history['loss'], label='FT Train')
    axes[1].plot([len(hist1.history['val_loss']) + i for i in range(len(hist2.history['val_loss']))], 
                 hist2.history['val_loss'], label='FT Val')
    axes[1].axvline(len(hist1.history['loss']) - 1, color='gray', linestyle='--', label='Fine-tune start')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('day60_transfer_learning_results.png', dpi=150)
    print("\nPlot saved to 'day60_transfer_learning_results.png'")
    plt.close()

plot_history(history_tl, history_ft)

# 8. Parameter analysis
print("\n" + "="*60)
print("PARAMETER ANALYSIS")
print("="*60)

total_params = model_ft.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model_ft.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {non_trainable_params:,}")
print(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")

# 9. Class-wise accuracy (fine-tuned model)
from sklearn.metrics import classification_report
y_pred = model_ft.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print("\nClassification Report (Fine-tuned model):")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

print("\n" + "="*60)
print("EXPERIMENT COMPLETE - Day 60: Transfer Learning & Fine-tuning")
print("="*60)