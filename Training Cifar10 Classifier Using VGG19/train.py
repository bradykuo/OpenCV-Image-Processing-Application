import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

def create_model():
    # Create base model
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 32, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create complete model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_model():
    # Load and preprocess CIFAR10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Create model
    model = create_model()
    
    # Compile model with specified parameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Adam optimizer
        loss='categorical_crossentropy',       # CrossEntropyLoss
        metrics=['accuracy']                   # Track accuracy
    )
    
    # Create checkpoint callback to save weights during training
    checkpoint = ModelCheckpoint(
        'vgg19_weights.keras',  # Changed file extension to .keras
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train model for 30 epochs
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Save final model
    model.save('vgg19_final.keras')  # Changed file extension to .keras
    
    # Save training history
    np.savez('training_history.npz',
             acc=history.history['accuracy'],
             val_acc=history.history['val_accuracy'],
             loss=history.history['loss'],
             val_loss=history.history['val_loss'])
    
    # Convert accuracy to percentage
    acc = np.array(history.history['accuracy']) * 100
    val_acc = np.array(history.history['val_accuracy']) * 100
    
    # Create figure exactly as shown in the example
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training', color='blue')
    plt.plot(val_acc, label='Testing', color='orange')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('%')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training', color='blue')
    plt.plot(history.history['val_loss'], label='Testing', color='orange')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_model()