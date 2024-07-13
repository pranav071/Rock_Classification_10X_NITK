import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Set the paths to your individual folders
granite_dir = 'granite1'
sandstone_dir = 'sandstone1'
limestone_dir = 'limestone1'
quartzite_dir = 'quartzite1'

# Define the input image size and batch size
img_size = (224, 224)
batch_size = 16

# Create data generator for training
train_datagen = CustomImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='.',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=[granite_dir, sandstone_dir, limestone_dir, quartzite_dir]
)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 20
steps_per_epoch = train_generator.samples // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs
)

# Evaluate the model on the training data
train_loss, train_accuracy = model.evaluate(train_generator)
print("Training Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
