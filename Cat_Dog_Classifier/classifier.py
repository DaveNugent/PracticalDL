import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import math
import numpy as np
from tensorflow.keras.models import load_model

TRAIN_DATA_DIR = 'data/train_data/'
VALIDATION_DATA_DIR = 'data/val_data'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMAGE_HEIGHT = 224, 224
BATCH_SIZE = 64

def model_maker():
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMAGE_HEIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    input = Input(shape=(IMG_WIDTH, IMAGE_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D() (custom_model)
    custom_model = Dense(64, activation='relu') (custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax') (custom_model)
    return Model(inputs=input, outputs=predictions)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR, target_size=(IMG_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE, shuffle=True, seed=12345, class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(VALIDATION_DATA_DIR, target_size=(IMG_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE, shuffle=False, class_mode='categorical')

model = model_maker()
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['acc'])
num_steps = math.ceil(float(TRAIN_SAMPLES/BATCH_SIZE))
model.fit_generator(train_generator, steps_per_epoch = num_steps, epochs=10, validation_data=validation_generator, validation_steps=num_steps)

model.save('model.h5')
