from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

VALIDATION_DATA_DIR = 'data/val_data'
IMG_WIDTH, IMAGE_HEIGHT = 224, 224
BATCH_SIZE = 64

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = val_datagen.flow_from_directory(VALIDATION_DATA_DIR, target_size=(IMG_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE, shuffle=False, class_mode='categorical')

model = load_model('model.h5')

img_path = 'naddah.jpg'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)

expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img =preprocess_input(expanded_img_array)
prediction = model.predict(preprocessed_img)
print("\nprediction for Naddah")
print(prediction)
print(validation_generator.class_indices)

img_path = 'jakobi.jpg'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)

expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img =preprocess_input(expanded_img_array)
prediction = model.predict(preprocessed_img)
print("\nprediction for Jakobi")
print(prediction)
print(validation_generator.class_indices)

img_path = 'testCat.jpg'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)

expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img =preprocess_input(expanded_img_array)
prediction = model.predict(preprocessed_img)
print("\nprediction for test cat")
print(prediction)
print(validation_generator.class_indices)

img_path = 'testDog.jpg'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)

expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img =preprocess_input(expanded_img_array)
prediction = model.predict(preprocessed_img)
print("\nprediction for test dog")
print(prediction)
print(validation_generator.class_indices)