import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers


def preprocess_data(train_data):
    processed_data = []

    for image in train_data:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224))
        image_normalized = image_resized / 255.0
        processed_data.append(image_normalized)

    processed_data = np.array(processed_data)
    return processed_data


def split_data(train_data, train_labels_encoded, test_size=0.1, random_state=42):
    train_data, test_data, train_labels_encoded, test_labels_encoded = train_test_split(
        train_data, train_labels_encoded, test_size=test_size, random_state=random_state
    )

    return train_data, test_data, train_labels_encoded, test_labels_encoded


folder_path = "E:\\New folder"
labels = ["Thit", "Banh", "Nuoc ngot", "Keo"]

train_data = []
train_labels = []

for label in labels:
    label_folder = os.path.join(folder_path, label)
    for file_name in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file_name)
        image = cv2.imread(file_path)
        image = cv2.resize(image, (224, 224))
        train_data.append(image)
        train_labels.append(label)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

train_data = preprocess_data(train_data)
train_data, test_data, train_labels_encoded, test_labels_encoded = split_data(train_data, train_labels_encoded, test_size=0.2, random_state=42)

num_classes = len(set(train_labels_encoded))
train_labels_encoded = to_categorical(train_labels_encoded, num_classes)
test_labels_encoded = to_categorical(test_labels_encoded, num_classes)

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow(train_data, train_labels_encoded, batch_size=32)
test_generator = datagen.flow(test_data, test_labels_encoded, batch_size=32)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.12), metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=test_generator)
#
# test_loss, test_acc = model.evaluate(test_generator)
# print('Test Accuracy:', test_acc)
#
# model.save('my_model.h5')
# print("Model saved successfully.")
# model = load_model('my_model.h5')

image_path = "E:\\abc\\Data test\\caphe.jpg"

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
preprocessed_img = img_resized / 255.0
preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

# Dự đoán nhãn của file ảnh
predictions = model.predict(preprocessed_img)
predicted_label = labels[np.argmax(predictions)]

print("Predicted Label:", predicted_label)
