import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
import shutil

labels = ["Thit", "Banh", "Nuoc ngot", "Keo"]
loaded_model = load_model('my_model.h5')
print("Model loaded successfully.")
# Đường dẫn đến thư mục chứa các tệp ảnh cần đánh giá
folder_path = "E:\\abc\\Data test"

# Danh sách các tệp ảnh trong thư mục
image_files = os.listdir(folder_path)

# Duyệt qua từng tệp ảnh
for file_name in image_files:
    file_path = os.path.join(folder_path, file_name)

    # Đọc và tiền xử lý file ảnh
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    preprocessed_img = img_resized / 255.0
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

    # Dự đoán nhãn của ảnh
    predictions = loaded_model.predict(preprocessed_img)
    predicted_label = labels[np.argmax(predictions)]

    # Đường dẫn tới thư mục đích (dựa trên nhãn dự đoán)
    destination_folder = os.path.join("E:\\New folder", predicted_label)

    # Kiểm tra xem thư mục đích đã tồn tại chưa, nếu chưa thì tạo mới
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Di chuyển tệp ảnh vào thư mục đích
    destination_path = os.path.join(destination_folder, file_name)
    shutil.move(file_path, destination_path)

    print("Moved image:", file_name, "to", destination_folder)