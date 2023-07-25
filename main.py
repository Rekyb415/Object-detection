import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from mnist import load_mnist
from multi_layer_net import MultiLayerNetKeras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

input_shape = (28, 28, 1)  # Input shape for MNIST images
num_classes = 10  # Number of output classes for MNIST digits
model = MultiLayerNetKeras(input_shape=input_shape, num_classes=num_classes)
model.load_model("saved_model.h5")

# Assuming you have a list of class labels for reference
class_labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Replace with your actual class labels

def preprocess_image(image):
    img = image.convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape((1, 28, 28, 1))
    img_array = img_array.astype('float32')
    img_array /= 255.0  # Normalize the pixel values to [0, 1]
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("Error processing")
        return jsonify({"error": "No image found in request"}), 400

    image_file = request.files['image']
    if not image_file:
        print("Error processing")
        return jsonify({"error": "Invalid image"}), 400

    try:
        image = Image.open(io.BytesIO(image_file.read()))
        processed_image = preprocess_image(image)
        predicted_probs = model.predict(processed_image)
        print(predicted_probs)
        predicted_class = int(np.squeeze(np.argmax(predicted_probs, axis=-1)))
        predicted_label = class_labels[predicted_class]
        print(predicted_label)
        
        # Lưu ảnh vào thư mục "image" trên server
        save_image_path = os.path.join("image", image_file.filename)
        image.save(save_image_path)
        
        return jsonify({"predicted_class": predicted_class, "predicted_label": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
