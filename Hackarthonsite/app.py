from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app, support_credentials=True)

# Load your trained model
model = load_model('./predictemotionmodel.keras')

def preprocess_image(image):
    # Convert the image to RGB
    image = image.convert('RGB')
    # Resize the image
    image = image.resize((150, 150))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the image
    image_array = image_array / 255.0
    # Reshape the image to add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/predict', methods=['POST'])
@cross_origin(supports_credentials=True)
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    # Convert the PIL Image to a numpy array
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    # Assuming your model outputs a binary classification for 'Happy' and 'Sad'
    print(prediction[0][0])
    if prediction[0][0] == 1:
        emotion = 'Happy'
    else:
        emotion = 'Sad'

    return jsonify({'prediction': emotion})

if __name__ == '__main__':
    app.run(debug=True)
