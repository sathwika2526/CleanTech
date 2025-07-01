from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = load_model('vgg16.h5')

# Define class names (adjust if your dataset has different names)
class_names = ['biodegradable', 'recyclable', 'trash']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        
        img_file = request.files['image']
        if img_file.filename == '':
            return "No selected file"
        
        # Save the uploaded image temporarily
        img_path = os.path.join('static/uploads', img_file.filename)
        img_file.save(img_path)

        # Preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, image_path=img_path)

    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template("blog.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # process/store form if needed
        return render_template("blog-single.html", success=True)
    return render_template("blog-single.html", success=False)


if __name__ == '__main__':
    app.run(debug=True)

