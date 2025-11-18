from flask import Flask #Used to create the web application instance.
import request #: Provides access to the incoming request data from the client.
import jsonify # Converts Python data structures (dictionaries in this case) into JSON format for sending responses to the client.
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
loaded_model = tf.keras.models.load_model('cnn_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_data = file.read()  
    testing_image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    gray_image = cv2.cvtColor(testing_image, cv2.COLOR_BGR2GRAY)

    th, im = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)

    resized_image = cv2.resize(im, (28, 28))

    reshaped_image = resized_image.reshape(-1, 28, 28, 1)
    

    arr_image = np.array(reshaped_image, dtype='float32')

    

    prediction = loaded_model.predict(arr_image)  
    predicted_class = np.argmax(prediction)



    encoded_image_data = base64.b64encode(testing_image).decode('utf-8')

    response_json = {'image_data': encoded_image_data, 'predicted_class': int(predicted_class)}

    return jsonify(response_json)


if __name__ == '__main__':
    app.run(debug=True)

