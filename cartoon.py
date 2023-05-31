from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/cartoonify', methods=['POST'])
def cartoonify():
    # Get the uploaded image file from the request
    image_file = request.files['image']

    # Read the image file into a numpy array
    image_np = np.fromfile(image_file, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while preserving edges
    gray_blur = cv2.bilateralFilter(gray, 7, 75, 75)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Convert edges back to color
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Apply cartoonifying effect by bitwise_and between the original image and edges
    cartoon = cv2.bitwise_and(image, edges)

    # Convert the cartoon image back to byte array
    _, cartoon_np = cv2.imencode('.jpg', cartoon)
    cartoon_bytes = cartoon_np.tobytes()

    # Return the cartoonified image as response
    return jsonify({'cartoon_image': cartoon_bytes.decode('latin1')}), 200


if __name__ == '__main__':
    app.run(debug=True)
