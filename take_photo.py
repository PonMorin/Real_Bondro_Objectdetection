import tensorflow as tf
import numpy as np
import cv2
import time

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./model/tf_bottle_model2.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assuming you have a labels.txt file with the class names
with open("model/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


def takePicture():
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        result, image = cam.read()
        if result:
            cv2.imwrite("Bottle.png", image)
        else:
            print("No image detected. Please! try again")

# Function to make predictions
def classify_image(image):
    # Ensure input data is in the correct shape and type
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    
    # Set the input tensor for the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor (prediction)
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Squeeze removes single-dimensional entries from the shape of the array
    return np.squeeze(predictions)

def prediction():
    # Read an image from file
    image_path = 'Bottle.png'  # Replace with your image path
    image = cv2.imread(image_path)

    # Preprocess the image to fit the model input
    img_resized = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))

    # Normalize the image if your model expects normalization
    img_resized = (img_resized.astype(np.float32) / 127.5) - 1  # Adjust this depending on your model's needs

    # Make predictions
    prediction = classify_image(img_resized)

    # Get the index of the class with the highest probability
    predicted_class = np.argmax(prediction)

    # Get the confidence score of the prediction
    confidence_score = np.max(prediction) * 100

    # Print the prediction and confidence score
    print(f"Prediction: {class_names[predicted_class]} ({confidence_score:.2f}%)")

    # # Optionally, display the image with prediction info
    # cv2.putText(image, f"Prediction: {class_names[predicted_class]} ({confidence_score:.2f}%)", 
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imshow('Image Classification', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    while True:
        start = time.time()
        userInput = int(input("Enter yes:"))
        if userInput ==  1:
            takePicture()
            prediction()
            end = time.time()
            print("The time of execution of above program is :",
            (end-start) * 10**3, "ms")

if __name__ == '__main__':
    main()