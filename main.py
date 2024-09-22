import tensorflow as tf
import numpy as np
import cv2

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/tf_bottle_model2.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assuming you have a labels.txt file with the class names
with open("model/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

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

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the image to the input size of the model
    img_resized = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))

    # Normalize the image if your model expects normalization (adjust the range as needed)
    img_resized = (img_resized.astype(np.float32) / 127.5) - 1  # This depends on the model's preprocessing

    # Make predictions
    prediction = classify_image(img_resized)

    # Get the index of the class with the highest probability
    predicted_class = np.argmax(prediction)

    # Get the confidence score of the prediction
    confidence_score = np.max(prediction) * 100

    # Draw the prediction and confidence score on the frame
    cv2.putText(frame, f"Prediction: {class_names[predicted_class]} ({confidence_score:.2f}%)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-time Classification', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
