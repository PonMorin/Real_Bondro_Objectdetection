import tensorflow as tf
import numpy as np
import cv2
import time
import serial

# Set up serial communication
ser = serial.Serial ("/dev/ttyS0", 9600)    #Open port with baud rate
ser.flush()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./model/bondro_mark3/bondro_modelV3.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assuming you have a labels.txt file with the class names
with open("./model/bondro_mark3/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def takePicture():
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    result, image = cam.read()
    if result:
        cv2.imwrite("Bottle.png", image)
    else:
        print("No image detected. Please try again.")

# Function to make predictions
def classify_image(image):
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return np.squeeze(predictions)

def prediction():
    image_path = 'Bottle.png'
    image = cv2.imread(image_path)
    img_resized = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    img_resized = (img_resized.astype(np.float32) / 127.5) - 1

    prediction = classify_image(img_resized)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction) * 100

    # print(f"Prediction: {class_names[predicted_class]} ({confidence_score:.2f}%)")
    
    return class_names[predicted_class], confidence_score

def read_serial():
    if ser.inWaiting() > 0:
        data = ser.readline().decode('utf-8').rstrip()
        print(f"Data from ESP32: {data}")
        return data
    return None

def write_serial(message):
    # ser.write(message.encode('utf-8'))
    message = message +"\n" #"\n" for line seperation
    message = message.encode('utf_8')
    ser.write(message)
    print(f"Sent to ESP32: {message}")

def main():
    while True:
        esp_data = read_serial()
        if esp_data == '1':
            print(f"Response from ESP32: {esp_data}")
            takePicture()
            label, confidence = prediction()
            print("label", label)
            if label == '0 Bottle':
                # Send the result to ESP32
                write_serial('True')
                time.sleep(4)
            elif label == '1 No_Bottle':
                write_serial('False')
                time.sleep(4)
        # elif esp_data == '0' and check == 1:
        #     print("End")
        #     check = 0
        else:
            print("Nothing")
            time.sleep(1)

if __name__ == '__main__':
    main()
