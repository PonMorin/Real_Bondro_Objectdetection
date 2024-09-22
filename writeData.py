'''
UART communication on Raspberry Pi using Python
http://www.electronicwings.com
'''
import serial
from time import sleep

# Initialize the serial port for communication
ser = serial.Serial("/dev/ttyS0", 9600)  # Adjust the port and baud rate if needed

while True:
    # Data to send over serial (as an example)
    data_to_send = "Hello from Raspberry Pi"

    # Transmit the data serially
    ser.write(data_to_send.encode())  # Convert string to bytes and send

    # Print confirmation message
    print(f"Sent: {data_to_send}")

    # Wait before sending the next message (optional)
    sleep(1)  # Adjust the delay as per requirement
