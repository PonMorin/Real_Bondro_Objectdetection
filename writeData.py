<<<<<<< HEAD
import serial
import time
if __name__ == '__main__':
    # if connected via USB cable
    # ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1) #9600 is baud rate(must be same with that of NodeMCU)
    # if connected via serial Pin(RX, TX)
    ser = serial.Serial('/dev/ttyS0', 9600, timeout=1) #9600 is baud rate(must be same with that of NodeMCU)
    ser.flush()
while True:
        string = input("enter string:") #input from user
        string = string +"\n" #"\n" for line seperation
        string = string.encode('utf_8')
        ser.write(string) #sending over UART
        # line = ser.readline().decode('utf-8').rstrip()
        # print("received: ",line)
        time.sleep(1) #delay of 1 second
=======
# sender.py
import time
import serial

ser = serial.Serial(
  port='/dev/ttyS0', # Change this according to connection methods, e.g. /dev/ttyUSB0
  baudrate = 115200,
  parity=serial.PARITY_NONE,
  stopbits=serial.STOPBITS_ONE,
  bytesize=serial.EIGHTBITS,
  timeout=1
)

msg = ""
i = 0

while True:
    i+=1
    print("Counter {} - Hello from Raspberry Pi".format(i))
    ser.write('hello'.encode('utf-8'))
    time.sleep(2)
>>>>>>> 0f1638c795243ca7319a02115d05b1baaa160b53
