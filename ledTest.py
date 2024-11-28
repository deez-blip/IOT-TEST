import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO_PIN = 16
GPIO.setup(GPIO_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(GPIO_PIN, GPIO.HIGH)
        print("LED ON")
        time.sleep(1)
        
        GPIO.output(GPIO_PIN, GPIO.LOW)
        print("LED OFF")
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nArrêt du script")
    
finally:
    GPIO.cleanup()