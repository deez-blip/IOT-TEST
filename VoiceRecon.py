import vosk
import os
import sys
import sounddevice as sd
import queue
import json
import RPi.GPIO as GPIO
import time
import threading

# Configuration des pins GPIO pour les LEDs et le moteur
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)  # LED verte
GPIO.setup(16, GPIO.OUT)  # LED bleue
servo_pin = 17  # Pin pour le signal du servo
GPIO.setup(servo_pin, GPIO.OUT)

# Initialisation PWM pour contrôler le servo
pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz pour le servo
pwm.start(0)

# Modèle Vosk
MODEL_PATH = "/home/iim-iot/Modèles/vosk-model-small-fr-0.22"
if not os.path.exists(MODEL_PATH):
    print(f"Erreur: Le modèle {MODEL_PATH} n'existe pas.")
    sys.exit(1)
model = vosk.Model(MODEL_PATH)

q = queue.Queue()

samplerate = 16000
rec = vosk.KaldiRecognizer(model, samplerate)

# Fonction pour allumer les LEDs
def allumer_led(couleur):
    if couleur == "vert":
        GPIO.output(21, GPIO.HIGH)  # Allume la LED verte
        time.sleep(2)
        GPIO.output(21, GPIO.LOW)  # Éteint la LED verte
    elif couleur == "bleu":
        GPIO.output(16, GPIO.HIGH)  # Allume la LED bleue
        time.sleep(2)
        GPIO.output(16, GPIO.LOW)  # Éteint la LED bleue

# ? pwm.ChangeDutyCycle(5) -> Environ 0° (ou angle minimum).
# ? pwm.ChangeDutyCycle(7.5) -> Environ 90° (position médiane).
# ? pwm.ChangeDutyCycle(10) -> Environ 180° (ou angle maximum).

# Fonction pour faire tourner le moteur (servo) pendant quelques secondes
def rotate_servo():
    pwm.ChangeDutyCycle(7)
    time.sleep(3)
    pwm.ChangeDutyCycle(0)  # Arrêter le signal PWM

# Callback pour récupérer l'audio depuis le micro
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Fonction de traitement de l'audio
def process_audio():
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_json = json.loads(result)
            print("Transcription: ", result_json['text'])
            if "vert" in result_json['text'].lower():
                allumer_led("vert")
            elif "bleu" in result_json['text'].lower():
                allumer_led("bleu")
            elif "moteur" in result_json['text'].lower():
                rotate_servo()

        else:
            partial_result = rec.PartialResult()
            partial_result_json = json.loads(partial_result)
            print("En cours de transcription: ", partial_result_json['partial'])

# Démarrer le thread pour traiter l'audio
audio_thread = threading.Thread(target=process_audio, daemon=True)
audio_thread.start()

# Fonction principale pour démarrer l'enregistrement
def transcribe():
    print("Enregistrement en cours, parlez...")
    with sd.InputStream(callback=callback, channels=1, samplerate=samplerate, dtype='int16', blocksize=1024, latency='low'):
        while True:
            pass

if __name__ == "__main__":
    try:
        transcribe()
    except KeyboardInterrupt:
        print("\nEnregistrement terminé.")
        pwm.stop()
        GPIO.cleanup()  # Nettoyage des GPIO