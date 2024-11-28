import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import mediapipe as mp
import time

# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Ouvre la caméra
cap = cv2.VideoCapture(0)

# Liste des noms des articulations des doigts
landmark_names = {
    0: "Pouce 1", 1: "Pouce 2", 2: "Pouce 3", 3: "Pouce 4", 4: "Pouce 5",
    5: "Index 1", 6: "Index 2", 7: "Index 3", 8: "Index 4", 9: "Majeur 1",
    10: "Majeur 2", 11: "Majeur 3", 12: "Majeur 4", 13: "Annulaire 1",
    14: "Annulaire 2", 15: "Annulaire 3", 16: "Annulaire 4", 17: "Auriculaire 1",
    18: "Auriculaire 2", 19: "Auriculaire 3", 20: "Auriculaire 4"
}

# Temps de la dernière impression
last_print_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur : Impossible de lire l'image.")
        break

    # Convertir l'image en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des mains
    results = hands.process(rgb_frame)

    # Si des mains sont détectées
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les landmarks de la main
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Vérifier si 0.5 secondes se sont écoulées depuis la dernière impression
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                # Afficher les coordonnées des landmarks avec des noms
                print("Positions des doigts :")
                for i, landmark in enumerate(hand_landmarks.landmark):
                    # Utiliser les noms définis dans le dictionnaire landmark_names
                    name = landmark_names.get(i, f"Point {i}")
                    print(f"{name}: x={landmark.x:.2f}, y={landmark.y:.2f}, z={landmark.z:.2f}")
                
                # Mettre à jour le temps de la dernière impression
                last_print_time = current_time

    # Afficher le flux vidéo avec les détections
    cv2.imshow("Hand Landmark Positions", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()