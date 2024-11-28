import cv2
import mediapipe as mp

# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Ouvre la caméra
cap = cv2.VideoCapture(0)

# Fonction pour vérifier si la main est ouverte
def is_hand_open(hand_landmarks):
    # Calculer les distances entre les extrémités des doigts
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calculer les distances en termes d'écartement des doigts (X et Y)
    thumb_index_dist = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
    index_middle_dist = abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y)
    middle_ring_dist = abs(middle_tip.x - ring_tip.x) + abs(middle_tip.y - ring_tip.y)
    ring_pinky_dist = abs(ring_tip.x - pinky_tip.x) + abs(ring_tip.y - pinky_tip.y)

    # Si les distances sont suffisamment grandes, c'est une main ouverte
    if thumb_index_dist > 0.05 and index_middle_dist > 0.05 and middle_ring_dist > 0.05 and ring_pinky_dist > 0.05:
        return True
    return False

# Fonction pour vérifier si l'index et le majeur sont levés
def is_index_and_middle_raised(hand_landmarks):
    # Vérifier si l'index et le majeur sont levés
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Calculer les distances pour l'index et le majeur par rapport à la paume
    index_dist = abs(index_tip.y - palm_base.y)
    middle_dist = abs(middle_tip.y - palm_base.y)

    # ! Vérifier si l'index et le majeur sont suffisamment éloignés pour être levés
    if index_dist < 0.04 and middle_dist < 0.04:
        return True
    return False

# Fonction pour vérifier si le pouce est levé
def is_thumb_raised(hand_landmarks):
    # Vérifier si le pouce est levé
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    thumb_dist = abs(thumb_tip.y - palm_base.y)
    
    # ! Le pouce doit être éloigné de la paume
    if thumb_dist < 0.04:
        return True
    return False

# Fonction pour vérifier si le pouce et le petit doigt sont levés
def is_thumb_and_pinky_raised(hand_landmarks):
    # Vérifier si le pouce et le petit doigt sont levés
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Calculer la distance entre le pouce et le petit doigt
    thumb_pinky_dist = abs(thumb_tip.x - pinky_tip.x) + abs(thumb_tip.y - pinky_tip.y)
    
    # ! Si la distance entre le pouce et le petit doigt est suffisamment grande
    if thumb_pinky_dist > 0.1:
        return True
    return False

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

            # Vérification des gestes
            if is_hand_open(hand_landmarks):
                print("Avancer")
            elif is_index_and_middle_raised(hand_landmarks):
                print("Reculer")
            elif is_thumb_raised(hand_landmarks):
                print("Aller à droite")
            elif is_thumb_and_pinky_raised(hand_landmarks):
                print("Aller à gauche")

    # Afficher le flux vidéo avec les détections
    cv2.imshow("Hand Gesture Detection", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()