import tensorflow as tf
import cv2
import numpy as np

# Charger le modèle au format SavedModel
model = tf.saved_model.load("/home/iim-iot/Modèles/model.savedmodel")

# Afficher les signatures disponibles
print("Signatures du modèle : ", model.signatures)

# Charger les étiquettes (labels) à partir d'un fichier texte
class_names = open("/home/iim-iot/Modèles/labels.txt", "r").readlines()

# Initialiser la caméra
camera = cv2.VideoCapture(0)

while True:
    # Lire l'image de la caméra
    ret, image = camera.read()

    # Si l'image est lue correctement
    if not ret:
        print("Erreur de lecture de l'image.")
        break

    # Redimensionner l'image à 224x224 pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Afficher l'image dans une fenêtre
    cv2.imshow("Webcam Image", image_resized)

    # Prétraiter l'image pour le modèle
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)  # Forme (1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1  # Normalisation des pixels dans la plage [-1, 1]

    # Prédiction avec le modèle
    predictions = model.signatures["serving_default"](tf.constant(image_array))  # Utiliser la signature du modèle

    # # Vérifier le nom des clés dans predictions
    # print(predictions)  # Affichez le contenu pour voir les clés disponibles

    # Modifiez "dense" en fonction de la sortie correcte
    prediction = predictions["sequential_3"].numpy()  # Changez "output_0" selon ce que vous trouvez
    index = np.argmax(prediction)  # Trouver l'index de la classe avec la probabilité maximale
    class_name = class_names[index].strip()  # Récupérer le nom de la classe
    confidence_score = prediction[0][index]  # Récupérer le score de confiance

    # Afficher la prédiction et le score de confiance
    print(f"Classe: {class_name}")
    print(f"Score de confiance: {np.round(confidence_score * 100, 2)}%")

    # Vérifier si l'utilisateur a appuyé sur la touche 'ESC' pour quitter
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # Code ASCII pour 'ESC'
        break

# Libérer la caméra et fermer toutes les fenêtres OpenCV
camera.release()
cv2.destroyAllWindows()