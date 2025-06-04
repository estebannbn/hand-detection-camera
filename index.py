import cv2
import mediapipe as mp
import time

# Inicialización de MediaPipe para manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# iniciamos camara
cap = cv2.VideoCapture(0)

def hand_closed(landmarks):
    # Verifica si todos los dedos (excepto el pulgar) están doblados
    # y si la punta del dedo está más cerca de la palma que la articulación media
    fingers_folded = 0
    finger_tips = [8, 12, 16, 20]
    finger_dips = [6, 10, 14, 18]

    for tip, dip in zip(finger_tips, finger_dips):
        if landmarks[tip].y > landmarks[dip].y:
            fingers_folded += 1

    return fingers_folded == 4  # Todos los dedos doblados

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Volteamos la imagen para efecto espejo y la convertimos a RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibuja los puntos y conexiones de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtiene los puntos como lista
            h, w, _ = frame.shape
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            
            # Crea un rectángulo alrededor de la mano
            x_list = [x for x, y in coords]
            y_list = [y for x, y in coords]
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

            # Detectar puño cerrado
            if hand_closed(hand_landmarks.landmark):
                timestamp = int(time.time())
                filename = f"captura_mano_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"📸 Imagen capturada: {filename}")
                time.sleep(1)  # Evita múltiples capturas por segundo

    # Mostrar ventana
    cv2.imshow("Detección de Mano", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
