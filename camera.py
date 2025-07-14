import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import winsound  # For beep sound on Windows
import threading
import requests  # Import for sending HTTP requests
from datetime import datetime
import random  # Import the random module

model = AccidentDetectionModel("model.json", 'model_weights.keras')
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the API endpoint
API_ENDPOINT = "https://acidentedeviacao.xyz/api/report_emergecy1.php"




def send_emergency_report(camera_name, photo_path):
    emergency_id = random.randint(0, 999)  # Generate a random emergency_id
    data_atual = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    payload = {
        "emergency_id": emergency_id,
        "agency_name": "Policia",
        "agency_id": 1337,
        "case_severity": "Critical",
        "emergency_category": "Colisão",
        "dates": data_atual,
        "state": f"Bairro {camera_name}",
        "phone_number": "845694156",
        "address": camera_name,
        "name": f"Colisao {camera_name}",
        "status": "Pendente",
        "email": f"{camera_name.lower().replace(' ', '')}@gmail.com",
        "victim_id": 32535,
        "description": "Colisão entre duas viaturas"
    }

    files = {
        'photo': open(photo_path, 'rb')
    }

    try:
        response = requests.post(API_ENDPOINT, data=payload, files=files)
        if response.status_code == 200:
            print(f"Emergency reported successfully: {response.json()}")
        else:
            print(f"Failed to report emergency: {
                  response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"Error reporting emergency: {e}")


from datetime import datetime

def process_video(video_path, window_name):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Erro: {video_path}")
        return

    frame_interval = 3
    frame_count = 0
    last_detection_time = None
    detection_delay = 5  # segundos entre alertas para ignorar frames repetidos do mesmo acidente

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        roi = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (250, 250))
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

        if pred == "Accident" and prob[0][0] > 0.95:
            now = datetime.now()

            if not last_detection_time or (now - last_detection_time).total_seconds() > detection_delay:
                last_detection_time = now

                photo_path = f"accident_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                cv2.imwrite(photo_path, frame)
                send_emergency_report(window_name, photo_path)

                if os.name == 'nt':
                    winsound.Beep(1000, 500)

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {round(prob[0][0]*100, 2)}%", (20, 30), font, 1, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyWindow(window_name)


def start_application():
    video_files = [
        'Car2.mp4',

    ]

    threads = []

    for i, video_path in enumerate(video_files):
        file_name_without_ext = os.path.splitext(os.path.basename(video_path))[0]
        window_name = file_name_without_ext
        print(f"Processing {video_path}...")
        thread = threading.Thread(
            target=process_video, args=(video_path, window_name))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_application()
