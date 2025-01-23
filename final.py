import firebase_admin 
from firebase_admin import credentials, firestore, storage
import cv2
import numpy as np
from mtcnn import MTCNN
from datetime import datetime
from dotenv import load_dotenv
import scipy.spatial
import os
from tensorflow.keras.models import load_model

load_dotenv()

firebase_cred_path = os.getenv('FIREBASE_ADMIN_SDK_PATH')
firebase_bucket_name = os.getenv('FIREBASE_STORAGE_BUCKET')

cred = credentials.Certificate("adminsdk.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': firebase_bucket_name
})
db = firestore.client()
bucket = storage.bucket()

facenet_model = load_model('models/facenet_keras.h5')

detector = MTCNN()

def preprocess_image(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_embedding(face_image):
    preprocessed_image = preprocess_image(face_image)
    embedding = facenet_model.predict(preprocessed_image)
    return embedding

def fetch_embeddings_from_firestore():
    docs = db.collection('face_embeddings').stream()
    embeddings = {}
    for doc in docs:
        embeddings[doc.id] = np.array(doc.to_dict()['embedding'])
    return embeddings

def mark_attendance_in_firestore(name):
    doc_ref = db.collection('attendance').document(name)
    now = datetime.now().strftime('%H:%M:%S')
    doc_ref.set({'last_seen': now}, merge=True)

known_embeddings = fetch_embeddings_from_firestore()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faces_cur_frame = detector.detect_faces(imgS)

    for face in faces_cur_frame:
        x, y, w, h = face['box']
        face_image = imgS[y:y+h, x:x+w]

        embedding = generate_embedding(face_image)

        min_dist = float("inf")
        matched_name = None

        for name, known_embed in known_embeddings.items():
            dist = scipy.spatial.distance.euclidean(embedding, known_embed)
            if dist < min_dist:
                min_dist = dist
                matched_name = name

        if min_dist < 0.6:
            mark_attendance_in_firestore(matched_name)
            cv2.putText(img, matched_name, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "NOT RECOGNIZED", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    img = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()