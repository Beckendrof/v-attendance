import firebase_admin
from firebase_admin import credentials, firestore
import cv2
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv

facenet_model = load_model('facenet_keras.h5')
print("FaceNet model loaded.")

load_dotenv()

firebase_cred_path = os.getenv('FIREBASE_ADMIN_SDK_PATH')

cred = credentials.Certificate(firebase_cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

def preprocess_image(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_embedding(face_image):
    preprocessed_image = preprocess_image(face_image)
    embedding = facenet_model.predict(preprocessed_image)
    return embedding[0]

def upload_face_embeddings_to_firestore(local_path):
    embeddings = {}
    for image_file in os.listdir(local_path):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(f"{local_path}/{image_file}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = generate_embedding(img_rgb)
            name = os.path.splitext(image_file)[0]
            db.collection('face_embeddings').document(name).set({'embedding': embedding.tolist()})
            print(f"Uploaded embedding for {name} to Firestore.")

upload_face_embeddings_to_firestore('Database')