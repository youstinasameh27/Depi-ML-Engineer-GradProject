from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pickle
from pathlib import Path
from utils.detection import detect_face
from utils.alignment import align_face
from utils.embedding import get_embedding
from utils.compare import train_knn, recognize_with_knn, find_duplicate

app = Flask(__name__)
CORS(app)

# Directories
EMBEDDINGS_DIR = Path(__file__).parent / 'embeddings'
EMBEDDINGS_DIR.mkdir(exist_ok=True)
DB_PATH = EMBEDDINGS_DIR / 'database.pkl'

USER_IMAGES_DIR = Path(__file__).parent / 'user_images'
USER_IMAGES_DIR.mkdir(exist_ok=True)

# Load or create database
def load_database():
    if DB_PATH.exists():
        with open(DB_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        # Create empty database
        db = {}
        with open(DB_PATH, 'wb') as f:
            pickle.dump(db, f)
        return db

def save_database(db):
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)

database = load_database()

def decode_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/register', methods=['POST'])
def register():
    """Register new user with 3 images"""
    try:
        data = request.json
        name = data.get('name')
        images_data = data.get('images')  # expects 3 images

        if not name or not images_data or len(images_data) != 3:
            return jsonify({'success': False, 'message': 'الاسم و3 صور مطلوبة'}), 400

        # Check for duplicates first
        for img_data in images_data:
            img = decode_image(img_data)
            face_box = detect_face(img)
            if face_box is None:
                continue
            aligned_face = align_face(img, face_box)
            embedding = get_embedding(aligned_face)
            dup_name, similarity = find_duplicate(embedding, database)
            if dup_name and similarity > 0.75:
                return jsonify({'success': False, 'message': f'هذا الوجه مسجل مسبقًا باسم: {dup_name}'}), 400

        # Save images and embeddings
        person_embeddings = []
        person_folder = USER_IMAGES_DIR / name
        person_folder.mkdir(exist_ok=True)

        for idx, img_data in enumerate(images_data):
            img = decode_image(img_data)
            face_box = detect_face(img)
            aligned_face = align_face(img, face_box)
            embedding = get_embedding(aligned_face)
            person_embeddings.append(embedding)

            cv2.imwrite(str(person_folder / f'{name}_{idx+1}.jpg'), aligned_face)

        database[name] = person_embeddings
        save_database(database)

        return jsonify({'success': True, 'message': f'تم تسجيل {name} بنجاح', 'registered_users': len(database)})

    except Exception as e:
        print(f"Error in registration: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'الصورة مطلوبة'}), 400

        if len(database) == 0:
            return jsonify({'success': False, 'message': 'لا يوجد مستخدمين مسجلين'}), 400

        img = decode_image(image_data)
        face_box = detect_face(img)
        aligned_face = align_face(img, face_box)
        embedding = get_embedding(aligned_face)

        name, confidence = recognize_with_knn(embedding, database)

        if confidence > 0.6:
            return jsonify({'success': True, 'recognized': True, 'name': name, 'confidence': float(confidence)})
        else:
            return jsonify({'success': True, 'recognized': False, 'message': 'الوجه غير معروف', 'best_match': name, 'confidence': float(confidence)})

    except Exception as e:
        print(f"Error in recognition: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/users', methods=['GET'])
def get_users():
    return jsonify({'success': True, 'users': list(database.keys()), 'count': len(database)})

@app.route('/delete/<name>', methods=['DELETE'])
def delete_user(name):
    if name in database:
        del database[name]
        save_database(database)
        user_folder = USER_IMAGES_DIR / name
        if user_folder.exists():
            for f in user_folder.iterdir():
                f.unlink()
            user_folder.rmdir()
        return jsonify({'success': True, 'message': f'تم حذف {name}'})
    return jsonify({'success': False, 'message': 'المستخدم غير موجود'}), 404

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'registered_users': len(database), 'model': 'FaceNet + KNN'})

if __name__ == '__main__':
    print(f"Database loaded with {len(database)} users")
    app.run(debug=True, host='0.0.0.0', port=5000)
