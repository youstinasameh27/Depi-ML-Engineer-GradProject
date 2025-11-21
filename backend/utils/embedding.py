import cv2
import numpy as np
from keras_facenet import FaceNet

# تهيئة FaceNet model مرة واحدة
print("Loading FaceNet model...")
facenet_model = FaceNet()
print("FaceNet model loaded successfully!")

def get_embedding(face_image):
    """
    استخراج embedding من صورة الوجه باستخدام FaceNet
    Returns: embedding vector (512 dimensions)
    """
    try:
        # التأكد من أن الصورة 160x160
        if face_image.shape[:2] != (160, 160):
            face_image = cv2.resize(face_image, (160, 160))
        
        # تحويل BGR إلى RGB
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # إضافة batch dimension
        face_pixels = np.expand_dims(rgb_face, axis=0)
        
        # استخراج embedding باستخدام FaceNet
        embedding = facenet_model.embeddings(face_pixels)
        
        # إرجاع embedding كـ 1D array
        return embedding[0]
        
    except Exception as e:
        print(f"Error in FaceNet embedding extraction: {e}")
        return None

def get_embeddings_batch(face_images):
    """
    استخراج embeddings لمجموعة من الوجوه دفعة واحدة (أسرع)
    """
    try:
        processed_faces = []
        
        for face_image in face_images:
            if face_image.shape[:2] != (160, 160):
                face_image = cv2.resize(face_image, (160, 160))
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            processed_faces.append(rgb_face)
        
        # تحويل إلى numpy array
        faces_array = np.array(processed_faces)
        
        # استخراج embeddings
        embeddings = facenet_model.embeddings(faces_array)
        
        return embeddings
        
    except Exception as e:
        print(f"Error in batch embedding extraction: {e}")
        return None

def normalize_embedding(embedding):
    """
    تطبيع الـ embedding
    """
    try:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    except Exception as e:
        print(f"Error in normalization: {e}")
        return embedding

def get_embedding_distance(embedding1, embedding2):
    """
    حساب المسافة الإقليدية بين embedding اثنين
    """
    try:
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance
    except Exception as e:
        print(f"Error in distance calculation: {e}")
        return float('inf')