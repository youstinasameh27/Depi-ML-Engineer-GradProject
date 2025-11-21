import cv2
import numpy as np
from mtcnn import MTCNN

# تهيئة MTCNN detector مرة واحدة
detector = MTCNN()

def detect_face(image):
    """
    اكتشاف الوجه في الصورة باستخدام MTCNN
    Returns: (x, y, w, h) أو None
    """
    try:
        # تحويل BGR إلى RGB (MTCNN يحتاج RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # اكتشاف الوجوه
        detections = detector.detect_faces(rgb_image)
        
        if len(detections) == 0:
            return None
        
        # إذا كان هناك أكثر من وجه، نأخذ الأكبر (الأعلى ثقة)
        if len(detections) > 1:
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # استخراج bounding box
        x, y, w, h = detections[0]['box']
        
        # التأكد من القيم موجبة
        x, y = abs(x), abs(y)
        
        return (x, y, w, h)
        
    except Exception as e:
        print(f"Error in MTCNN face detection: {e}")
        # Fallback إلى Haar Cascade
        return detect_face_haar(image)

def detect_face_haar(image):
    """
    Fallback: اكتشاف الوجه باستخدام Haar Cascade
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        return faces[0]
        
    except Exception as e:
        print(f"Error in Haar Cascade detection: {e}")
        return None

def detect_faces_multiple(image):
    """
    اكتشاف وجوه متعددة في الصورة
    Returns: list of (x, y, w, h)
    """
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_image)
        
        faces = []
        for detection in detections:
            x, y, w, h = detection['box']
            x, y = abs(x), abs(y)
            faces.append((x, y, w, h))
        
        return faces
        
    except Exception as e:
        print(f"Error in multiple face detection: {e}")
        return []