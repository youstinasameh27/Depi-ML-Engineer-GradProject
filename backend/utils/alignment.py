import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def align_face(image, face_box):
    """
    محاذاة وقص الوجه من الصورة مع تحسينات
    """
    try:
        x, y, w, h = face_box
        
        # إضافة padding
        padding = int(min(w, h) * 0.2)
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # قص الوجه
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # تغيير الحجم إلى 160x160 (حجم FaceNet القياسي)
        face_resized = cv2.resize(face, (160, 160))
        
        return face_resized
        
    except Exception as e:
        print(f"Error in face alignment: {e}")
        return None

def align_face_with_landmarks(image):
    """
    محاذاة متقدمة باستخدام نقاط الوجه (Landmarks)
    """
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_image)
        
        if len(detections) == 0:
            return None
        
        detection = detections[0]
        x, y, w, h = detection['box']
        keypoints = detection['keypoints']
        
        # استخراج نقاط العيون
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # حساب زاوية الدوران
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # تدوير الصورة
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)
        
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # قص الوجه المحاذى
        x, y = abs(x), abs(y)
        face = aligned[y:y+h, x:x+w]
        
        if face.size == 0:
            return None
        
        face_resized = cv2.resize(face, (160, 160))
        
        return face_resized
        
    except Exception as e:
        print(f"Error in advanced alignment: {e}")
        return None

def preprocess_face(face_image):
    """
    معالجة مسبقة للوجه (Preprocessing)
    """
    try:
        # تطبيع الإضاءة
        face_normalized = cv2.normalize(face_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # تحسين التباين
        lab = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        face_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return face_enhanced
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return face_image