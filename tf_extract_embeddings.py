import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

IMG_SIZE = (160,160)
MODEL_FILE = 'facenet_model.h5'

mtcnn = MTCNN()


def detect_and_crop_bgr(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = mtcnn.detect_faces(rgb)
    if not results:
        return None
    box = max(results, key=lambda r: r['box'][2] * r['box'][3])['box']
    x, y, w, h = box
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = x1 + w, y1 + h
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, IMG_SIZE)
    face = face.astype('float32') / 255.0
    return face


class EmbeddingExtractor:
    def __init__(self, model_file=MODEL_FILE):
        if not tf.io.gfile.exists(model_file):
            raise FileNotFoundError(f'Model file {model_file} not found. Train first.')
        self.model = tf.keras.models.load_model(model_file, compile=False)

    def get_embedding_from_path(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError('Image not found: ' + image_path)
        face = detect_and_crop_bgr(img)
        if face is None:
            raise ValueError('No face detected in image: ' + image_path)
        x = np.expand_dims(face, 0)
        emb = self.model.predict(x)
        return emb[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default=MODEL_FILE)
    args = parser.parse_args()
    ex = EmbeddingExtractor(args.model)
    e = ex.get_embedding_from_path(args.image)
    print('Embedding length:', len(e))
    print(e.tolist())

