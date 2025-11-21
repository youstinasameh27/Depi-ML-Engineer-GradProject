import os
import random
import glob
import numpy as np
from mtcnn import MTCNN
import cv2
import tensorflow as tf

# -------- CONFIG --------
IMG_SIZE = (160, 160)
EMBEDDING_SIZE = 128
BATCH_TRIPLETS = 16  # number of triplets per batch -> effective batch size = 3*BATCH_TRIPLETS
EPOCHS = 8
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
MODEL_OUT = 'facenet_model.h5'
CHECKPOINT_DIR = 'checkpoints'
MARGIN = 0.2

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------- Utilities --------
mtcnn = MTCNN()


def detect_and_crop(img):
    # img: BGR numpy array from cv2
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mtcnn.detect_faces(rgb)
    if not results:
        return None
    # choose largest box
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


def build_indexed_dataset(split_dir):
    # returns dict: classname -> list of image paths
    classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    data = {}
    for c in classes:
        files = glob.glob(os.path.join(split_dir, c, '*.jpg')) + glob.glob(os.path.join(split_dir, c, '*.png'))
        if files:
            data[c] = files
    return data


def triplet_generator(class_images, batch_triplets=BATCH_TRIPLETS):
    # class_images: dict class -> list of image paths
    classes = list(class_images.keys())
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError('Need at least two classes to form triplets')

    while True:
        anchors, positives, negatives = [], [], []
        for _ in range(batch_triplets):
            # pick anchor class and image
            anchor_cls = random.choice(classes)
            pos_cls = anchor_cls
            neg_cls = random.choice(classes)
            while neg_cls == anchor_cls:
                neg_cls = random.choice(classes)

            a_files = class_images[anchor_cls]
            p_files = class_images[pos_cls]
            n_files = class_images[neg_cls]

            if len(a_files) < 2:
                # cannot form anchor-positive from this class; pick another
                continue

            a_path = random.choice(a_files)
            p_path = random.choice(p_files)
            while p_path == a_path and len(p_files) > 1:
                p_path = random.choice(p_files)
            n_path = random.choice(n_files)

            # load and preprocess
            a_img = cv2.imread(a_path)
            p_img = cv2.imread(p_path)
            n_img = cv2.imread(n_path)
            if a_img is None or p_img is None or n_img is None:
                continue
            a_face = detect_and_crop(a_img)
            p_face = detect_and_crop(p_img)
            n_face = detect_and_crop(n_img)
            if a_face is None or p_face is None or n_face is None:
                continue

            anchors.append(a_face)
            positives.append(p_face)
            negatives.append(n_face)

        if len(anchors) == 0:
            continue

        A = np.stack(anchors)
        P = np.stack(positives)
        N = np.stack(negatives)
        # concatenate along batch axis
        X = np.concatenate([A, P, N], axis=0)
        # y is not used; return zeros placeholder
        yield X, np.zeros((X.shape[0],))


# -------- Model --------

def make_model(input_shape=(160, 160, 3), embedding_size=EMBEDDING_SIZE):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0)(inp)  # already normalized but keep layer
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(embedding_size)(x)
    out = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name='embedding')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


# triplet loss utilities

def pairwise_distance(emb):
    dot = tf.matmul(emb, tf.transpose(emb))
    sq = tf.linalg.diag_part(dot)
    dist = tf.expand_dims(sq, 1) - 2.0 * dot + tf.expand_dims(sq, 0)
    dist = tf.maximum(dist, 0.0)
    dist = tf.sqrt(dist + 1e-16)
    return dist


def batch_all_triplet_loss(labels_unused, embeddings, margin=MARGIN):
    # embeddings: (batch, dim)
    # We assume embeddings come in order [A...P...N...] where each of A,P,N has same batch size
    b = tf.shape(embeddings)[0]
    # split into thirds
    third = b // 3
    anc = embeddings[:third]
    pos = embeddings[third:2*third]
    neg = embeddings[2*third:3*third]
    # compute distances
    pos_dist = tf.reduce_sum(tf.square(anc - pos), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anc - neg), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss


# -------- Training --------

def train():
    train_index = build_indexed_dataset(TRAIN_DIR)
    val_index = build_indexed_dataset(VAL_DIR)

    if not train_index:
        raise ValueError('No training data found in ' + TRAIN_DIR)

    model = make_model()
    optimizer = tf.keras.optimizers.Adam(1e-4)

    # checkpoints
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'facenet_epoch_{epoch}.h5')

    train_gen = triplet_generator(train_index, batch_triplets=BATCH_TRIPLETS)
    val_gen = triplet_generator(val_index, batch_triplets=BATCH_TRIPLETS) if val_index else None

    steps_per_epoch = max(1, sum(len(v) for v in train_index.values()) // (BATCH_TRIPLETS * 3))
    val_steps = max(1, sum(len(v) for v in val_index.values()) // (BATCH_TRIPLETS * 3)) if val_index else 0

    print('Start training: steps_per_epoch=', steps_per_epoch, 'val_steps=', val_steps)

    for epoch in range(EPOCHS):
        print('Epoch', epoch+1, '/', EPOCHS)
        # training loop
        total_loss = 0.0
        for step in range(steps_per_epoch):
            X, _ = next(train_gen)
            with tf.GradientTape() as tape:
                embeddings = model(X, training=True)
                loss = batch_all_triplet_loss(None, embeddings)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += float(loss)
            if (step + 1) % 10 == 0:
                print(f'  step {step+1}/{steps_per_epoch} loss={float(loss):.4f}')
        avg_loss = total_loss / max(1, steps_per_epoch)
        print('  train loss:', avg_loss)

        # validation
        if val_gen and val_steps > 0:
            total_vloss = 0.0
            for vstep in range(val_steps):
                VX, _ = next(val_gen)
                vemb = model(VX, training=False)
                vloss = batch_all_triplet_loss(None, vemb)
                total_vloss += float(vloss)
            print('  val loss:', total_vloss / val_steps)

        # save checkpoint
        model.save(ckpt_path.format(epoch=epoch+1))

    # final save
    model.save(MODEL_OUT)
    print('Saved final model to', MODEL_OUT)


if __name__ == '__main__':
    train()


# File: tf_extract_embeddings.py
"""
Load facenet_model.h5 and extract embeddings for an image.
This file assumes model input normalization is the same as training (0-1 RGB).
"""
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

