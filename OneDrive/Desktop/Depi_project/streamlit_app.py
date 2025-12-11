import streamlit as st
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Setup
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.title("Face Recognition System")

# Load MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=10, device=device)

# Load pretrained FaceNet model
model = InceptionResnetV1(pretrained=None, classify=False).to(device)
model.load_state_dict(torch.load("best_facenet_model.pth", map_location=device), strict=False)
model.eval()

# ------------------------------
# Load known faces
# ------------------------------
known_embeddings = []
known_names = []

known_dir = "known_faces"
for person_name in os.listdir(known_dir):
    person_path = os.path.join(known_dir, person_name)
    if not os.path.isdir(person_path):
        continue
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            face_embedding = model(face.unsqueeze(0).to(device))
            known_embeddings.append(face_embedding.detach().cpu())
            known_names.append(person_name)

if len(known_embeddings) == 0:
    st.warning("No known faces loaded! Check your 'known_faces/' folder.")
else:
    known_embeddings = torch.cat(known_embeddings)

# ------------------------------
# Upload image
# ------------------------------
uploaded_file = st.file_uploader("Upload an image to recognize", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Detect face
    face = mtcnn(img)
    if face is None:
        st.error("No face detected in the uploaded image.")
    else:
        face_embedding = model(face.unsqueeze(0).to(device))

        # Compare with known embeddings
        sims = cosine_similarity(face_embedding.detach().cpu().numpy(), known_embeddings.numpy())
        best_idx = np.argmax(sims)
        best_score = sims[0][best_idx]

        if best_score > 0.7:  # threshold, can adjust
            st.success(f"Recognized as: {known_names[best_idx]} (similarity: {best_score:.2f})")
        else:
            st.warning(f"Face not recognized (highest similarity: {best_score:.2f})")
