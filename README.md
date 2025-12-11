# Depi-ML-Engineer-GradProject
dataset used in the project is :"https://www.kaggle.com/datasets/jessicali9530/lfw-dataset"
## Project Overview
This project is a **Face Recognition system** using deep learning.  
It detects faces, generates embeddings, and recognizes identities based on a trained model.

## Folder Structure
Face Recognition Project/
│
├── 1_data_preparation/
├── 2_baseline_model/
├── 3_advanced_model/
├── 4_deployment/
└── 5_evaluation/

## Model
- **Architecture:** InceptionResnetV1 (pretrained on VGGFace2)  
- **Input:** 160 × 160 RGB face images  
- **Output:** 128-dimensional normalized embedding vector  
- **Trained Model:** `best_facenet_model.pth` (PyTorch)

## Training
- **Dataset:** LFW (faces cropped and aligned using MTCNN)  
- **Augmentation:** Horizontal flip, rotation, brightness/contrast adjustment  
- **Loss function:** Cross-Entropy or Triplet Loss  
- **Optimizer:** Adam, learning rate 0.0001  
- **Epochs:** 6 (PyTorch) / 8 (TensorFlow)

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- False Acceptance Rate (FAR)  

## Usage
1. Load the trained model.  
2. Detect and crop faces from new images using MTCNN.  
3. Generate embeddings using the model.  
4. Compare embeddings to recognize identities.

## Notes
- The PyTorch model can be converted to **ONNX/TorchScript** for faster inference.  
- Can be integrated into a **real-time system** using OpenCV and live camera feed.  
- Adding new identities is possible by adding their embeddings to the database.
