from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def train_knn(database):
    X = []
    y = []
    for name, embeddings in database.items():
        for emb in embeddings:
            X.append(emb)
            y.append(name)
    
    if len(X) == 0:
        return None
    
    X = np.array(X)
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(X, y)
    return knn

def recognize_with_knn(embedding, database):
    knn = train_knn(database)
    if knn is None:
        return (None, 0)

    embedding = np.array(embedding).reshape(1, -1)
    predicted_name = knn.predict(embedding)[0]
    distances, _ = knn.kneighbors(embedding)
    confidence = 1 / (1 + distances[0][0])
    return (predicted_name, confidence)

def find_duplicate(embedding, database, threshold=0.75):
    """
    Check if the embedding matches any existing user.
    Returns (name, similarity) if duplicate found.
    """
    knn = train_knn(database)
    if knn is None:
        return (None, 0)

    embedding = np.array(embedding).reshape(1, -1)
    distances, indices = knn.kneighbors(embedding)
    nearest_distance = distances[0][0]
    
    # استخدم الـ indices للعثور على اسم الشخص من قاعدة البيانات
    all_names = []
    for name, embeddings in database.items():
        all_names.extend([name]*len(embeddings))
    
    nearest_name = all_names[indices[0][0]]
    similarity = 1 / (1 + nearest_distance)

    if similarity >= threshold:
        return (nearest_name, similarity)
    return (None, similarity)
