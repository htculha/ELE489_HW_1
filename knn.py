import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Euclidean distance fonksiyonu
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
    
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))
    
# k-NN tahmin fonksiyonu
def knn_predict(X_train, y_train, x_test, k=3, distance_func=euclidean_distance):
    distances = []

    for i in range(len(X_train)):
        dist = distance_func(x_test, X_train.iloc[i])
        distances.append((dist, y_train.iloc[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for (_, label) in distances[:k]]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Test seti için tüm tahminleri üret
k = 7
y_pred = []
for i in range(len(X_test)):
    prediction = knn_predict(X_train, y_train, X_test.iloc[i], k=k, distance_func=euclidean_distance)
    y_pred.append(prediction)

# Doğruluk oranını yazdır
print("k:"k)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(9, 5))
plt.plot(k_values, euclidean_accuracies, marker='o', label='Euclidean', color='purple')
plt.plot(k_values, manhattan_accuracies, marker='s', label='Manhattan', color='teal')

plt.title("Accuracy vs K (Euclidean vs Manhattan)")
plt.xlabel("k Value")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

