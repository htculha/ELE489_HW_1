import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Distance fonksiyonları
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# 2. k-NN tahmin fonksiyonu
def knn_predict(X_train, y_train, x_test, k=k, distance_func=euclidean_distance):
    distances = []
    for i in range(len(X_train)):
        dist = distance_func(x_test, X_train.iloc[i])
        distances.append((dist, y_train.iloc[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for (_, label) in distances[:k]]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# 3. K değerlerini dene
k_values = [1, 3, 5, 7, 9]
euclidean_accuracies = []
manhattan_accuracies = []

for k in k_values:
    # Euclidean
    y_pred_euc = [knn_predict(X_train, y_train, X_test.iloc[i], k=k, distance_func=euclidean_distance)
                  for i in range(len(X_test))]
    acc_euc = accuracy_score(y_test, y_pred_euc)
    euclidean_accuracies.append(acc_euc)

    # Manhattan
    y_pred_man = [knn_predict(X_train, y_train, X_test.iloc[i], k=k, distance_func=manhattan_distance)
                  for i in range(len(X_test))]
    acc_man = accuracy_score(y_test, y_pred_man)
    manhattan_accuracies.append(acc_man)

    print(f"\n🔹 k = {k}")
    print(f"Euclidean Accuracy: {acc_euc:.2f}")
    print(f"Manhattan Accuracy: {acc_man:.2f}")
    
    print("\n✅ Confusion Matrix (Euclidean):")
    print(confusion_matrix(y_test, y_pred_euc))
    print(classification_report(y_test, y_pred_euc))

    print("✅ Confusion Matrix (Manhattan):")
    print(confusion_matrix(y_test, y_pred_man))
    print(classification_report(y_test, y_pred_man))

# 4. Grafiği çiz
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
