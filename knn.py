# k-nn algoritması ve uzaklık modelleri; bu kısımda açıklamalar fazla olacaktır

#gerekli kütüphanelerin elenmesi, grafik çizim,accuracy ve confusion matrix hesaplaması vs
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#bu iki fonksiyon, euclidian ve manhattan uzunluk fonksiyonlarını tanımlıyor
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


""" k-NN tahmin fonksiyonu
    fonksiyona, X_train ve y_train adlı eğitim verileri gönderiliyor. K-NN algoritması bir lazy learning
    metodu olmasıyla, her test verisini her training verisiyle kıyaslıyarak bir sonuç elde ediyor.
    k: farklı k değerleri için bir değişken, distance_funciton da farklı uzunluk fonksiyonlarını kullanmak
    için fonksiyona gönderiliyor. """

def knn_predict(X_train, y_train, x_test, k=1, distance_func=euclidean_distance):
    
    """bu kod parçası, test verisinin training verisindeki değerlere olan bütün uzaklıklarını hesaplıyor
     ardından training verisinin bulunduğu sınıfla birlike, bu şekilde saklanıyor:
    
    [
    (1.25, 0),
    (2.89, 1),
    (0.73, 1),
    (3.12, 2),
    ...
    ]
    
    """
    distances = []
    for i in range(len(X_train)):
        dist = distance_func(x_test, X_train.iloc[i])
        distances.append((dist, y_train.iloc[i]))
    
    #distance matrixisini küçüten büyüğe sıralıyor
    distances.sort(key=lambda x: x[0]) 
    
    #en yakın 'k' kadar komşunun class bilgisini alıyor
    k_nearest_labels = [label for (_, label) in distances[:k]] 
    
    #en çok tekrar eden 1. değeri most_common değişkenine atar
    most_common = Counter(k_nearest_labels).most_common(1)
    
    #fonksiyon en çok tekrar eden sınıf değerini döndürür
    return most_common[0][0]


#farklı k değerlerini iki farklı uzunluk fonksiyonunda test edeceğiz

k_values = [1, 3, 5, 7, 9]
euclidean_accuracies = []
manhattan_accuracies = []

for k in k_values:
    
    """ Euclidean mesafe fonksiyonunu kullanarak, 3. kısımda ayırdığımız test setindeki her değer için 
        tahmin üretir ve bu tahminlerini y_pred_euc da saklar"""
    
    y_pred_euc = [knn_predict(X_train, y_train, X_test.iloc[i], k=k, distance_func=euclidean_distance)
                  for i in range(len(X_test))]

    """accuracy hesaplama satırları, y_test deki sınıf ile y_pred_euc daki tahmin edilen sınıfları 
       kıyaslıyor. ardından sonuçları euclidean_accuracie matrixinin sonuna ekliyoruz """ 
    acc_euc = accuracy_score(y_test, y_pred_euc)
    euclidean_accuracies.append(acc_euc)

    # aynı işlemler Manhattan uzalık hesaplaması için de yapılıyor
    y_pred_man = [knn_predict(X_train, y_train, X_test.iloc[i], k=k, distance_func=manhattan_distance)
                  for i in range(len(X_test))]
    acc_man = accuracy_score(y_test, y_pred_man)
    manhattan_accuracies.append(acc_man)


    """ bu kısımdan sonrası, elde edilen bilgilerden çıktı almak ve görselleştirmek üzerinedir."""
    
    print(f"\n-> k = {k}")
    print(f"Euclidean Accuracy: {acc_euc:.2f}")
    print(f"Manhattan Accuracy: {acc_man:.2f}")
    
    print("\n-> Confusion Matrix (Euclidean):")
    print(confusion_matrix(y_test, y_pred_euc))
    print(classification_report(y_test, y_pred_euc))

    print("-> Confusion Matrix (Manhattan):")
    print(confusion_matrix(y_test, y_pred_man))
    print(classification_report(y_test, y_pred_man))


plt.figure(figsize=(9, 5))
plt.plot(k_values, euclidean_accuracies, marker='o', label='Euclidean', color='purple')
plt.plot(k_values, manhattan_accuracies, marker='s', label='Manhattan', color='teal')
plt.title("Accuracy vs K (Euclidean vs Manhattan)")
plt.xlabel("k Value")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.legend()
