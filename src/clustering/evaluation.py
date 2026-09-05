import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def calculate_fpc(u: np.ndarray) -> float:
    """
    Tính Fuzzy Partition Coefficient (FPC).
    Giá trị FPC thuộc [1/c, 1]. Càng gần 1 thì các cụm phân chia càng rõ ràng.
    """
    n = u.shape[1]
    fpc = np.trace(np.dot(u, u.T)) / float(n)
    return float(fpc)

def calculate_fpe(u: np.ndarray) -> float:
    """
    Tính Fuzzy Partition Entropy (FPE).
    Giá trị FPE càng gần 0 thì mức độ mờ/hỗn loạn giữa các cụm càng thấp.
    """
    n = u.shape[1]
    u_safe = np.clip(u, 1e-10, 1.0)
    fpe = -np.sum(u_safe * np.log(u_safe)) / float(n)
    return float(fpe)


def calculate_normalized_fpe(u: np.ndarray) -> float:
    """Normalize fuzzy partition entropy to [0, 1]."""
    n_clusters = u.shape[0]
    return calculate_fpe(u) / np.log(n_clusters)

def calculate_xie_beni(X: np.ndarray, cntr: np.ndarray, u: np.ndarray, m: float = 2.0) -> float:
    """
    Tính Xie-Beni Index (XB).
    XB = (Tổng biến thiên trọng số mờ) / (n * min_distance_between_centroids^2)
    Giá trị XB càng nhỏ càng thể hiện cụm chặt chẽ và tách biệt tốt.
    
    X: shape (n_samples, n_features)
    cntr: shape (c, n_features)
    u: shape (c, n_samples)
    """
    n_samples = X.shape[0]
    c = cntr.shape[0]
    
    # 1. Tính biến thiên nội cụm có trọng số mờ
    total_variation = 0.0
    for i in range(c):
        dists_sq = np.sum((X - cntr[i]) ** 2, axis=1)
        total_variation += np.sum((u[i] ** m) * dists_sq)
        
    # 2. Tính khoảng cách bình phương tối thiểu giữa 2 tâm cụm bất kỳ
    min_centroid_dist_sq = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            dist_sq = np.sum((cntr[i] - cntr[j]) ** 2)
            if dist_sq < min_centroid_dist_sq:
                min_centroid_dist_sq = dist_sq
                
    min_centroid_dist_sq = max(min_centroid_dist_sq, 1e-10)
    xb_index = total_variation / (n_samples * min_centroid_dist_sq)
    return float(xb_index)

def evaluate_fcm(X: np.ndarray, cntr: np.ndarray, u: np.ndarray, m: float = 2.0) -> dict:
    """
    Đánh giá toàn diện mô hình phân cụm mờ FCM.
    
    Parameters:
      X   : Dữ liệu mẫu shape (n_samples, n_features)
      cntr: Tâm cụm shape (c, n_features)
      u   : Ma trận độ thuộc shape (c, n_samples)
      m   : Trọng số mờ (fuzziness coefficient)
    """
    # 1. Các chỉ số mờ (Fuzzy Metrics)
    fpc = calculate_fpc(u)
    fpe = calculate_fpe(u)
    normalized_fpe = calculate_normalized_fpe(u)
    xb = calculate_xie_beni(X, cntr, u, m=m)
    
    # 2. Gán nhãn cứng (Hard Labels) để tính chỉ số phân cụm truyền thống
    labels = np.argmax(u, axis=0)
    
    # Kiểm tra điều kiện số cụm hợp lệ (>1 cụm có phần tử)
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        sil = float(silhouette_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))
        chi = float(calinski_harabasz_score(X, labels))
    else:
        sil, dbi, chi = -1.0, -1.0, -1.0

    return {
        "fpc": fpc,
        "fpe": fpe,
        "fpe_normalized": normalized_fpe,
        "xie_beni": xb,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": chi
    }