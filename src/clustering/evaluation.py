import numpy as np

def calculate_fpc(u: np.ndarray) -> float:
    """
    Tính Fuzzy Partition Coefficient (FPC).
    Giá trị FPC càng gần 1 thì các cụm phân chia càng rõ ràng.
    """
    n = u.shape[1]
    fpc = np.trace(np.dot(u, u.T)) / float(n)
    return float(fpc)

def calculate_fpe(u: np.ndarray) -> float:
    """
    Tính Fuzzy Partition Entropy (FPE).
    Giá trị FPE càng gần 0 thì mức độ hỗn loạn/mờ giữa các cụm càng thấp.
    """
    n = u.shape[1]
    # Tránh lỗi log(0) bằng cách thêm epsilon nhỏ
    u_safe = np.clip(u, 1e-10, 1.0)
    fpe = -np.sum(u_safe * np.log(u_safe)) / float(n)
    return float(fpe)

def evaluate_fcm(u: np.ndarray) -> dict:
    """
    Đánh giá tổng thể ma trận độ thuộc U.
    """
    fpc = calculate_fpc(u)
    fpe = calculate_fpe(u)
    return {
        "fpc": fpc,
        "fpe": fpe
    }