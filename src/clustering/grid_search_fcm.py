import numpy as np
import skfuzzy as fuzz
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.config import DATA_PROCESSED_DIR


# Đọc dữ liệu đã chuẩn hóa
X_scaled_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "normalized_scores.csv"))
data = X_scaled_df.values.T  # Form (features, N)

print("🔍 ĐANG THỬ NGHIỆM TÌM THAM SỐ m TỐI ƯU CHO FCM:\n")
print(f"{'m (Fuzziness)':<15}{'FPC (Càng cao càng tốt)':<25}{'FPE (Càng thấp càng tốt)':<25}")
print("-" * 65)

for m in [1.2, 1.3, 1.5, 1.7, 2.0]:
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=3, m=m, error=0.005, maxiter=1000, seed=42
    )
    
    # Tính FPE (Fuzzy Partition Entropy)
    # Tránh log(0) bằng cách cộng eps
    fpe = -np.mean(np.sum(u * np.log(u + 1e-10), axis=0))
    
    print(f"{m:<15.1f}{fpc:<25.4f}{fpe:<25.4f}")