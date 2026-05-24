import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter

def calculate_mtf(image_path):
    # 1. 画像の読み込み（グレースケール）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("画像が見つかりません。")
        return

    # ノイズ除去のために軽いガウシアンフィルタを適用
    img = gaussian_filter(img, sigma=0.5)

    # 2. エッジ方向の検出 (Sobelフィルタ)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    # エッジの勾配方向を計算
    angle = np.arctan2(gy, gx)

    # 3. ESF (Edge Spread Function) の抽出
    # 中央付近でエッジに垂直なラインをサンプリング
    h, w = img.shape
    cx, cy = w // 2, h // 2
    
    # エッジの法線方向（垂直方向）ベクトル
    # 単純化のため、画像中央のエッジ角度を使用
    edge_angle = angle[cy, cx]
    nx, ny = -np.sin(edge_angle), np.cos(edge_angle)

    # サンプリング範囲の設定
    num_samples = 50  # 何本のラインで平均化するか
    sample_dist = 2   # ラインの間隔
    lsfs = []

    for i in range(-num_samples // 2, num_samples // 2):
        # サンプリング開始点（エッジに沿った方向）
        tx = cx + i * sample_dist * np.cos(edge_angle)
        ty = cy + i * sample_dist * np.sin(edge_angle)

        # エッジを横切る方向のプロファイルを抽出 (ESF)
        profile = []
        for j in range(-20, 20):
            px = int(tx + j * nx)
            py = int(ty + j * ny)
            if 0 <= px < w and 0 <= py < h:
                profile.append(img[py, px])
        
        if len(profile) == 40:
            # ESFを微分してLSF (Line Spread Function) を得る
            lsf = np.diff(profile)
            lsfs.append(lsf)

    # 全サンプルの平均 LSF を計算
    avg_lsf = np.mean(lsfs, axis=0)

    # 4. MTFの計算 (LSFのフーリエ変換)
    # 窓関数を適用して端の不連続性を抑える
    window = np.hanning(len(avg_lsf))
    weighted_lsf = avg_lsf * window
    
    mtf_raw = np.abs(fft(weighted_lsf))
    mtf = mtf_raw / np.max(mtf_raw) # 正規化 (0.0 ~ 1.0)

    # 半分だけ抽出（対称なため）
    mtf = mtf[:len(mtf)//2]
    
    # 周波数軸の作成 (簡易的に index として表示)
    freqs = np.linspace(0, 1, len(mtf)) 

    return freqs, mtf

# --- 実行部分 ---
image_file = 'edge_sample.png' # GIMPで保存した画像パスを指定してください
try:
    freqs, mtf = calculate_mtf(image_file)

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, mtf, linewidth=2)
    plt.title("MTF Curve (Slanted Edge Method)")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Modulation Transfer")
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.show()
except Exception as e:
    print(f"エラーが発生しました: {e}")

