import numpy as np

def save_as_pgm(filename, width, height, data):
    """RAWデータをPGM形式(P5)で保存する（モザイク状に保存される）"""
    with open(filename, 'wb') as f:
        # PGMヘッダー: P5 \n 幅 高さ \n 最大値 \n
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(data.tobytes())
    print(f"Saved: {filename}")

def save_as_ppm(filename, width, height, data):
    """
    RAWデータを簡易的にPPM形式(P6)で保存する。
    ※本来はデモザイク処理が必要だが、ここでは単純に3回繰り返して擬似カラーにする例を示す。
    正しく色を出すには OpenCV の cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB) 等を使用すること。
    """
    # ここでは簡単のため、Bayerデータをグレースケールとして扱い、R=G=Bとして保存する例
    # 実際の色再現をしたい場合は OpenCV を導入してください。
    rgb_data = np.stack([data, data, data], axis=-1)
    
    with open(filename, 'wb') as f:
        # PPMヘッダー: P6 \n 幅 高さ \n 最大値 \n
        header = f"P6\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(rgb_data.tobytes())
    print(f"Saved: {filename}")

def convert_raw(input_file, output_file, width, height, bit_depth=8):
    # 1. RAWファイルの読み込み
    # Lucid Tritonのbit-depthに合わせて dtype を変更してください (uint8 or uint16)
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    
    try:
        raw_data = np.fromfile(input_file, dtype=dtype)
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # サイズチェック
    expected_size = width * height
    if raw_data.size != expected_size:
        print(f"Warning: Data size ({raw_data.size}) does not match dimensions ({expected_size}).")
        # 端数を切り捨てるか、エラーにする
        raw_data = raw_data[:expected_size]

    # 2次元配列にリシェイプ
    image = raw_data.reshape((height, width))

    # ビット深度が10bitや12bitでuint16に入っている場合は、8bitに正規化する必要がある
    if bit_depth > 8:
        # 例: 12bit(0-4095) -> 8bit(0-255)
        image = (image / (2**(bit_depth-8))).astype(np.uint8)

    # 出力形式に応じて保存
    if output_file.endswith('.pgm'):
        save_as_pgm(output_file, width, height, image)
    elif output_file.endswith('.ppm'):
        save_as_ppm(output_file, width, height, image)
    else:
        print("Unsupported output format. Use .pgm or .ppm")

# --- 設定項目 ---
INPUT_RAW = "capture.raw"    # 保存したRAWファイル名
OUTPUT_PGM = "result.pgm"    # 出力ファイル名
WIDTH = 1920                 # カメラの解像度（横）
HEIGHT = 1080                # カメラの解像度（縦）
BIT_DEPTH = 8                # 8, 10, 12 等

if __name__ == "__main__":
    convert_raw(INPUT_RAW, OUTPUT_PGM, WIDTH, HEIGHT, BIT_DEPTH)
