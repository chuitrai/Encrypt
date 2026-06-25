import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import insightface
from insightface.app import FaceAnalysis

print("🚀 Bắt đầu quá trình trích xuất ĐẠI TRÀ 5 cấp độ tần số...")

# --- 1. Hàm cắt miền tần số ---
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def apply_frequency_privacy(img_bgr, mask_ratio):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    
    Y, Cr, Cb = cv2.split(img_ycrcb)
    dct_Y = dct2(Y)
    
    h, w = dct_Y.shape
    dct_Y[0:int(h * mask_ratio), 0:int(w * mask_ratio)] = 0 
    
    Y_recovered = np.clip(idct2(dct_Y), 0, 255).astype(np.uint8)
    img_ycrcb_recovered = cv2.merge([Y_recovered, Cr, Cb])
    img_privacy_rgb = cv2.cvtColor(img_ycrcb_recovered, cv2.COLOR_YCrCb2RGB)
    return cv2.cvtColor(img_privacy_rgb, cv2.COLOR_RGB2BGR)

# --- 2. Khởi tạo AI ---
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)

# --- 3. Quét thư mục và xử lý ---
INPUT_DIR = "dataset_test/celeba_samples" # Đảm bảo thư mục này có ảnh
mask_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

# Dictionary lưu vector hỏng theo từng cấp độ
freq_embeddings_dict = {ratio: [] for ratio in mask_ratios}
target_embeddings_list = []

print(f"🔍 Đang quét thư mục: {INPUT_DIR}")
success_count = 0

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    img_path = os.path.join(INPUT_DIR, filename)
    img_original = cv2.imread(img_path)
    if img_original is None: continue

    # Lấy vector ảnh gốc
    faces_original = app.get(img_original)
    if len(faces_original) == 0: continue
    emb_target = faces_original[0].embedding
    
    # Biến cờ: Kiểm tra xem ảnh này có "sống sót" qua mọi cấp độ cắt không
    # (Để đảm bảo độ dài ma trận E_target và E_freq luôn bằng nhau)
    survived_all_ratios = True
    temp_freq_embs = {}
    
    for ratio in mask_ratios:
        img_privacy = apply_frequency_privacy(img_original, ratio)
        faces_freq = app.get(img_privacy)
        
        if len(faces_freq) > 0:
            temp_freq_embs[ratio] = faces_freq[0].embedding
        else:
            survived_all_ratios = False
            break # Mù ở 1 cấp độ thì bỏ luôn ảnh này để đảm bảo data cân bằng
            
    # Nếu InsightFace nhận diện thành công mặt ở CẢ 5 cấp độ nhiễu
    if survived_all_ratios:
        target_embeddings_list.append(emb_target)
        for ratio in mask_ratios:
            freq_embeddings_dict[ratio].append(temp_freq_embs[ratio])
        success_count += 1
        print(f"✔️ Xử lý trọn vẹn 5 cấp độ: {filename}")

# --- 4. Lưu ra ổ cứng ---
if success_count > 0:
    np.save("E_target.npy", np.array(target_embeddings_list))
    for ratio in mask_ratios:
        np.save(f"E_freq_{ratio}.npy", np.array(freq_embeddings_dict[ratio]))
        
    print(f"\n🎉 XONG! Đã lưu E_target.npy và 5 file E_freq_X.npy (Size: {success_count} ảnh).")
    print("👉 Bây giờ bạn có thể bật 'benchmark_pipeline.py' lên chạy lại được rồi!")
else:
    print("❌ Thất bại: Không có ảnh nào sống sót qua được mức cắt 0.9. Hãy nới lỏng det_thresh!")
