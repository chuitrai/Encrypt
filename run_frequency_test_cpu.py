import os
import cv2
import numpy as np
import pandas as pd
from scipy.fftpack import dct, idct
import insightface
from insightface.app import FaceAnalysis

# --- 1. HÀM XỬ LÝ DCT NHƯ ĐÃ THIẾT KẾ ---
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def apply_frequency_privacy(img_bgr, mask_ratio):
    # InsightFace dùng BGR, ta chuyển sang RGB rồi YCrCb để xử lý
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    
    Y, Cr, Cb = cv2.split(img_ycrcb)
    dct_Y = dct2(Y)
    
    h, w = dct_Y.shape
    dct_Y[0:int(h * mask_ratio), 0:int(w * mask_ratio)] = 0 
    
    Y_recovered = np.clip(idct2(dct_Y), 0, 255).astype(np.uint8)
    
    img_ycrcb_recovered = cv2.merge([Y_recovered, Cr, Cb])
    img_privacy_rgb = cv2.cvtColor(img_ycrcb_recovered, cv2.COLOR_YCrCb2RGB)
    
    # Trả về BGR để InsightFace dễ đọc
    return cv2.cvtColor(img_privacy_rgb, cv2.COLOR_RGB2BGR)

# --- 2. HÀM TÍNH COSINE SIMILARITY ---
def cosine_similarity(v1, v2):
    # Công thức: (v1 dot v2) / (||v1|| * ||v2||)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# --- 3. KHỞI TẠO INSIGHTFACE TRÊN CPU ---
print("Đang khởi tạo InsightFace trên CPU...")
# Bắt buộc set det_thresh thấp (0.2) như trong báo cáo của bạn để detect mặt nhiễu
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)

# --- 4. CẤU HÌNH THƯ MỤC VÀ CHẠY THỰC NGHIỆM ---
INPUT_DIR = "assets/examples/"  # Thư mục chứa 12 ảnh public của bạn
MASK_RATIO = 0.3                # Cấp độ cắt tần số (Có thể chỉnh 0.1, 0.3, 0.5)
results = []

print(f"Bắt đầu quét thư mục: {INPUT_DIR} với Mask Ratio: {MASK_RATIO}")

# Lặp qua tất cả các file ảnh trong thư mục
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    img_path = os.path.join(INPUT_DIR, filename)
    img_original = cv2.imread(img_path)
    
    # Bước A: Trích xuất ảnh gốc
    faces_original = app.get(img_original)
    if len(faces_original) == 0:
        print(f"[Cảnh báo] Không tìm thấy mặt gốc trong {filename}")
        continue
    
    emb_target = faces_original[0].embedding
    
    # Bước B: Tạo ảnh bị hỏng (Frequency Privacy)
    img_privacy = apply_frequency_privacy(img_original, MASK_RATIO)
    
    # Bước C: Trích xuất ảnh hỏng
    faces_freq = app.get(img_privacy)
    
    if len(faces_freq) == 0:
        print(f"[Cảnh báo] InsightFace bị MÙ hoàn toàn với ảnh hỏng của {filename}")
        results.append({
            "File": filename, 
            "Detected_Privacy": False, 
            "Cosine_Similarity": None
        })
    else:
        emb_freq = faces_freq[0].embedding
        sim_score = cosine_similarity(emb_target, emb_freq)
        print(f"[Thành công] {filename} | Cosine Sim: {sim_score:.4f}")
        
        results.append({
            "File": filename, 
            "Detected_Privacy": True, 
            "Cosine_Similarity": sim_score
        })

# --- 5. LƯU KẾT QUẢ RA CSV ---
df = pd.DataFrame(results)
df.to_csv("metrics_frequency_test.csv", index=False)
print("\nĐã lưu báo cáo thành công vào 'metrics_frequency_test.csv'!")

if df['Cosine_Similarity'].count() > 0:
    print(f"==> ĐỘ TƯƠNG ĐỒNG TRUNG BÌNH KHI CHƯA DÙNG ADAPTER: {df['Cosine_Similarity'].mean():.4f}")
