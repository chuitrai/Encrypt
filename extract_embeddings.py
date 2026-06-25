import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import insightface
from insightface.app import FaceAnalysis

print("🚀 Bắt đầu quá trình Chế biến Dữ liệu và Trích xuất Đặc trưng...")

# ==========================================
# 1. HÀM CẮT MIỀN TẦN SỐ (DCT MASKING)
# ==========================================
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def apply_frequency_privacy(img_bgr, mask_ratio=0.3):
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

# ==========================================
# 2. KHỞI TẠO INSIGHTFACE (TRÊN CPU)
# ==========================================
print("⏳ Khởi tạo InsightFace...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# det_thresh=0.2 để nó cố gắng tìm cả những khuôn mặt đã bị mờ
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)

# ==========================================
# 3. QUÉT THƯ MỤC LFW VÀ TRÍCH XUẤT
# ==========================================
INPUT_DIR = "dataset_test/celeba_samples"  # Đường dẫn tới thư mục 100 ảnh của bạn
E_target_list = []
E_freq_list = []

print(f"🔍 Bắt đầu quét thư mục: {INPUT_DIR}")

# Đếm số lượng ảnh xử lý thành công
success_count = 0

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    img_path = os.path.join(INPUT_DIR, filename)
    img_original = cv2.imread(img_path)
    
    if img_original is None:
        continue

    # Bước A: Tìm mặt và trích xuất vector từ ẢNH GỐC
    faces_original = app.get(img_original)
    if len(faces_original) == 0:
        continue # Bỏ qua nếu ảnh gốc mà cũng không thấy mặt
        
    emb_target = faces_original[0].embedding
    
    # Bước B: Làm hỏng ảnh bằng DCT (Cắt 30%)
    img_privacy = apply_frequency_privacy(img_original, mask_ratio=0.3)
    
    # Bước C: Tìm mặt và trích xuất vector từ ẢNH HỎNG
    faces_freq = app.get(img_privacy)
    
    # Nếu InsightFace vẫn nhìn ra được mặt trong ảnh hỏng, ta mới lấy data này để train
    if len(faces_freq) > 0:
        emb_freq = faces_freq[0].embedding
        
        # Lưu vào danh sách
        E_target_list.append(emb_target)
        E_freq_list.append(emb_freq)
        success_count += 1
        print(f"✔️ Đã trích xuất thành công: {filename}")
    else:
        print(f"❌ InsightFace bị mù với ảnh hỏng của: {filename}")

# ==========================================
# 4. LƯU THÀNH FILE .NPY CHO BẢO TRAIN ADAPTER
# ==========================================
print(f"\n🎉 Xong! Trích xuất thành công {success_count} cặp embeddings.")

# Chuyển list thành ma trận Numpy (Shape: [success_count, 512])
E_target_matrix = np.array(E_target_list)
E_freq_matrix = np.array(E_freq_list)

# Lưu ra ổ cứng
np.save("E_target.npy", E_target_matrix)
np.save("E_freq.npy", E_freq_matrix)

print("💾 Đã lưu 2 file 'E_target.npy' và 'E_freq.npy'. Hãy ném 2 file này cho Bảo nhé!")
