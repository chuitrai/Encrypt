import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- PHẦN 1: KIẾN TRÚC MẠNG ADAPTER ---
class FaceAdapter(nn.Module):
    """
    Mạng Adapter thực hiện Domain Adaptation ánh xạ vector 512 chiều.
    """
    def __init__(self):
        super(FaceAdapter, self).__init__()
        # Kiến trúc MLP đơn giản: 512 -> 1024 -> ReLU -> 512
        self.fc1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # Bắt buộc áp dụng L2 Normalization ở bước cuối cùng
        x = F.normalize(x, p=2, dim=1)
        return x

# --- PHẦN 2: DỮ LIỆU THỰC TẾ (REAL DATA) ---
def get_real_dataloader(batch_size=32):
    """
    Hàm load dữ liệu thực tế từ file .npy
    """
    print("Đang tải dữ liệu từ file E_freq.npy và E_target.npy...")
    try:
        E_freq_np = np.load("E_freq.npy")
        E_target_np = np.load("E_target.npy")
    except FileNotFoundError:
        print("[LỖI] Không tìm thấy file dữ liệu. Bạn chắc chắn đã chạy extract_embeddings.py chưa?")
        sys.exit(1)
        
    print(f"Số lượng sample tải được: {E_freq_np.shape[0]}")
    
    # Chuyển đổi Numpy array sang PyTorch Tensor
    E_freq = torch.tensor(E_freq_np, dtype=torch.float32)
    E_target = torch.tensor(E_target_np, dtype=torch.float32)
    
    # Bắt buộc L2 Normalize cả 2 tensor
    E_freq = F.normalize(E_freq, p=2, dim=1)
    E_target = F.normalize(E_target, p=2, dim=1)
    
    # Đóng gói vào Dataset và DataLoader
    dataset = TensorDataset(E_freq, E_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Tính Cosine Similarity ban đầu trên toàn bộ dataset
    with torch.no_grad():
        sim_before = F.cosine_similarity(E_freq, E_target).mean().item()
        
    return dataloader, sim_before, E_freq, E_target

# --- PHẦN 3: TRAINING LOOP ---
def train_model():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nĐang chạy trên thiết bị: {device}")
    
    # 1. Khởi tạo mô hình
    model = FaceAdapter().to(device)
    
    # 2. Khởi tạo Dataloader
    dataloader, sim_before, E_freq_full, E_target_full = get_real_dataloader(batch_size=32)
    E_target_full_device = E_target_full.to(device)
    
    # 3. Khai báo Loss và Optimizer
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    print(f"\n[Bảo's Task] Bắt đầu huấn luyện thực tế ({epochs} Epochs)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_f, batch_t in dataloader:
            batch_f, batch_t = batch_f.to(device), batch_t.to(device)
            outputs = model(batch_f)
            target_label = torch.ones(batch_f.size(0)).to(device)
            loss = criterion(outputs, batch_t, target_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {avg_loss:.4f}")
        
    # 4. Lưu mô hình
    save_path = "real_adapter_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nĐã lưu weights của model thực tế tại: {save_path}")
    
    # 5. Đánh giá hiệu quả Adapter trên TOÀN BỘ dataset
    model.eval()
    with torch.no_grad():
        recovered_embeddings = model(E_freq_full.to(device))
        sim_after = F.cosine_similarity(recovered_embeddings, E_target_full_device).mean().item()
        
    print("\n================ KẾT QUẢ NGHIỆM THU ================")
    print(f"-> Cosine Similarity TRƯỚC Adapter : {sim_before:.4f}")
    print(f"-> Cosine Similarity SAU Adapter   : {sim_after:.4f}")
    
    if sim_after > sim_before:
        improvement = (sim_after - sim_before) / sim_before * 100
        print(f"=> THÀNH CÔNG RỰC RỠ: Độ tương đồng đã được kéo tăng lên {improvement:.1f}%!")
        
    # --- PHẦN MỚI: TÍNH TOÁN L2 VÀ VẼ T-SNE ---
    print("\n================ PHÂN TÍCH CHUYÊN SÂU ================")
    # 1. Tính L2 Distance
    l2_distance_before = torch.norm(E_freq_full.to(device) - E_target_full_device, dim=1).mean().item()
    l2_distance_after = torch.norm(recovered_embeddings - E_target_full_device, dim=1).mean().item()

    print(f"Khoảng cách L2 TRƯỚC Adapter: {l2_distance_before:.4f}")
    print(f"Khoảng cách L2 SAU Adapter: {l2_distance_after:.4f} (Càng nhỏ càng tốt)")

    # 2. Xây dựng data để vẽ t-SNE
    print("\nĐang chạy t-SNE để giảm chiều dữ liệu từ 512D xuống 2D...")
    # Vì dữ liệu khá ít (100 ảnh), chỉnh perplexity nhỏ lại (vd: 15) để tránh lỗi
    N = E_target_full.shape[0]
    perplexity_val = min(30, N - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_val, max_iter=1000, random_state=42)
    
    all_embeddings = torch.cat([E_target_full_device, E_freq_full.to(device), recovered_embeddings], dim=0).cpu().detach().numpy()
    tsne_results = tsne.fit_transform(all_embeddings)

    tsne_target = tsne_results[0:N, :]
    tsne_freq = tsne_results[N:2*N, :]
    tsne_recovered = tsne_results[2*N:, :]

    # 3. Vẽ biểu đồ và lưu lại
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_freq[:, 0], tsne_freq[:, 1], c='red', alpha=0.6, label='Freq (Bị làm hỏng)')
    plt.scatter(tsne_target[:, 0], tsne_target[:, 1], c='blue', alpha=0.6, label='Target (Gốc)')
    plt.scatter(tsne_recovered[:, 0], tsne_recovered[:, 1], c='gold', alpha=0.8, marker='^', s=100, label='Recovered (Qua Adapter)')

    # Vẽ các đường nối Freq -> Recovered -> Target cho vài điểm đầu tiên để minh hoạ
    for i in range(min(5, N)):
        plt.plot([tsne_freq[i, 0], tsne_recovered[i, 0]], [tsne_freq[i, 1], tsne_recovered[i, 1]], 'k--', alpha=0.2)
        plt.plot([tsne_recovered[i, 0], tsne_target[i, 0]], [tsne_recovered[i, 1], tsne_target[i, 1]], 'g-', alpha=0.4)

    plt.title("t-SNE Visualization of Latent Space Manifold Alignment")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Cấu hình lưu vào thư mục dự án và thư mục artifact
    artifact_dir = r"C:\Users\ADMIN\.gemini\antigravity\brain\7bb22b4d-efc9-4710-8e7e-9efc9a8e2af1"
    plot_path1 = "tsne_plot.png"
    plot_path2 = os.path.join(artifact_dir, "tsne_plot.png")
    
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    import shutil
    try:
        shutil.copy(plot_path1, plot_path2)
    except:
        pass
    
    print(f"Đã lưu biểu đồ t-SNE tại: {plot_path1}")

if __name__ == "__main__":
    train_model()
