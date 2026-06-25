import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# --- PHẦN 1: KIẾN TRÚC MẠNG ADAPTER ---
class FaceAdapter(nn.Module):
    """
    Mạng Adapter thực hiện Domain Adaptation ánh xạ vector 512 chiều.
    """
    def __init__(self):
        super(FaceAdapter, self).__init__()
        # Kiến trúc MLP đơn giản
        self.fc1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        
    def forward(self, x):
        # Truyền qua các lớp Linear và ReLU
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # Bắt buộc áp dụng L2 Normalization ở bước cuối cùng
        # Để đảm bảo vector đầu ra luôn nằm trên unit hypersphere (độ dài = 1)
        x = F.normalize(x, p=2, dim=1)
        return x

# --- PHẦN 2: DỮ LIỆU GIẢ LẬP (MOCK DATA) ---
def get_mock_dataloader(num_samples=1000, batch_size=32):
    """
    Hàm sinh dữ liệu giả lập để huấn luyện.
    """
    # Khởi tạo ngẫu nhiên E_freq và E_target
    E_freq = torch.randn(num_samples, 512)
    E_target = torch.randn(num_samples, 512)
    
    # Để giả lập việc E_freq có sự tương đồng nhất định với E_target (cosine similarity > 0)
    # Ta cộng một phần của E_target vào E_freq trước khi chuẩn hóa
    E_freq = E_freq + 0.5 * E_target
    
    # Bắt buộc L2 Normalize cả 2 tensor ngay từ lúc sinh ra
    E_freq = F.normalize(E_freq, p=2, dim=1)
    E_target = F.normalize(E_target, p=2, dim=1)
    
    # Đóng gói vào Dataset và DataLoader
    dataset = TensorDataset(E_freq, E_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# --- PHẦN 3: TRAINING LOOP ---
def train_model():
    # Setup Device (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")
    
    # 1. Khởi tạo mô hình
    model = FaceAdapter().to(device)
    
    # ---- KIỂM TRA CHÉO (Checklist Task 2) ----
    dummy_input = torch.randn(5, 512).to(device)
    dummy_output = model(dummy_input)
    norms = torch.norm(dummy_output, dim=1)
    print("\n[Checklist] Kiểm tra L2 Normalization:")
    print(f"-> Norm của 5 mẫu thử (phải xấp xỉ 1.0): {norms.detach().cpu().numpy()}")
    # ------------------------------------------
    
    # 2. Khởi tạo Dataloader
    dataloader = get_mock_dataloader(num_samples=1000, batch_size=32)
    
    # ---- KIỂM TRA CHÉO (Checklist Task 3) ----
    batch_freq, batch_target = next(iter(dataloader))
    print("\n[Checklist] Kiểm tra DataLoader:")
    print(f"-> Shape batch_freq: {batch_freq.shape} (Cần: [32, 512])")
    print(f"-> Shape batch_target: {batch_target.shape} (Cần: [32, 512])")
    # ------------------------------------------
    
    # Lưu lại giá trị độ tương đồng trung bình trước khi train để nghiệm thu Task 4
    with torch.no_grad():
        sim_before = F.cosine_similarity(batch_freq.to(device), batch_target.to(device)).mean().item()
    
    # 3. Khai báo Loss và Optimizer
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    print(f"\n[Checklist] Bắt đầu huấn luyện ({epochs} Epochs)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_f, batch_t in dataloader:
            batch_f, batch_t = batch_f.to(device), batch_t.to(device)
            
            # Forward pass
            outputs = model(batch_f)
            
            # CosineEmbeddingLoss yêu cầu nhãn target_label = 1 cho các cặp tương đồng
            target_label = torch.ones(batch_f.size(0)).to(device)
            
            # Tính loss
            loss = criterion(outputs, batch_t, target_label)
            
            # Backward và optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    # 4. Lưu mô hình
    save_path = "mock_adapter_day1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[Checklist] Đã lưu weights của model tại: {save_path}")
    
    # ---- KIỂM TRA CHÉO (Checklist Task 4) ----
    model.eval()
    with torch.no_grad():
        outputs_after = model(batch_freq.to(device))
        sim_after = F.cosine_similarity(outputs_after, batch_target.to(device)).mean().item()
        
    print("\n[Checklist] Đánh giá hiệu quả Adapter trên 1 batch:")
    print(f"-> Cosine Similarity TRƯỚC khi qua Adapter : {sim_before:.4f}")
    print(f"-> Cosine Similarity SAU khi qua Adapter  : {sim_after:.4f}")
    if sim_after > sim_before:
        print("=> THÀNH CÔNG: Mạng Adapter đã học được cách kéo gần vector E_freq về E_target!")
    else:
        print("=> CẢNH BÁO: Similarity không tăng, cần kiểm tra lại.")

if __name__ == "__main__":
    train_model()
