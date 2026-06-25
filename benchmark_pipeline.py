import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
import math
import os
import warnings
warnings.filterwarnings("ignore")

# --- 1. Module Architectures ---
class LinearAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512)
        
    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

class BottleneckAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 512)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + identity
        return F.normalize(out, p=2, dim=1)

# --- 2. Module Loss Functions ---
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        labels = torch.ones(outputs.size(0)).to(outputs.device)
        l_cos = self.cos_loss(outputs, targets, labels)
        l_mse = self.mse_loss(outputs, targets)
        return self.alpha * l_cos + self.beta * l_mse

# --- 3. Module Metric Engine ---
def evaluate_adapter(model, E_freq, E_target):
    model.eval()
    with torch.no_grad():
        E_recovered = model(E_freq)
        
        # 1. Cosine Similarity
        cos_sim = F.cosine_similarity(E_recovered, E_target, dim=1).mean().item()
        
        # 2. L2 Distance
        l2_dist = torch.norm(E_recovered - E_target, p=2, dim=1).mean().item()
        
        # 3. Feature Variance Shift
        var_target = torch.var(E_target, dim=0).mean().item()
        var_recovered = torch.var(E_recovered, dim=0).mean().item()
        var_shift = abs(var_target - var_recovered)
        
    return {
        "Cosine_Similarity": cos_sim,
        "L2_Distance": l2_dist,
        "Feature_Variance_Shift": var_shift
    }, E_recovered

# --- 4. Module Visualization ---
def plot_tsne(E_target, E_freq, E_recovered, title_suffix=""):
    E_target_np = E_target.cpu().numpy()
    E_freq_np = E_freq.cpu().numpy()
    E_recovered_np = E_recovered.cpu().numpy()
    
    all_embeddings = np.concatenate([E_target_np, E_freq_np, E_recovered_np], axis=0)
    N = E_target_np.shape[0]
    
    perplexity_val = min(30, N - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_val, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(all_embeddings)
    
    tsne_target = tsne_results[0:N, :]
    tsne_freq = tsne_results[N:2*N, :]
    tsne_recovered = tsne_results[2*N:, :]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_freq[:, 0], tsne_freq[:, 1], c='red', alpha=0.5, label='Freq (Corrupted)')
    plt.scatter(tsne_target[:, 0], tsne_target[:, 1], c='blue', alpha=0.5, label='Target (Original)')
    plt.scatter(tsne_recovered[:, 0], tsne_recovered[:, 1], c='gold', alpha=0.7, marker='^', label='Recovered')
    
    for i in range(min(5, N)):
        plt.plot([tsne_freq[i, 0], tsne_recovered[i, 0]], [tsne_freq[i, 1], tsne_recovered[i, 1]], 'k--', alpha=0.2)
        plt.plot([tsne_recovered[i, 0], tsne_target[i, 0]], [tsne_recovered[i, 1], tsne_target[i, 1]], 'g-', alpha=0.3)
        
    plt.title(f"Latent Space Manifold Alignment\n{title_suffix}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs("benchmark_plots", exist_ok=True)
    plt.savefig(f"benchmark_plots/tsne_{title_suffix.replace(' ', '_')}.png", dpi=300)
    plt.close()

def plot_pareto_frontier(results_df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="Mask_Ratio", y="Cosine_Similarity", hue="Architecture", marker='o')
    plt.title("Pareto Frontier: Privacy vs Utility")
    plt.xlabel("Mask Ratio (Privacy Level)")
    plt.ylabel("Cosine Similarity (Utility)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs("benchmark_plots", exist_ok=True)
    plt.savefig("benchmark_plots/pareto_frontier.png", dpi=300)
    plt.close()

def plot_radar_chart(results_df):
    archs = results_df['Architecture'].unique()
    metrics = ['Cosine_Similarity', 'L2_Distance', 'Feature_Variance_Shift']
    
    avg_df = results_df.groupby('Architecture')[metrics].mean()
    
    # Normalize metrics 0-1 for radar chart
    for m in metrics:
        min_val = avg_df[m].min()
        max_val = avg_df[m].max()
        if max_val > min_val:
            if m == 'Cosine_Similarity':
                avg_df[m] = (avg_df[m] - min_val) / (max_val - min_val)
            else:
                # Invert for L2 and Variance (Smaller is better -> closer to 1)
                avg_df[m] = 1 - (avg_df[m] - min_val) / (max_val - min_val)
        else:
            avg_df[m] = 1.0

    labels=np.array(['Cosine Sim (Norm)', 'Inverse L2 Dist (Norm)', 'Inverse Var Shift (Norm)'])
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    colors = ['b', 'r', 'g']
    for idx, arch in enumerate(archs):
        values = avg_df.loc[arch].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=2, label=arch)
        ax.fill(angles, values, color=colors[idx], alpha=0.25)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title('Radar Chart: Architecture Trade-offs')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    os.makedirs("benchmark_plots", exist_ok=True)
    plt.savefig("benchmark_plots/radar_chart.png", dpi=300)
    plt.close()

# --- 5. Main Loop ---
def generate_dynamic_insight_report(df):
    print("\n" + "="*80)
    print("DEEP INSIGHT REPORT: DYNAMIC ANALYSIS OF ABLATION STUDY")
    print("="*80)
    
    # 1. Analyze performance at low mask ratios (e.g., 0.1)
    low_ratio_df = df[df['Mask_Ratio'] == 0.1]
    if not low_ratio_df.empty:
        avg_cos_low = low_ratio_df['Cosine_Similarity'].mean()
        print(f"[1] Tại mức nhiễu thấp (Mask Ratio = 0.1), độ tương đồng trung bình đạt {avg_cos_low:.4f}.")
        if avg_cos_low > 0.9:
            print("    -> Các kiến trúc dễ dàng tái thiết hoàn hảo (Manifold Alignment) do lượng thông tin bị mất rất ít.")
        else:
            print("    -> Việc khôi phục có dấu hiệu khó khăn ngay cả ở mức nhiễu thấp.")
            
    # 2. Analyze performance drop at high mask ratios (e.g., 0.9)
    high_ratio_df = df[df['Mask_Ratio'] == 0.9]
    if not high_ratio_df.empty and not low_ratio_df.empty:
        avg_cos_high = high_ratio_df['Cosine_Similarity'].mean()
        drop_percent = (avg_cos_low - avg_cos_high) / avg_cos_low * 100 if avg_cos_low > 0 else 0
        print(f"[2] Khi Mask Ratio tăng lên cực đại (0.9), hiệu suất giảm {drop_percent:.1f}% xuống còn {avg_cos_high:.4f}.")
        print("    -> Dấu hiệu rõ ràng của 'Nút thắt thông tin' (Information Bottleneck) khi lượng entropy định danh bị phá hủy quá mức.")
        
    # 3. Compare Architectures at highest stress
    if not high_ratio_df.empty:
        best_idx = high_ratio_df['Cosine_Similarity'].idxmax()
        worst_idx = high_ratio_df['Cosine_Similarity'].idxmin()
        best_arch = high_ratio_df.loc[best_idx]['Architecture']
        worst_arch = high_ratio_df.loc[worst_idx]['Architecture']
        
        print(f"[3] Phân tích dưới áp lực mất thông tin cao nhất (Mask Ratio = 0.9):")
        print(f"    - Kiến trúc TỐT NHẤT: {best_arch} (Cosine: {high_ratio_df.loc[best_idx]['Cosine_Similarity']:.4f})")
        print(f"    - Kiến trúc TỆ NHẤT:  {worst_arch} (Cosine: {high_ratio_df.loc[worst_idx]['Cosine_Similarity']:.4f})")
        
        if best_arch == 'ResidualAdapter':
            print("    -> ResidualAdapter thể hiện sức đề kháng vượt trội nhờ skip-connection giúp bảo toàn dòng thông tin định danh xuyên suốt mạng.")
        elif best_arch == 'BottleneckAdapter':
            print("    -> BottleneckAdapter ép feature qua không gian nhỏ gọn giúp lọc nhiễu hiệu quả hơn dưới áp lực lớn.")
            
        worst_var_shift = high_ratio_df.loc[worst_idx]['Feature_Variance_Shift']
        print(f"    -> Kiến trúc {worst_arch} ghi nhận Feature Variance Shift ở mức {worst_var_shift:.6f}.")
        if worst_var_shift > 0.05:
            print("       (Mức shift cao cảnh báo rủi ro Model Collapse - mạng nơ-ron có xu hướng sinh ra cùng một khuôn mặt tĩnh).")
    
    print("="*80)

def get_real_data(mask_ratio):
    """
    Đọc dữ liệu thực tế từ file .npy theo mask_ratio.
    """
    E_freq_np = np.load(f"E_freq_{mask_ratio}.npy")
    E_target_np = np.load("E_target.npy")
    
    E_freq = torch.tensor(E_freq_np, dtype=torch.float32)
    E_target = torch.tensor(E_target_np, dtype=torch.float32)
    
    E_freq = F.normalize(E_freq, p=2, dim=1)
    E_target = F.normalize(E_target, p=2, dim=1)
    
    return E_freq, E_target

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running benchmark on: {device}")
    
    mask_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    architectures = {
        "LinearAdapter": LinearAdapter,
        "BottleneckAdapter": BottleneckAdapter,
        "ResidualAdapter": ResidualAdapter
    }
    
    criterion = HybridLoss(alpha=0.7, beta=0.3)
    results = []
    
    for ratio in mask_ratios:
        print(f"\n========== MASK RATIO: {ratio} ==========")
        try:
            E_freq, E_target = get_real_data(ratio)
        except FileNotFoundError:
            print(f"[CẢNH BÁO] Không tìm thấy file E_freq_{ratio}.npy hoặc E_target.npy. Tự động bỏ qua mức này.")
            continue
            
        dataset = TensorDataset(E_freq, E_target)
        
        best_recovered_for_tsne = None
        
        for arch_name, ArchClass in architectures.items():
            model = ArchClass().to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            epochs = 30
            model.train()
            
            pbar = tqdm(range(epochs), desc=f"Training {arch_name}", leave=False, bar_format="{l_bar}{bar:20}{r_bar}")
            for epoch in pbar:
                for batch_f, batch_t in dataloader:
                    batch_f, batch_t = batch_f.to(device), batch_t.to(device)
                    outputs = model(batch_f)
                    loss = criterion(outputs, batch_t)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Evaluate
            metrics, E_recovered = evaluate_adapter(model, E_freq.to(device), E_target.to(device))
            
            results.append({
                "Mask_Ratio": ratio,
                "Architecture": arch_name,
                **metrics
            })
            
            if arch_name == "ResidualAdapter":
                best_recovered_for_tsne = E_recovered
                
        # Vẽ tSNE cho ResidualAdapter tại mỗi ratio
        if best_recovered_for_tsne is not None:
            plot_tsne(E_target.to(device), E_freq.to(device), best_recovered_for_tsne, title_suffix=f"ResidualAdapter_Ratio_{ratio}")
            
    # Save CSV
    df = pd.DataFrame(results)
    
    if df.empty:
        print("\n[LỖI] Dữ liệu trống do không tìm thấy bất kỳ file .npy nào! Hệ thống dừng vẽ biểu đồ.")
        return
        
    df.to_csv("benchmark_results.csv", index=False)
    print("\n[+] Đã lưu file kết quả: benchmark_results.csv")
    
    # Plot Evaluation
    plot_pareto_frontier(df)
    plot_radar_chart(df)
    print("[+] Đã xuất các biểu đồ vào thư mục 'benchmark_plots'")
    
    # Deep Insight Report
    generate_dynamic_insight_report(df)

if __name__ == "__main__":
    run_benchmark()
