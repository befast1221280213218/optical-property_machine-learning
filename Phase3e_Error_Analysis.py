
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kurtosis, skew
from tqdm import tqdm

# --------------------------
# 1. 模型架构 (保持与 Phase 3a 训练一致)
# --------------------------
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, phys_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(phys_dim, phys_dim * 2), nn.SiLU(), nn.Linear(phys_dim * 2, feature_dim * 2))
    def forward(self, x, phys):
        params = self.mlp(phys).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return x * gamma + beta

class MetamaterialFourierGemini(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.SiLU()
        )
        self.phys_gate = FiLMLayer(64, 6)
        self.res_blocks = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128)
        )
        self.shortcut = nn.Conv2d(64, 128, 1, 2)
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(128, 32, 1), nn.SiLU(), nn.Conv2d(32, 128, 1), nn.Sigmoid())
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(nn.Linear(128 + 6, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, pixel_values, physical_features):
        x = self.stem(pixel_values)
        x = self.phys_gate(x, physical_features)
        res = self.shortcut(x)
        x = F.silu(self.res_blocks(x) + res)
        x = x * self.attn(x)
        x_feat = self.global_pool(x).view(x.size(0), -1)
        logits = self.regressor(torch.cat([x_feat, physical_features], dim=1))
        return {"logits": logits}

# --------------------------
# 2. 物理 Dataset (安全过滤版)
# --------------------------
class FourierPhysicsDataset(Dataset):
    def __init__(self, data_dir, result_file, target_size=(224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size
        df = pd.read_csv(result_file, sep=r'\s+', engine='python')
        df.columns = [c.replace(',', '').strip() for c in df.columns]
        self.results = df

    def __len__(self): return len(self.results)
    
    def __getitem__(self, idx):
        row = self.results.iloc[idx]
        w, n, k, S, C, seed = float(row['wavelength']), float(row['n']), float(row['k']), float(row['S']), float(row['C']), int(row['seed'])
        delta = 0.0125 * w * C 
        phys_features = torch.tensor([w, n, k, S, C, delta], dtype=torch.float32)

        def fmt(val): return str(int(val)) if float(val).is_integer() else str(val)
        filename = f"z(x,y)um_w{fmt(w)}_n{fmt(n)}_k{fmt(k)}_S{fmt(S)}_C{fmt(C)}_{seed}.txt"
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path): return None

        try:
            matrix = np.loadtxt(file_path, delimiter='\t')
            img_t = torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)
            img_spatial = F.interpolate(img_t, size=self.target_size, mode='bilinear').squeeze(0)
            fft_res = torch.fft.rfft2(img_spatial)
            mag = torch.log1p(torch.abs(fft_res))
            phase = torch.angle(fft_res)
            mag_map = F.interpolate(mag.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
            phase_map = F.interpolate(phase.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
            combined_input = torch.cat([img_spatial, mag_map, phase_map], dim=0)
            return {"pixel_values": combined_input, "phys": phys_features, "gt": torch.tensor([row['R']]).float(), "raw": matrix, "name": filename}
        except Exception: return None

def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# --------------------------
# 3. 统计逻辑与主运行
# --------------------------
def get_structure_stats(matrix):
    grad_x, grad_y = np.gradient(matrix)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return {
        "complexity_grad_var": np.var(grad_mag),
        "peak_skewness": skew(matrix.flatten()),
        "clumpiness_kurtosis": kurtosis(matrix.flatten())
    }


def run_error_analysis():
    data_path = "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1"
    res_file = os.path.join(data_path, "ART_result.txt")
    weight_path = "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/output_phase3/checkpoint-best/pytorch_model.bin"
    output_dir = "./Phase3e_Error_Analysis"
    os.makedirs(output_dir, exist_ok=True)


    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetamaterialFourierGemini().to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path), strict=False)
        print("✅ 权重加载成功。")

    dataset = FourierPhysicsDataset(data_path, res_file)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=safe_collate)

    model.eval()
    error_data = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Error Hunting"):
            if batch is None: continue
            imgs = batch["pixel_values"].to(device)
            phys = batch["phys"].to(device)
            gt_r = batch["gt"].to(device).flatten() # 强制压平
            
            outputs = model(imgs, phys)
            pred_r = outputs["logits"].flatten() # 强制压平
            
            # 转换为 numpy 以进行逐个计算
            preds_np = pred_r.cpu().numpy()
            gts_np = gt_r.cpu().numpy()

            for i in range(len(preds_np)):
                p, g = float(preds_np[i]), float(gts_np[i])
                abs_err = abs(g - p)
                
                raw_mat = batch["raw"][i].numpy()
                stats = get_structure_stats(raw_mat)
                
                error_data.append({
                    "filename": batch["name"][i],
                    "GT_R": g, "Pred_R": p, "Abs_Error": abs_err,
                    **stats,
                    "wavelength": phys[i, 0].item()
                })

    # 4. 统计与可视化
    df = pd.DataFrame(error_data)
    top_20 = df.sort_values(by="Abs_Error", ascending=False).head(20)
    top_20.to_csv(f"{output_dir}/Top20_Error_Report.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(df["complexity_grad_var"], df["Abs_Error"], alpha=0.3, color='blue')
    plt.xlabel("Gradient Variance (Structural Complexity)")
    plt.ylabel("Absolute Prediction Error")
    plt.savefig(f"{output_dir}/Structural_Complexity_vs_Error.png")
    
    print(f"🚀 分析完成，结果存放在 {output_dir}")

if __name__ == "__main__":
    run_error_analysis()