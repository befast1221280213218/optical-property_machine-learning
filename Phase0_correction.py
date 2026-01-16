import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

# --------------------------
# 1. 核心架构定义 (确保与训练时完全一致)
# --------------------------
class MetamaterialGemini(nn.Module):
    def __init__(self):
        super().__init__()
        # 3通道输入：空间域 + 频域幅度 + 频域相位
        self.stem = nn.Conv2d(3, 64, 7, 2, 3) 
        self.head = nn.Sequential(
            nn.Linear(64*7*7 + 6, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )

    def forward(self, pixel_values, physical_features):
        x = F.silu(self.stem(pixel_values))
        x = F.adaptive_avg_pool2d(x, (7, 7)).view(x.size(0), -1)
        logits = self.head(torch.cat([x, physical_features], dim=1))
        return {"logits": logits}

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
            return {"pixel_values": combined_input, "physical_features": phys_features, "raw": matrix}
        except Exception:
            return None

# --------------------------
# 2. 全局相关性计算引擎 (Saliency Aggregation)
# --------------------------
def run_global_correlation_analysis():
    # 路径配置
    data_path = "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1"
    res_file = os.path.join(data_path, "ART_result.txt")
    weight_path = "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/output_phase3_pro/holdout_91/pytorch_model.bin"
    output_dir = "./Phase0_Global_Correlation"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = MetamaterialGemini().to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print(f"✅ 模型权重加载成功，开始全局分析...")

    # 加载数据
    dataset = FourierPhysicsDataset(data_path, res_file)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda x: [y for y in x if y is not None])

    # 初始化全局累加矩阵 (对应 224x224 像素)
    global_importance_map = np.zeros((224, 224))
    total_samples = 0

    model.eval()
    print(f"📊 正在扫描全数据集 ({len(dataset)} 样本)...")

    for batch in loader:
        if not batch: continue
        
        # 提取 Batch 数据
        pixel_values = torch.stack([item["pixel_values"] for item in batch]).to(device)
        phys_features = torch.stack([item["physical_features"] for item in batch]).to(device)
        
        # 开启梯度追踪以计算灵敏度
        pixel_values.requires_grad = True
        
        outputs = model(pixel_values, phys_features)
        logits = outputs["logits"]
        
        # 对预测的 R 值求和并反向传播
        model.zero_grad()
        logits.sum().backward()
        
        # 提取空间域通道 (Channel 0) 的绝对梯度
        # 梯度值越大，代表该像素点对 R 的影响越显著
        grad_map = pixel_values.grad.data.abs()[:, 0, :, :].cpu().numpy()
        global_importance_map += np.sum(grad_map, axis=0)
        total_samples += pixel_values.size(0)

    # 3. 统计平均与归一化
    avg_importance = global_importance_map / total_samples
    # 归一化到 [0, 1] 以便可视化
    norm_importance = (avg_importance - avg_importance.min()) / (avg_importance.max() - avg_importance.min() + 1e-10)

    # 4. 输出像素点对应的重要性权重矩阵 (.txt)
    txt_output_path = os.path.join(output_dir, "Global_Pixel_Importance_Weights.txt")
    np.savetxt(txt_output_path, norm_importance, delimiter='\t', fmt='%.6f')
    
    # 5. 可视化全局热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(norm_importance, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Statistical Correlation with R')
    plt.title("Universal Pixel Importance Map for Reflectivity (R)")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.savefig(os.path.join(output_dir, "Global_Importance_Heatmap.png"), dpi=300)
    
    print(f"🚀 分析完成！")
    print(f"1. 全局像素权重矩阵已保存: {txt_output_path}")
    print(f"2. 全局通用热力图已保存: {output_dir}/Global_Importance_Heatmap.png")

if __name__ == "__main__":
    run_global_correlation_analysis()