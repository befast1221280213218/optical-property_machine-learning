import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------
# 1. 核心架构定义 (确保与您 Phase3a 训练代码完全对齐)
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
        logits = self.head_fusion(x_feat, physical_features)
        return logits

    def head_fusion(self, x_feat, physical_features):
        return self.regressor(torch.cat([x_feat, physical_features], dim=1))

# --------------------------
# 2. 全局统计分析引擎
# --------------------------
def run_global_xai_analysis():
    # 路径配置
    data_dir = "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1"
    res_file = os.path.join(data_dir, "ART_result.txt")
    # ✅ 指向您训练好的最优 checkpoint
    model_weight_path = "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/output_phase3/checkpoint-best/pytorch_model.bin"
    output_dir = "./Phase3c_Global_XAI_Results"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集 (使用您 Phase3a 中的 Dataset 类逻辑)
    from Phase3a_Fourier_Gemini import FourierPhysicsDataset 
    dataset = FourierPhysicsDataset(data_dir, res_file)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 初始化并加载最优模型
    model = MetamaterialFourierGemini().to(device)
    if os.path.exists(model_weight_path):
        # 兼容 Transformers Trainer 的权重加载
        state_dict = torch.load(model_weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 已成功调用最优模型权重进行深度分析。")

    model.eval()
    global_saliency = np.zeros((224, 224))
    total_count = 0

    print("📊 正在执行全量数据像素关联性扫描...")
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(device)
        phys_features = batch["physical_features"].to(device)
        
        # 核心：开启像素级梯度追踪
        pixel_values.requires_grad = True
        
        logits = model(pixel_values, phys_features)
        
        # 对 R 值求和并反向传播，提取“像素影响力”
        model.zero_grad()
        logits.sum().backward()
        
        # 提取空间域通道 (Channel 0) 的绝对梯度
        # 这代表了每个 (x,y) 像素点对最终 R 值的敏感度
        grads = pixel_values.grad.data.abs()[:, 0, :, :].cpu().numpy()
        global_saliency += np.sum(grads, axis=0)
        total_count += pixel_values.size(0)

    # 3. 统计平均与标准化
    avg_saliency = global_saliency / total_count
    # 归一化到 [0, 1] 方便观察相关性强度
    norm_saliency = (avg_saliency - avg_saliency.min()) / (avg_saliency.max() - avg_saliency.min() + 1e-10)

    # 4. 输出像素点关联权重矩阵 (.txt)
    # 每一行每一列对应您高程矩阵的一个像素点
    txt_path = os.path.join(output_dir, "Global_Pixel_R_Correlation.txt")
    np.savetxt(txt_path, norm_saliency, delimiter='\t', fmt='%.6f')

    # 5. 综合相关性热力图可视化
    plt.figure(figsize=(10, 8))
    plt.imshow(norm_saliency, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Pixel-to-R Correlation Weight')
    plt.title("Global Statistical Sensitivity Map\n(Where AI looks for Reflectivity)")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.savefig(os.path.join(output_dir, "Global_R_Correlation_Heatmap.png"), dpi=300)

    print(f"🚀 分析完成！")
    print(f"1. 全局像素关联矩阵已保存至: {txt_path}")
    print(f"2. 综合热力图已保存至: {output_dir}/Global_R_Correlation_Heatmap.png")

if __name__ == "__main__":
    run_global_xai_analysis()