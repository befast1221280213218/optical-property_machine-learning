"""
torchrun --nproc_per_node 8 --master_port 29505 \
    /inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Code/Phase3a_Fourier_Gemini.py \
    --data_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1" \
    --result_file "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1/ART_result.txt" \
    --out_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/output_phase3" \
    --plot_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/plots_phase3" \
    --epochs 150 --bs 64 --lr 3e-4

    

"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed, TrainerCallback

# --------------------------
# 1. 傅里叶特征提取与物理 Dataset
# --------------------------
class FourierPhysicsDataset(Dataset):
    def __init__(self, data_dir, result_file, target_size=(224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size
        # 加载并清洗 ART_result.txt
        df = pd.read_csv(result_file, sep=r'\s+', engine='python')
        df.columns = [c.replace(',', '').strip() for c in df.columns]
        self.results = df

    def __len__(self): return len(self.results)
    
    def __getitem__(self, idx):
        row = self.results.iloc[idx]
        # 物理参数
        w, n, k, S, C, seed = float(row['wavelength']), float(row['n']), float(row['k']), float(row['S']), float(row['C']), int(row['seed'])
        delta = 0.0125 * w * C # README 物理公式
        phys_features = torch.tensor([w, n, k, S, C, delta], dtype=torch.float32)

        def fmt(val): return str(int(val)) if float(val).is_integer() else str(val)
        filename = f"z(x,y)um_w{fmt(w)}_n{fmt(n)}_k{fmt(k)}_S{fmt(S)}_C{fmt(C)}_{seed}.txt"
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            # 读取大矩阵
            matrix = np.loadtxt(file_path, delimiter='\t')
            img_t = torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)
            # 空间域统一化
            img_spatial = F.interpolate(img_t, size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            
            # --- Phase 3: 傅里叶特征编码 ---
            # 对缩放后的图像进行 2D FFT
            fft_res = torch.fft.rfft2(img_spatial)
            mag = torch.abs(fft_res)
            phase = torch.angle(fft_res)
            
            # 将频域特征插值回与空间域相同大小，构造 3 通道输入 [Spatial, Mag, Phase]
            mag_map = F.interpolate(mag.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
            phase_map = F.interpolate(phase.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
            
            # 对幅度谱进行 Log 缩放以平滑数值
            mag_map = torch.log1p(mag_map)
            
            combined_input = torch.cat([img_spatial, mag_map, phase_map], dim=0) # [3, H, W]

            return {
                "pixel_values": combined_input, 
                "physical_features": phys_features, 
                "labels": torch.tensor([row['R']]).float() # 目标反射率 R
            }
        except Exception:
            return {"pixel_values": torch.zeros((3, *self.target_size)), "physical_features": phys_features, "labels": torch.tensor([0.0])}

# --------------------------
# 2. 增强型 Gemini 3 架构 (3-Channel Input + FiLM)
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
        # 输入通道由 1 变为 3 (Spatial + Fourier Mag + Phase)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU()
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
        self.loss_fn = nn.HuberLoss()

    def forward(self, pixel_values, physical_features, labels=None):
        x = self.stem(pixel_values)
        x = self.phys_gate(x, physical_features)
        res = self.shortcut(x)
        x = F.silu(self.res_blocks(x) + res)
        x = x * self.attn(x)
        x_feat = self.global_pool(x).view(x.size(0), -1)
        logits = self.regressor(torch.cat([x_feat, physical_features], dim=1))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# --------------------------
# 3. 指标与回调逻辑
# --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds, labels = logits.flatten(), labels.flatten()
    return {
        "r2": r2_score(labels, preds),
        "rmse": np.sqrt(mean_squared_error(labels, preds)),
        "mae": mean_absolute_error(labels, preds),
        "mape": np.mean(np.abs((labels - preds) / (labels + 1e-10))) * 100
    }

class Phase3Callback(TrainerCallback):
    def __init__(self, log_path, plot_dir):
        self.log_path, self.plot_dir = log_path, plot_dir
        self.history = []
        os.makedirs(plot_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_r2" in metrics:
            entry = {"epoch": state.epoch, "r2": metrics["eval_r2"], "rmse": metrics["eval_rmse"], "mae": metrics["eval_mae"], "mape": metrics["eval_mape"]}
            self.history.append(entry)
            pd.DataFrame(self.history).to_csv(self.log_path, index=False)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.history: return
        df = pd.DataFrame(self.history)
        plt.figure(figsize=(15, 12))
        for i, m in enumerate(["r2", "rmse", "mae", "mape"]):
            plt.subplot(2, 2, i+1); plt.plot(df["epoch"], df[m], 'r-o')
            plt.title(f"{m.upper()} Convergence (Fourier Phase)"); plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "phase3_fourier_metrics.png"))

# --------------------------
# 4. 执行入口
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True); parser.add_argument("--result_file", required=True)
    parser.add_argument("--out_dir", default="./output_phase3"); parser.add_argument("--plot_dir", default="./plots_phase3")
    parser.add_argument("--epochs", type=int, default=150); parser.add_argument("--bs", type=int, default=64); parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    set_seed(42)
    dataset = FourierPhysicsDataset(args.data_dir, args.result_file)
    train_size = int(0.9 * len(dataset))
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    training_args = TrainingArguments(
        output_dir=args.out_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.bs,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        bf16=True, tf32=True, load_best_model_at_end=True, metric_for_best_model="r2", greater_is_better=True,
        ddp_find_unused_parameters=False, report_to="none"
    )

    trainer = Trainer(
        model=MetamaterialFourierGemini(),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[Phase3Callback(os.path.join(args.out_dir, "phase3_metrics.csv"), args.plot_dir)]
    )

    trainer.train()

if __name__ == "__main__":
    main()