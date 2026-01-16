"""
torchrun --nproc_per_node 8 \
    /inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Code/Phase2_Gemini_Optimized.py \
    --data_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1" \
    --result_file "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1/ART_result.txt" \
    --out_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/output_final/Phase2_Gemini_Optimized" \
    --plot_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/plots/Phase2_Gemini_Optimized" \
    --epochs 100 --bs 64 --lr 2e-4




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
# 1. 物理条件化层 (FiLM) - Gemini 3 核心创新点
# --------------------------
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, phys_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(phys_dim, phys_dim * 2),
            nn.SiLU(),
            nn.Linear(phys_dim * 2, feature_dim * 2)
        )

    def forward(self, x, phys):
        params = self.mlp(phys).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return x * gamma + beta

# --------------------------
# 2. 优化后的模型架构
# --------------------------
class MetamaterialGeminiRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.phys_gate1 = FiLMLayer(64, 6)
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.SiLU(),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(128 + 6, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.loss_fn = nn.HuberLoss(delta=1.0)

    def forward(self, pixel_values, physical_features, labels=None):
        x = self.stem(pixel_values)
        x = self.phys_gate1(x, physical_features)
        residual = self.shortcut(x)
        x = self.res_block(x)
        x = F.silu(x + residual)
        x = x * self.attn(x)
        x_feat = self.global_pool(x).view(x.size(0), -1)
        combined = torch.cat([x_feat, physical_features], dim=1)
        logits = self.regressor(combined)
        
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# --------------------------
# 3. 数据集与监控逻辑
# --------------------------
class PhysicsDataset(Dataset):
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
            img_resized = F.interpolate(img_t, size=self.target_size, mode='bilinear').squeeze(0)
            return {"pixel_values": img_resized, "physical_features": phys_features, "labels": torch.tensor([row['R']]).float()}
        except Exception:
            return {"pixel_values": torch.zeros((1, *self.target_size)), "physical_features": phys_features, "labels": torch.tensor([0.0])}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds, labels = logits.flatten(), labels.flatten()
    return {
        "r2": r2_score(labels, preds),
        "rmse": np.sqrt(mean_squared_error(labels, preds)),
        "mae": mean_absolute_error(labels, preds),
        "mape": np.mean(np.abs((labels - preds) / (labels + 1e-10))) * 100
    }

class MetricsLogCallback(TrainerCallback):
    def __init__(self, log_path, plot_dir):
        self.log_path, self.plot_dir = log_path, plot_dir
        self.history = []
        os.makedirs(plot_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_r2" in metrics:
            entry = {"epoch": state.epoch, "r2": metrics["eval_r2"], "rmse": metrics["eval_rmse"], 
                     "mae": metrics["eval_mae"], "mape": metrics["eval_mape"]}
            self.history.append(entry)
            pd.DataFrame(self.history).to_csv(self.log_path, index=False)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.history: return
        df = pd.DataFrame(self.history)
        plt.figure(figsize=(12, 10))
        for i, m in enumerate(["r2", "rmse", "mae", "mape"]):
            plt.subplot(2, 2, i+1); plt.plot(df["epoch"], df[m], 'b-o')
            plt.title(f"{m.upper()} Convergence"); plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "gemini_optimized_metrics.png"))

# --------------------------
# 4. 主函数：补全所有参数
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./output_gemini")
    parser.add_argument("--plot_dir", type=str, default="./plots")
    
    # ✅ 补全训练超参数 (解决 error: unrecognized arguments 报错)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=64, help="per_device_train_batch_size")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    # 加载数据集并划分
    full_dataset = PhysicsDataset(args.data_dir, args.result_file)
    train_size = int(0.9 * len(full_dataset))
    train_ds, eval_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        eval_strategy="epoch",  # 使用修复后的参数名
        save_strategy="epoch",
        bf16=True, tf32=True,
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        greater_is_better=True,
        report_to="none",
        ddp_find_unused_parameters=False
    )

    log_callback = MetricsLogCallback(
        log_path=os.path.join(args.out_dir, "gemini_metrics.csv"),
        plot_dir=args.plot_dir
    )

    trainer = Trainer(
        model=MetamaterialGeminiRegressor(),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[log_callback]
    )

    trainer.train()

if __name__ == "__main__":
    main()