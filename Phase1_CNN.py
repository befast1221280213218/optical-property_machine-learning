"""
torchrun --nproc_per_node 8 \
    /inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Code/Phase1_CNN.py \
    --data_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1" \
    --result_file "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1/ART_result.txt" \
    --out_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/output_final" \
    --plot_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/plots" \
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
from transformers import (
    Trainer, 
    TrainingArguments, 
    set_seed,
    TrainerCallback
)

# --------------------------
# 1. 物理统一化 Dataset
# --------------------------
class PhysicsDataset(Dataset):
    def __init__(self, data_dir, result_file, target_size=(224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size
        # 加载并清洗结果文件列名
        df = pd.read_csv(result_file, sep=r'\s+', engine='python')
        df.columns = [c.replace(',', '').strip() for c in df.columns]
        self.results = df

    def __len__(self): return len(self.results)
    
    def __getitem__(self, idx):
        row = self.results.iloc[idx]
        # 提取物理参数并根据公式计算点间距 delta
        w, n, k = float(row['wavelength']), float(row['n']), float(row['k'])
        S, C, seed = float(row['S']), float(row['C']), int(row['seed'])
        delta = 0.0125 * w * C # 物理统一化核心公式
        
        phys_features = torch.tensor([w, n, k, S, C, delta], dtype=torch.float32)

        def fmt(val): return str(int(val)) if float(val).is_integer() else str(val)
        # 按照命名规则定位高程图文件
        filename = f"z(x,y)um_w{fmt(w)}_n{fmt(n)}_k{fmt(k)}_S{fmt(S)}_C{fmt(C)}_{seed}.txt"
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            matrix = np.loadtxt(file_path, delimiter='\t')
            img_t = torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)
            img_resized = F.interpolate(img_t, size=self.target_size, mode='bilinear').squeeze(0)
            return {
                "pixel_values": img_resized, 
                "physical_features": phys_features, 
                "labels": torch.tensor([row['R']]).float() # 目标为反射率 R
            }
        except Exception:
            return {"pixel_values": torch.zeros((1, *self.target_size)), "physical_features": phys_features, "labels": torch.tensor([0.0])}

# --------------------------
# 2. 模型与指标逻辑
# --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    preds, labels = logits.flatten(), labels.flatten()
    r2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    mape = np.mean(np.abs((labels - preds) / (labels + 1e-10))) * 100
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}

class MetricsLogCallback(TrainerCallback):
    def __init__(self, log_path, plot_dir):
        self.log_path = log_path
        self.plot_dir = plot_dir
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
        metrics_to_plot = ["r2", "rmse", "mae", "mape"]
        plt.figure(figsize=(15, 12))
        for i, m in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i+1)
            plt.plot(df["epoch"], df[m], marker='o')
            plt.title(f"{m.upper()} Convergence")
            plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "convergence_plots.png"))

# --------------------------
# 3. 主函数与参数解析
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    # 手动输入路径的参数
    parser.add_argument("--data_dir", type=str, required=True, help="结构.txt文件所在的目录")
    parser.add_argument("--result_file", type=str, required=True, help="ART_result.txt文件的完整路径")
    parser.add_argument("--out_dir", type=str, default="./output_h100", help="模型检查点输出目录")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="收敛图保存目录")
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    set_seed(42)

    full_dataset = PhysicsDataset(args.data_dir, args.result_file)
    train_size = int(0.9 * len(full_dataset))
    train_ds, eval_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset)-train_size])

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        eval_strategy="epoch", # 已修复之前的参数名错误
        save_strategy="epoch",
        bf16=True, tf32=True, # H100 硬件加速
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        greater_is_better=True,
        report_to="none",
        ddp_find_unused_parameters=False
    )

    # 简化的模型定义
    class MetamaterialRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            self.head = nn.Sequential(nn.Linear(64*7*7 + 6, 512), nn.GELU(), nn.Linear(512, 1))
            self.loss_fn = nn.MSELoss()
        def forward(self, pixel_values, physical_features, labels=None):
            feats = self.backbone(pixel_values).view(pixel_values.size(0), -1)
            logits = self.head(torch.cat([feats, physical_features], 1))
            loss = self.loss_fn(logits, labels) if labels is not None else None
            return {"loss": loss, "logits": logits} if loss is not None else logits

    log_callback = MetricsLogCallback(
        log_path=os.path.join(args.out_dir, "epoch_metrics.csv"),
        plot_dir=args.plot_dir
    )

    trainer = Trainer(
        model=MetamaterialRegressor(),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[log_callback]
    )

    trainer.train()

if __name__ == "__main__":
    main()