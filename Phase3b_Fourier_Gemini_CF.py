
"""
torchrun --nproc_per_node 8 --master_port 29510 \
    /inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Code/Phase3b_Fourier_Gemini_CF.py \
    --data_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1" \
    --result_file "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/Data/1/ART_result.txt" \
    --out_dir "/inspire/hdd/global_user/zhongzhiyan-253108050052/Article_Panjy/Article_1/output_phase3_CF" \
    --epochs 150 --bs 64 --lr 2e-4



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
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Subset
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
        w, n, k, S, C, seed = float(row['wavelength']), float(row['n']), float(row['k']), float(row['S']), float(row['C']), int(row['seed'])
        delta = 0.0125 * w * C  # 物理点间距公式
        phys_features = torch.tensor([w, n, k, S, C, delta], dtype=torch.float32)

        def fmt(val): return str(int(val)) if float(val).is_integer() else str(val)
        filename = f"z(x,y)um_w{fmt(w)}_n{fmt(n)}_k{fmt(k)}_S{fmt(S)}_C{fmt(C)}_{seed}.txt"
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            matrix = np.loadtxt(file_path, delimiter='\t')
            img_t = torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)
            img_spatial = F.interpolate(img_t, size=self.target_size, mode='bilinear').squeeze(0)
            
            # Phase 3: 傅里叶特征编码
            fft_res = torch.fft.rfft2(img_spatial)
            mag = torch.log1p(torch.abs(fft_res))
            phase = torch.angle(fft_res)
            
            mag_map = F.interpolate(mag.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
            phase_map = F.interpolate(phase.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
            
            # 空间域 + 频域幅度 + 频域相位
            combined_input = torch.cat([img_spatial, mag_map, phase_map], dim=0)
            return {
                "pixel_values": combined_input, 
                "physical_features": phys_features, 
                "labels": torch.tensor([row['R']]).float() # 目标反射率 R
            }
        except Exception:
            return {"pixel_values": torch.zeros((3, *self.target_size)), "physical_features": phys_features, "labels": torch.tensor([0.0])}

# --------------------------
# 2. 增强型 Gemini 3 架构
# --------------------------
class MetamaterialGemini(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 7, 2, 3)
        self.head = nn.Sequential(
            nn.Linear(64*7*7 + 6, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )
        self.loss_fn = nn.HuberLoss()

    def forward(self, pixel_values, physical_features, labels=None):
        x = F.silu(self.stem(pixel_values))
        x = F.adaptive_avg_pool2d(x, (7, 7)).view(x.size(0), -1)
        logits = self.head(torch.cat([x, physical_features], dim=1))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# --------------------------
# 3. 评估指标与详细日志回调
# --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    preds, labels = logits.flatten(), labels.flatten()
    return {
        "r2": r2_score(labels, preds),
        "rmse": np.sqrt(mean_squared_error(labels, preds)),
        "mae": mean_absolute_error(labels, preds),
        "mape": np.mean(np.abs((labels - preds) / (labels + 1e-10))) * 100
    }

class DetailedLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        entry = {"epoch": state.epoch}
        for k, v in metrics.items():
            if k.startswith("eval_"):
                entry[k.replace("eval_", "")] = v
        self.history.append(entry)
        pd.DataFrame(self.history).to_csv(self.log_path, index=False)

# --------------------------
# 4. 主执行逻辑
# --------------------------
def run_training(train_ds, eval_ds, output_dir, log_name, args):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True, tf32=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        ddp_find_unused_parameters=False
    )
    
    trainer = Trainer(
        model=MetamaterialGemini(),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[DetailedLoggerCallback(os.path.join(output_dir, log_name))]
    )
    trainer.train()
    return trainer.evaluate()

def main():
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--plot_dir", default="./plots_pro")
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    set_seed(42)
    dataset = FourierPhysicsDataset(args.data_dir, args.result_file)
    indices = list(range(len(dataset)))

    # --- 任务 1: 9:1 Hold-out 训练 ---
    print("\n🚀 启动 9:1 Hold-out 模式训练...")
    train_split = int(0.9 * len(dataset))
    train_ds_91 = Subset(dataset, indices[:train_split])
    test_ds_91 = Subset(dataset, indices[train_split:])
    
    holdout_dir = os.path.join(args.out_dir, "holdout_91")
    final_metrics = run_training(train_ds_91, test_ds_91, holdout_dir, "epoch_holdout_metrics.csv", args)
    print(f"✅ 9:1 模式测试集最终指标: {final_metrics}")

    # --- 任务 2: 5折交叉验证 ---
    print("\n🚀 启动 5折交叉验证...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for fold, (t_idx, v_idx) in enumerate(kf.split(indices)):
        print(f"--- 正在进行第 {fold+1} 折训练 ---")
        t_ds = Subset(dataset, t_idx)
        v_ds = Subset(dataset, v_idx)
        
        fold_dir = os.path.join(args.out_dir, f"fold_{fold+1}")
        fold_metrics = run_training(t_ds, v_ds, fold_dir, f"fold_{fold+1}_metrics.csv", args)
        fold_metrics["fold"] = fold + 1
        cv_results.append(fold_metrics)

    # 汇总交叉验证结果并另存为 CSV (相当于新 Sheet)
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(args.out_dir, "cv_summary_report.csv"), index=False)
    print(f"✅ 交叉验证平均 R2: {cv_df['eval_r2'].mean():.4f}")

if __name__ == "__main__":
    main()