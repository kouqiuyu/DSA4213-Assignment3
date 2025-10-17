import torch
import numpy as np
import os
from transformers import Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Union
from visualization import TrainingCurveCallback, plot_confusion_matrix


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }


def train_model(
        model,
        processed_datasets,
        tokenizer,
        finetune_strategy: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        output_dir: str
):
    """训练模型并添加可视化回调"""
    # 创建可视化回调实例
    # 初始化可视化回调
    visualization_callback = TrainingCurveCallback()  # 类名同步修改

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{finetune_strategy}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/{finetune_strategy}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=True,
        log_level="warning",
        remove_unused_columns=False
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[visualization_callback]  # 添加可视化回调
    )

    # 开始训练
    print(f"\n=== 开始 {finetune_strategy.upper()} 微调 ===")
    trainer.train()

    # 训练结束后保存可视化曲线
    visualization_callback.plot_training_curves(
        save_dir=f"{output_dir}/{finetune_strategy}/plots",
        strategy=finetune_strategy
    )

    return trainer


def evaluate_model(trainer: Trainer, finetune_strategy: str) -> Dict[str, Union[str, float, int]]:
    """评估模型并生成混淆矩阵"""
    print(f"\n=== {finetune_strategy.upper()} 微调模型评估结果 ===")
    predictions = trainer.predict(trainer.eval_dataset)
    results = predictions.metrics

    # 提取预测结果和真实标签用于绘制混淆矩阵
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # 生成并保存混淆矩阵
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        strategy=finetune_strategy,
        save_dir=f"./imdb_finetune_results/{finetune_strategy}/plots"
    )

    # 打印评估结果
    print(results)
    print(f"测试集准确率: {results['test_accuracy']:.2%}")
    print(f"测试集F1分数: {results['test_f1']:.2%}")

    # 返回评估结果
    return {
        "strategy": finetune_strategy,
        "accuracy": results["test_accuracy"],
        "f1": results["test_f1"],
        "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    }


def analyze_error_cases(trainer: Trainer, processed_datasets, tokenizer, finetune_strategy: str, num_samples=50):
    """分析并保存错误案例"""
    predictions = trainer.predict(processed_datasets["test"])
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    error_indices = [i for i, (p, t) in enumerate(zip(pred_labels, true_labels)) if p != t]
    if not error_indices:
        print("没有发现错误案例")
        return

    sampled_error_indices = np.random.choice(
        error_indices,
        size=min(num_samples, len(error_indices)),
        replace=False
    )

    import pandas as pd
    error_cases = []
    for idx in sampled_error_indices:
        idx = int(idx)  # 转换为Python整数
        input_ids = processed_datasets["test"][idx]["input_ids"]
        original_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        error_cases.append({
            "文本": original_text[:200] + "..." if len(original_text) > 200 else original_text,
            "真实情感": "正面" if true_labels[idx] == 1 else "负面",
            "模型预测": "正面" if pred_labels[idx] == 1 else "负面",
            "微调策略": finetune_strategy
        })

    # 保存错误案例
    save_path = f"./imdb_finetune_results/{finetune_strategy}"
    os.makedirs(save_path, exist_ok=True)
    pd.DataFrame(error_cases).to_csv(
        f"{save_path}/error_cases_{finetune_strategy}.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\n已保存{finetune_strategy}策略的错误案例到 {save_path}/error_cases_{finetune_strategy}.csv")
