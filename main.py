import torch
import os
import argparse  # 新增：用于解析命令行参数
from data_processing import load_and_preprocess_data
from model_setup import init_full_finetune_model, init_lora_finetune_model
from train_eval import train_model, evaluate_model, analyze_error_cases


def main():
    # --------------------------
    # 解析命令行参数（新增部分）
    # --------------------------
    parser = argparse.ArgumentParser(description="IMDb情感分析：全量微调与LoRA微调对比实验")
    parser.add_argument(
        "--strategy",
        type=str,
        default="full",
        choices=["full", "lora"],
        help="微调策略：full（全量微调）或 lora（LoRA微调），默认full"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮次，默认3"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批量大小，默认16"
    )
    parser.add_argument(
        "--no_analyze",
        action="store_true",
        help="不分析错误案例（默认会分析）"
    )
    args = parser.parse_args()

    # --------------------------
    # 参数设置（基于命令行输入）
    # --------------------------
    finetune_strategy = args.strategy
    batch_size = args.batch_size
    epochs = args.epochs
    analyze_errors_flag = not args.no_analyze  # 反转：--no_analyze则不分析
    learning_rate = None  # 为None时使用默认值（full=2e-5, lora=3e-4）
    output_dir = "./imdb_finetune_results"  # 结果保存根目录

    # 设置默认学习率
    if learning_rate is None:
        learning_rate = 2e-5 if finetune_strategy == "full" else 3e-4

    # --------------------------
    # 环境与数据准备
    # --------------------------
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cpu":
        print("警告：未检测到GPU，训练可能较慢")

    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("\n=== 加载并预处理IMDb数据集 ===")
    processed_datasets, tokenizer = load_and_preprocess_data(
        model_name="distilbert-base-uncased",
        max_seq_len=512
    )
    print(f"训练集样本数: {len(processed_datasets['train'])}")
    print(f"测试集样本数: {len(processed_datasets['test'])}")

    # --------------------------
    # 模型初始化与训练
    # --------------------------
    # 初始化模型
    print(f"\n=== 初始化 {finetune_strategy.upper()} 微调模型 ===")
    if finetune_strategy == "full":
        model = init_full_finetune_model()
    else:
        model = init_lora_finetune_model(
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
    model.to(device)

    # 训练模型
    print(f"\n=== 开始 {finetune_strategy.upper()} 微调（{epochs}轮） ===")
    trainer = train_model(
        model=model,
        processed_datasets=processed_datasets,
        tokenizer=tokenizer,
        finetune_strategy=finetune_strategy,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        output_dir=output_dir
    )

    # --------------------------
    # 评估与结果分析
    # --------------------------
    # 评估模型
    eval_results = evaluate_model(trainer, finetune_strategy)

    # 分析错误案例（可选）
    if analyze_errors_flag:
        analyze_error_cases(
            trainer=trainer,
            processed_datasets=processed_datasets,
            tokenizer=tokenizer,
            finetune_strategy=finetune_strategy,
            num_samples=50  # 保存50个错误案例
        )

    # 打印总结
    print("\n=== 实验总结 ===")
    print(f"微调策略: {eval_results['strategy'].upper()}")
    print(f"可训练参数: {eval_results['trainable_params']:,}")
    print(f"最终准确率: {eval_results['accuracy']:.2%}")
    print(f"最终F1分数: {eval_results['f1']:.2%}")
    print(f"结果保存路径: {output_dir}/{finetune_strategy}")


if __name__ == "__main__":
    main()