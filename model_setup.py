import torch
from typing import Union, Dict  # 新增导入 Union 和 Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)  # 隐藏模型加载提示


def init_full_finetune_model(
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2  # IMDb为二分类（0=负面，1=正面）
) -> AutoModelForSequenceClassification:
    """
    初始化全量微调模型（所有参数可训练）

    Args:
        model_name: 预训练模型名称
        num_labels: 分类任务的类别数

    Returns:
        model: 全量微调模型
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # 忽略预训练头与分类头的尺寸不匹配
    )

    # 确保所有参数可训练（全量微调核心）
    for param in model.parameters():
        param.requires_grad = True

    return model


from peft import LoraConfig, get_peft_model

def init_lora_finetune_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1
) -> PeftModel:
    # 加载基础模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    # 定义 LoRA 配置（关键：修改 target_modules 为 DistilBERT 的模块名）
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # DistilBERT 的注意力和前馈层模块名
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "fc1", "fc2"],
        bias="none"
    )
    # 注入 LoRA 适配器
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()  # 打印可训练参数信息
    return peft_model


def count_trainable_params(model: Union[AutoModelForSequenceClassification, PeftModel]) -> Dict[str, int]:
    """
    统计模型的可训练参数与总参数数量

    Args:
        model: 待统计的模型

    Returns:
        params_stats: 包含可训练参数与总参数的字典
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": round(trainable_params / total_params * 100, 4)
    }