from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple


def load_and_preprocess_data(
        model_name: str = "distilbert-base-uncased",
        max_seq_len: int = 512
) -> Tuple[dict, AutoTokenizer]:
    """
    加载IMDb数据集并进行预处理（分词、格式转换）

    Args:
        model_name: 预训练模型名称（用于匹配对应的Tokenizer）
        max_seq_len: 文本最大序列长度（截断/填充依据）

    Returns:
        processed_datasets: 预处理后的数据集（train/test）
        tokenizer: 用于分词的Tokenizer
    """
    # 1. 加载IMDb数据集（自动下载，约100MB）
    dataset = load_dataset("imdb")  # 输出格式：{'train': Dataset, 'test': Dataset}

    # 2. 加载模型对应的Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 为DistilBERT设置pad_token

    # 3. 定义分词函数（批量处理文本）
    def tokenize_function(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            max_length=max_seq_len,
            truncation=True,  # 超过max_seq_len截断
            padding="max_length",  # 不足max_seq_len填充
            return_overflowing_tokens=False
        )

    # 4. 对数据集批量分词（移除原始text列，保留label和分词结果）
    processed_datasets = dataset.map(
        tokenize_function,
        batched=True,  # 批量处理以提高效率
        remove_columns=["text"],  # 移除不需要的原始文本列
        desc="Tokenizing IMDb dataset"
    )

    # 5. 转换为PyTorch张量格式（适配模型输入）
    processed_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return processed_datasets, tokenizer