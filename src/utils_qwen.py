import os
import re
import json
import torch
import string
import numpy as np
from collections import Counter
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

# 强制离线
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from root_dir_path import ROOT_DIR
from prompt_template import get_prompt

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data_aug")


# =========================
# 稳健答案抽取，避免早截断
# =========================
def _extract_answer(text: str, with_cot: bool = False) -> str:
    """
    - 优先抽取 "The answer is: xxx" / "Answer: xxx"（大小写不敏感，中英兼容）
    - 否则取第一行（不再用 '.' 或 ',' 早截断，避免切掉实体名）
    - 去掉首尾引号和标点
    """
    if text is None:
        return ""

    s = text.strip()

    # 常见答案提示模式
    patterns = [
        r"(?:the\s+answer\s+is\s*:?\s*)(.+)",  # The answer is: xxx
        r"(?:answer\s*:?\s*)(.+)",  # Answer: xxx
        r"(?:答案是[:：]\s*)(.+)",  # 答案是：xxx
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            s = m.group(1).strip()
            break

    # 仅在换行处分段（避免 '.' ',' 过早截断）
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()

    # 清理首尾标点
    s = s.strip().strip("\"'“”‘’：:.,，。!！?？").strip()
    return s


class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
            cls,
            prediction: str,
            ground_truth: Union[str, List[str]],
            ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str) and hasattr(cls, "get_all_alias"):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        pred_n = cls.normalize_answer(prediction)
        correct = np.max([int(pred_n == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
            cls,
            prediction: str,
            ground_truth: Union[str, List[str]],
            ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str) and hasattr(cls, "get_all_alias"):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no',
                                           'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / max(len(prediction_tokens), 1)
            recall = 1.0 * num_same / max(len(ground_truth_tokens), 1)
            f1 = (2 * precision * recall) / max((precision + recall), 1e-8)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


def load_data(data_name, data_type, model_name):
    solve_dataset = []
    input_dir = os.path.join(DATA_ROOT_DIR, data_name, model_name)
    files = [f for f in os.listdir(input_dir)]

    if len(files) > 1:  # more types in dataset
        if data_type == "total":  # merge all types to total
            all_data = {}
            for filename in files:
                with open(os.path.join(input_dir, filename), "r") as fin:
                    all_data[filename] = json.load(fin)
            total_data = []
            idx = {filename: 0 for filename in files}
            for data in all_data["total.json"]:
                typ = data["type"] + ".json"
                if idx[typ] == len(all_data[typ]):
                    break
                aim_data = all_data[typ][idx[typ]]
                assert aim_data["question"] == data["question"]
                idx[typ] += 1
                total_data.append(aim_data)
            return [["total.json", total_data]]
        for filename in files:
            if filename != "total.json":
                with open(os.path.join(input_dir, filename), "r") as fin:
                    solve_dataset.append((filename, json.load(fin)))
        if data_type is None:
            return solve_dataset
        else:
            data_type = data_type + ".json"
            if data_type not in [v[0] for v in solve_dataset]:
                raise ValueError(f"Invalid {data_type} in Dataset {data_name}")
            tmp = []
            for filename, dataset in solve_dataset:
                if filename == data_type:
                    tmp.append((filename, dataset))
            return tmp
    else:
        with open(os.path.join(input_dir, "total.json"), "r") as fin:
            solve_dataset.append(("total.json", json.load(fin)))
        return solve_dataset


def get_model_path(model_name):
    # === 保持你本地的模型路径映射，不做变更 ===
    local_model_paths = {
        "qwen2.5-1.5b-instruct": "/home/dj/home/dj/PRAG-main/model/Qwen2.5-1.5B-Instruct",
    }
    if model_name in local_model_paths:
        local_path = local_model_paths[model_name]
        if os.path.isdir(local_path):
            print(f"[Local Model] Using {model_name} at {local_path}")
            return local_path
        raise FileNotFoundError(
            f"Local path not found for {model_name}: {local_path}\n"
            f"Please check the path and ensure required files exist."
        )
    raise ValueError(f"Unknown model_name: {model_name}. Please add it to get_model_path().")


def get_model(model_name, max_new_tokens=20):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    generation_config = dict(
        num_beams=1,
        do_sample=False,  # 论文设置：贪心
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    return model, tokenizer, generation_config


# -------------------------------- for augmentation ----------------------------------------
def model_generate(prompt, model, tokenizer, generation_config):
    messages = [{
        'role': 'user',
        'content': prompt,
    }]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True
    )
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    output = model.generate(
        input_ids,
        attention_mask=torch.ones(input_ids.shape).to(model.device),
        **generation_config
    )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text


# ------------------------------------------------------------------------------------


def read_complete(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        return data, len(data)
    except Exception:
        return [], 0


def evaluate(pred, ground_truth, with_cot=False):
    # 使用稳健抽取器；不再用 '.' / ',' 早截断
    pred = _extract_answer(pred, with_cot=with_cot)

    em = BaseDataset.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def _format_passages(passages: List[str]) -> str:
    if not passages:
        return ""
    lines = []
    for i, p in enumerate(passages):
        p = p.strip()
        if p:
            lines.append(f"Passage {i + 1}: {p}")
    return "\n".join(lines)


def predict(model, tokenizer, generation_config, question, with_cot, passages=None):
    """
    Qwen 统一走 chat_template，强制英文短答、无解释、无标点；
    其他模型保持原有 prompt_template.get_prompt 逻辑。
    """
    model.eval()

    name_or_path = (getattr(tokenizer, "name_or_path", "") or "").lower()
    use_qwen_chat = ("qwen" in name_or_path)

    if use_qwen_chat:
        sys_msg = (
            "You are a helpful assistant. "
            "Answer in English with a short noun phrase. "
            "Output ONLY the final answer, no punctuation, no explanation."
        )
        ctx = _format_passages(passages)
        if ctx:
            user_msg = f"{ctx}\nQuestion: {question}"
        else:
            user_msg = f"Question: {question}"

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids).to(model.device),
                **generation_config
            )
        sequences = output.sequences[0]
        gen = tokenizer.decode(sequences[input_ids.shape[1]:], skip_special_tokens=True)
        return gen

    # === 非 Qwen：沿用原始模板 ===
    input_ids = get_prompt(
        tokenizer,
        question,
        passages=passages,
        with_cot=with_cot
    )
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=torch.ones(input_ids.shape).to(model.device),
            **generation_config
        )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text
