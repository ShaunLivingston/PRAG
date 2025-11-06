# src/inference.py
# ============================================
# Parametric RAG - Inference with Rank-Weighted LoRA Fusion
# 改动要点：
# 1) 新增 --fusion_weighting 与 --tau 参数；
# 2) 在合并 LoRA 适配器前，按排名/等权计算权重；
# 3) 可选将最终权重写入 predict.json 便于复现实验。
# ============================================

import os
import gc
import json
import math
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete


def _softmax(xs, tau: float = 1.0):
    """数值稳定的 softmax；tau 越小分布越尖。"""
    if not xs:
        return []
    if tau <= 0:
        tau = 1e-6
    m = max(xs)
    exps = [math.exp((x - m) / tau) for x in xs]
    s = sum(exps)
    if s <= 0:
        # 理论上不会发生；兜底等权
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def main(args):
    # 载入增强后的数据切分
    data_list = load_data(args.dataset, args.data_type, args.augment_model)

    # 加载生成模型
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    cot_name = "cot" if args.with_cot else "direct"

    # LoRA 离线参数路径
    load_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )

    # 推理输出路径
    output_root_dir = os.path.join(
        ROOT_DIR,
        "output",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method,
    )

    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")

        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        # 记录当前运行配置
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4, ensure_ascii=False)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        # 子集/断点续跑
        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]

        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(_model, psgs, _weights=None):
                text = predict(_model, tokenizer, generation_config,
                               question, with_cot=args.with_cot,
                               passages=psgs)
                pred = {
                    "test_id": test_id,
                    "question": question,
                    "answer": answer,
                    "text": text,
                }
                # 新增部分：写入融合权重，便于复现实验（仅在有意义时记录）
                if _weights is not None:
                    pred["fusion_weighting"] = args.fusion_weighting
                    pred["weights"] = [round(float(w), 6) for w in _weights]
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            # 新增部分：纯 ICL：不做 LoRA 合并
            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages, _weights=None))

            else:
                # 逐段加载离线 LoRA
                K = len(passages)
                for pid in range(K):
                    adapter_path = os.path.join(
                        load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}"
                    )
                    if pid == 0:
                        model = PeftModel.from_pretrained(
                            model,
                            adapter_path,
                            adapter_name="0",
                            is_trainable=False
                        )
                    else:
                        model.load_adapter(adapter_path, adapter_name=str(pid))

                # === 计算融合权重 ===
                if args.fusion_weighting == "equal":
                    weights = [1.0] * K
                elif args.fusion_weighting == "rank":
                    # 排名越靠前，权重越大：raw = -rank
                    raw_scores = [-(i) for i in range(K)]
                    weights = _softmax(raw_scores, tau=args.tau)
                else:
                    # 兜底（理论上不会走到）
                    weights = [1.0] * K

                # 合并多 LoRA
                model.add_weighted_adapter(
                    adapters=[str(i) for i in range(K)],
                    weights=weights,
                    adapter_name="merge",
                    combination_type="cat",
                )
                model.set_adapter("merge")

                # PRAG：不再拼接 passages 文本（减少上下文）；COMBINE：两者并用
                use_psgs = None if args.inference_method == "prag" else passages
                ret.append(get_pred(model, psgs=use_psgs, _weights=weights))

                # 清理合并适配器并释放显存
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        # 持久化预测
        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4, ensure_ascii=False)

        # 汇总指标
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / max(len(ret), 1)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4, ensure_ascii=False)

        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1)  # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "prag", "combine"])
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)

    # === 新增：融合方式 & 温度（默认即为推荐的 rank 加权） ===
    parser.add_argument("--fusion_weighting", type=str, default="rank",
                        choices=["equal", "rank"],
                        help="LoRA 融合权重来源：equal(等权)、rank(按排名softmax)")
    parser.add_argument("--tau", type=float, default=0.7,
                        help="softmax 温度，越小越偏爱前排段落")

    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
