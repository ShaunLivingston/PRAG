# src/inference_rank.py
import os
import gc
import json
import math
import argparse
import importlib
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers.generation.configuration_utils",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="peft.tuners.tuners_utils",
)
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR


def _load_utils(model_name: str):
    """按模型名动态选择 utils 模块：qwen -> utils_qwen；llama -> utils_llama。"""
    name = (model_name or "").lower()
    if "qwen" in name:
        candidates = ["utils_qwen"]
    elif "llama" in name:
        candidates = ["utils_llama"]
    else:
        candidates = ["utils"]  # 兜底
    last_err = None
    for mod in candidates:
        try:
            return importlib.import_module(mod)
        except Exception as e:
            last_err = e
    raise last_err or ImportError(f"Cannot import utils module for model_name={model_name}")


def _softmax(xs, tau: float = 1.0):
    if not xs:
        return []
    if tau <= 0:
        tau = 1e-6
    m = max(xs)
    exps = [math.exp((x - m) / tau) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def _print_model_debug(args, model, tokenizer):
    try:
        print("model_name:", args.model_name)
        print("HF id / local path:", getattr(tokenizer, "name_or_path", "N/A"))
        print("hidden_size / n_layers:", getattr(model.config, "hidden_size", "N/A"),
              getattr(model.config, "num_hidden_layers", "N/A"))
    except Exception as e:
        print("[WARN] model debug print failed:", e)


def _parse_float_list(s: str):
    if not s:
        return []
    vals = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def _apply_floor(base_weights, floor, K: int):
    """将 softmax 权重加上最小权重下限 floor，保证每段至少有 floor 的占比。"""
    # 约束 floor < 1/K
    if floor >= 1.0 / max(K, 1):
        floor = (1.0 / max(K, 1)) - 1e-6
    if floor <= 0:
        return base_weights
    coef = max(1e-6, 1.0 - K * floor)
    return [floor + coef * b for b in base_weights]


def main(args):
    U = _load_utils(args.model_name)

    # 解析 sweep 网格（若提供 *_grid，则忽略单值）
    tau_list = _parse_float_list(args.tau_grid) if args.tau_grid else [args.tau]
    wf_list = _parse_float_list(args.weight_floor_grid) if args.weight_floor_grid else [args.weight_floor]

    # 载入数据与模型
    data_list = U.load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, generation_config = U.get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    _print_model_debug(args, model, tokenizer)

    cot_name = "cot" if args.with_cot else "direct"
    base_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )

    # === 对每个 (tau, weight_floor) 组合逐一运行 ===
    for tau in tau_list:
        for floor in wf_list:

            # 输出目录加入组合标识
            fusion_tag = f"fusion={args.fusion_weighting}"
            if args.fusion_weighting == "rank":
                fusion_tag += f"_tau={tau}_wf={floor}"

            output_root_dir = os.path.join(
                ROOT_DIR,
                "output",
                args.model_name,
                f"rank={args.lora_rank}_alpha={args.lora_alpha}",
                args.dataset,
                f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
                f"aug_model={args.augment_model}",
                args.inference_method,
                fusion_tag,
            )

            for filename, fulldata in data_list:
                filename = filename.split(".")[0]
                print(f"### Solving {filename} ###  (tau={tau}, floor={floor})")

                output_dir = os.path.join(output_root_dir, filename)
                os.makedirs(output_dir, exist_ok=True)

                # 记录本组合的运行配置
                cfg = vars(args).copy()
                cfg.update({"tau": float(tau), "weight_floor": float(floor)})
                with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as fout:
                    json.dump(cfg, fout, indent=4, ensure_ascii=False)

                predict_file = os.path.join(output_dir, "predict.json")
                ret, start_with = U.read_complete(predict_file)
                if args.overwrite:
                    ret, start_with = [], 0

                fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]

                # 新增：只在每个文件里第一次打印 FUSION 信息
                first_fusion_log = True

                for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
                    test_id = test_id + start_with
                    assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

                    question = data["question"]
                    passages = data["passages"]
                    answer = data["answer"]

                    def get_pred(_model, psgs, _weights=None):
                        text = U.predict(_model, tokenizer, generation_config,
                                         question, with_cot=args.with_cot,
                                         passages=psgs)
                        pred = {
                            "test_id": test_id,
                            "question": question,
                            "answer": answer,
                            "text": text,
                        }
                        if _weights is not None:
                            pred["fusion_weighting"] = args.fusion_weighting
                            pred["tau"] = float(tau)
                            pred["weight_floor"] = float(floor)
                            pred["weights"] = [round(float(w), 6) for w in _weights]
                        pred.update(U.evaluate(text, answer, args.with_cot))
                        return pred

                    if args.inference_method == "icl":
                        ret.append(get_pred(model, psgs=passages, _weights=None))
                    else:
                        K = len(passages)
                        # 加载每个 passage 的 LoRA
                        for pid in range(K):
                            adapter_path = os.path.join(
                                base_adapter_path, filename, f"data_{test_id}", f"passage_{pid}"
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

                        # 计算融合权重
                        if args.fusion_weighting == "rank":
                            raw = [-(i) for i in range(K)]
                            base = _softmax(raw, tau=tau)
                            weights = _apply_floor(base, floor=floor, K=K)
                        else:  # equal
                            weights = [1.0] * K

                        # 修改：只在第一次打印
                        if first_fusion_log:
                            from tqdm import tqdm as _tqdm  # 顶部已导入则不需要这一行
                            _tqdm.write(
                                f"[FUSION] mode={args.fusion_weighting} K={K}, "
                                f"tau={tau}, floor={floor}, "
                                f"weights={[round(w, 4) for w in weights]}"
                            )
                            first_fusion_log = False

                        # 合并
                        model.add_weighted_adapter(
                            adapters=[str(i) for i in range(K)],
                            weights=weights,
                            adapter_name="merge",
                            combination_type="cat",
                        )
                        model.set_adapter("merge")

                        use_psgs = None if args.inference_method == "prag" else passages
                        ret.append(get_pred(model, psgs=use_psgs, _weights=weights))

                        # 清理
                        try:
                            model.delete_adapter("merge")
                        except Exception:
                            pass
                        try:
                            model = model.unload()
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                        gc.collect()

                with open(predict_file, "w", encoding="utf-8") as fout:
                    json.dump(ret, fout, indent=4, ensure_ascii=False)

                metrics = ["em", "f1", "prec", "recall"]
                ret_str = ""
                for met in metrics:
                    acc = sum(float(d[met]) for d in ret) / max(len(ret), 1)
                    acc = round(acc, 4)
                    ret_str += f"{met}\t{acc}\n"
                ret_str += "\n" + json.dumps(cfg, indent=4, ensure_ascii=False)
                with open(os.path.join(output_dir, "result.txt"), "w", encoding="utf-8") as fout:
                    fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--augment_model", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "prag", "combine"])
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    # rank 融合参数
    parser.add_argument("--tau", type=float, default=0.7, help="softmax temperature for rank fusion (single value)")
    parser.add_argument("--weight_floor", type=float, default=0.0,
                        help="minimum weight per passage for rank fusion, 0 <= floor < 1/K")
    # 扫描网格（逗号分隔，提供后将覆盖单值）
    parser.add_argument("--tau_grid", type=str, default="", help="comma-separated tau values to sweep")
    parser.add_argument("--weight_floor_grid", type=str, default="", help="comma-separated floor values to sweep")
    parser.add_argument("--fusion_weighting", type=str, default="rank", choices=["rank", "equal"],
                        help="LoRA fusion weighting mode")
    parser.add_argument("--overwrite", action="store_true", help="ignore/overwrite existing predict.json")
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
