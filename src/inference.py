import os
import gc
import json
import argparse
import torch
import importlib
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR

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


def _load_utils(model_name: str):
    """
    按模型名选择 utils 模块:
    - 模型名包含 'qwen' -> utils_qwen
    - 模型名包含 'llama' -> utils_llama
    - 否则回退到原来的 utils
    """
    name = (model_name or "").lower()

    candidates = []
    if "qwen" in name:
        candidates.append("utils_qwen")
    if "llama" in name:
        candidates.append("utils_llama")

    # 如果既不是 qwen 也不是 llama，或者上面导入失败，则回退到通用 utils
    candidates.append("utils")

    last_err = None
    for mod in candidates:
        try:
            return importlib.import_module(mod)
        except Exception as e:
            last_err = e
    raise last_err or ImportError(f"Cannot import utils module for model_name={model_name}")



def main(args):
    # 1. 根据推理模型名动态选择对应的 utils 模块
    utils_mod = _load_utils(args.model_name)

    # 2. 后续所有 utils 相关函数都从该模块调用
    data_list = utils_mod.load_data(args.dataset, args.data_type, args.augment_model)

    model, tokenizer, generation_config = utils_mod.get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    cot_name = "cot" if args.with_cot else "direct"
    load_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
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
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = utils_mod.read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs):
                text = utils_mod.predict(
                    model,
                    tokenizer,
                    generation_config,
                    question,
                    with_cot=args.with_cot,
                    passages=psgs,
                )
                pred = {
                    "test_id": test_id,
                    "question": question,
                    "answer": answer,
                    "text": text,
                }
                pred.update(utils_mod.evaluate(text, answer, args.with_cot))
                return pred

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            else:
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
                    if pid == 0:
                        model = PeftModel.from_pretrained(
                            model,
                            adapter_path,
                            adapter_name="0",
                            is_trainable=False
                        )
                    else:
                        model.load_adapter(adapter_path, adapter_name=str(pid))
                        # merge
                model.add_weighted_adapter(
                    adapters=[str(i) for i in range(len(passages))],
                    weights=[1] * len(passages),
                    adapter_name="merge",
                    combination_type="cat",
                )
                model.set_adapter("merge")
                ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
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
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
