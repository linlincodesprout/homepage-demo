"""B 方向：在 GPU 上用 LoRA 把 Qwen3.5-2B 训成「贵州茅台董秘」。

课程同款：Colab/免费卡 + 4bit QLoRA + Qwen3.5-2B（LoRA 页、unsloth 页、录屏页都是这套）。
本脚本两种后端：
    --backend unsloth （默认，课程讲义用的框架，最快）
    --backend peft    （unsloth 装不上时的退路：transformers + peft + bitsandbytes）

用法（实例里）：
    export HF_ENDPOINT=https://hf-mirror.com
    python train_lora_dongmi.py --data sft_train.jsonl --out lora_dongmi \
        --epochs 2 --rank 16 --batch 2 --grad-accum 4
    python train_lora_dongmi.py --max-steps 3        # 冒烟测试，只跑 3 步
产物：
    lora_dongmi/adapter_model.safetensors + 训练日志 + 训练前后对照样例
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_MODEL = "Qwen/Qwen3.5-2B"


def load_rows(path: str) -> tuple[list, list]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    return train, val


def build_training_args(args: argparse.Namespace, bf16: bool, optim: str):
    """新版 TRL 把 dataset_text_field / max_seq_length 挪进了 SFTConfig，
    这里按现场版本自适应，免得换个环境就报 unexpected keyword。"""
    common = dict(
        output_dir="train_logs", per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        warmup_steps=5, logging_steps=1, save_steps=100, save_total_limit=2,
        eval_strategy="no", optim=optim, weight_decay=0.01,
        lr_scheduler_type="linear", seed=3407, bf16=bf16, fp16=not bf16,
        report_to="none", max_steps=args.max_steps if args.max_steps else -1,
    )
    try:
        from trl import SFTConfig
    except ImportError:
        from transformers import TrainingArguments
        return TrainingArguments(**common), False

    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    if "dataset_text_field" in fields:
        common["dataset_text_field"] = "text"
    if "max_seq_length" in fields:
        common["max_seq_length"] = args.max_len
    elif "max_length" in fields:
        common["max_length"] = args.max_len
    if "packing" in fields:
        common["packing"] = False
    return SFTConfig(**common), True


def build_trainer(model, tok, train_ds, val_ds, targs, cfg_has_dataset_text: bool):
    from trl import SFTTrainer
    kwargs = dict(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds)
    if not cfg_has_dataset_text:                     # 老版 TRL：这些参数在 trainer 上
        kwargs.update(dataset_text_field="text", max_seq_length=max_len_holder["v"])
    try:
        return SFTTrainer(processing_class=tok, **kwargs)   # 新版 TRL
    except TypeError:
        return SFTTrainer(tokenizer=tok, **kwargs)          # 老版 TRL


max_len_holder = {"v": 2048}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_train.jsonl")
    ap.add_argument("--out", default="lora_dongmi")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--backend", default="unsloth", choices=["unsloth", "peft"])
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    ap.add_argument("--max-len", type=int, default=2048, dest="max_len")
    ap.add_argument("--max-steps", type=int, default=0, dest="max_steps",
                    help=">0 时只跑这么多步（冒烟测试用）")
    args = ap.parse_args()
    max_len_holder["v"] = args.max_len

    import torch
    from datasets import Dataset

    train_rows, val_rows = load_rows(args.data)
    print(f"训练 {len(train_rows)} 条 / 验证 {len(val_rows)} 条，基座 = {args.model}")

    if args.backend == "unsloth":
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_len,
            load_in_4bit=True, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=3407)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        # Qwen3.5-2B 是原生多模态（Qwen3_5ForConditionalGeneration），
        # 不能按纯文本的 AutoModelForCausalLM 加载；文字任务用多模态类一样训。
        cfg = AutoConfig.from_pretrained(args.model)
        arch = "".join(getattr(cfg, "architectures", None) or [])
        if "ConditionalGeneration" in arch:
            from transformers import AutoModelForImageTextToText as ModelCls
            print(f"检测到多模态架构 {arch}，用 AutoModelForImageTextToText 加载")
        else:
            from transformers import AutoModelForCausalLM as ModelCls
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
        except Exception:                       # 多模态仓库可能只有 processor
            from transformers import AutoProcessor
            tok = AutoProcessor.from_pretrained(args.model).tokenizer
        model = ModelCls.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto",
            dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(row):
        return {"text": row["text"]}

    train_ds = Dataset.from_list([to_text(r) for r in train_rows])
    val_ds = Dataset.from_list([to_text(r) for r in val_rows])
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数 {n_train:,} / {n_params:,} = {n_train/n_params*100:.3f}%")

    bf16 = torch.cuda.is_bf16_supported()
    optim = "adamw_8bit" if args.backend == "unsloth" else "paged_adamw_8bit"
    targs, cfg_has_text = build_training_args(args, bf16, optim)
    print("训练参数类型：", type(targs).__name__)
    trainer = build_trainer(model, tok, train_ds, val_ds, targs, cfg_has_text)
    t0 = time.time()
    trainer.train()
    print(f"训练完成，用时 {(time.time()-t0)/60:.1f} 分钟")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    log = {"base_model": args.model, "rank": args.rank, "epochs": args.epochs,
           "lr": args.lr, "batch": args.batch, "grad_accum": args.grad_accum,
           "train_rows": len(train_rows), "val_rows": len(val_rows),
           "trainable_params": n_train, "total_params": n_params,
           "minutes": round((time.time() - t0) / 60, 2),
           "final_loss": float(trainer.state.log_history[-1].get("loss", -1))}
    with open(os.path.join(args.out, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
