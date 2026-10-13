"""B 方向：四格对照评测（GPU 端只做生成，检索上下文由本机预先算好）。

四格 = 训练过 / 没训练过 × 原问法 / 换问法；
再加两格：A 的知识库 + 同一个小模型（未微调 / 已微调）。

用法（实例里）：
    python eval_dongmi.py --eval eval_with_context.jsonl --adapter lora_dongmi \
        --out eval_results.jsonl --batch 6 --max-new 200
产物：
    eval_results.jsonl  每行 {config, pool, variant, question, output, reference}
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SYS_SECRETARY = ("你是贵州茅台酒股份有限公司的董事会秘书。回答投资者提问时只依据公司已披露的定期报告，"
                 "数字原样引用并说明口径，回答末尾标注（来源：报告全名 第 X 页）；"
                 "报告里没有的内容，直接说明未披露，不要推测。")
SYS_PLAIN = "你是贵州茅台酒股份有限公司的董事会秘书，请回答投资者提问。"
SYS_KB = ("你是贵州茅台酒股份有限公司的董事会秘书。只能依据【材料】回答，不得使用材料之外的知识。\n"
          "材料里经常同时出现**合并报表**和**母公司报表**，或同一张表的多个年份/多个栏目：\n"
          "先判断问题问的是哪个口径、哪个期间，只采用对应那一份的数据；"
          "问题没特别说明时优先用合并口径，并在回答里写明口径（如“合并口径”）。\n"
          "数字要和材料一模一样，不要换算。回答末尾标注（来源：报告全名 第 X 页）。"
          "材料里查不到就写“材料未覆盖”，不要推测。")


def load_model(model_name: str, adapter: str | None):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_name)
    arch = "".join(getattr(cfg, "architectures", None) or [])
    cls = AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM
    model = cls.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    except Exception:
        tok = None
    if tok is None or not getattr(tok, "chat_template", None):
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_name)
        tok = proc.tokenizer
        tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload() if adapter.endswith("__merged") else model
        model.eval()
    return model, tok


def render(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def build_prompts(tok, rows: list[dict], with_context: bool,
                  prompt: str = "kb") -> list[str]:
    """四格对照里只有「给不给材料」这一个变量，system 提示词必须完全一致。

    第一版给无材料的格子换了短 system，结果微调后的模型答得完全不像训练过的样子
    （编出 1,000,000,000 元这种数字），那是评测变量没控住，不是模型的问题。
    """
    prompts = []
    for r in rows:
        if with_context:
            user = f"【材料】\n{r['context']}\n\n【问题】\n{r['question']}"
            sys_prompt = SYS_KB if prompt == "kb_caliber" else SYS_SECRETARY
            prompts.append(render(tok, sys_prompt, user))
        else:
            prompts.append(render(tok, SYS_SECRETARY, r["question"]))
    return prompts


def generate_all(model, tok, prompts: list[str], batch: int, max_new: int) -> list[str]:
    import torch
    outs: list[str] = []
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
        done = min(i + batch, len(prompts))
        print(f"  {done}/{len(prompts)}  {(time.time()-t0)/max(done,1):.1f}s/条", flush=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_with_context.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default="lora_dongmi")
    ap.add_argument("--out", default="eval_results.jsonl")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=200, dest="max_new")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default="untrained,finetuned,kb_untrained,kb_finetuned",
                    help="只跑其中几格，逗号分隔")
    ap.add_argument("--prompt", default="kb", choices=["kb", "kb_caliber"],
                    help="知识库那两格用哪版系统提示词")
    ap.add_argument("--where", default="", help="只跑满足条件的题，如 pool=seen,variant=original")
    args = ap.parse_args()
    want = {c.strip() for c in args.configs.split(",") if c.strip()}

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    if args.where:
        conds = dict(p.split("=") for p in args.where.split(","))
        rows = [r for r in rows if all(str(r.get(k)) == v for k, v in conds.items())]
    if args.limit:
        rows = rows[:args.limit]
    print(f"评测 {len(rows)} 条")

    results = []

    print("\n== 载入基座（未微调）==")
    todo_base = [("untrained", False), ("kb_untrained", True)]
    if any(n in want for n, _ in todo_base):
        model, tok = load_model(args.model, None)
    for cfg_name, with_ctx in todo_base:
        if cfg_name not in want:
            continue
        prompts = build_prompts(tok, rows, with_ctx, args.prompt)
        outs = generate_all(model, tok, prompts, args.batch, args.max_new)
        for r, o in zip(rows, outs):
            results.append({"config": cfg_name, "pool": r["pool"], "variant": r["variant"],
                            "question": r["question"], "reference": r["reference"],
                            "context_ids": r["hits"], "output": o})
        with open(args.out, "w", encoding="utf-8") as f:
            for x in results:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{cfg_name} 完成")
    if any(n in want for n, _ in todo_base):
        del model

    print("\n== 载入基座 + LoRA（微调后）==")
    todo_ft = [("finetuned", False), ("kb_finetuned", True)]
    if any(n in want for n, _ in todo_ft):
        model, tok = load_model(args.model, args.adapter)
    for cfg_name, with_ctx in todo_ft:
        if cfg_name not in want:
            continue
        prompts = build_prompts(tok, rows, with_ctx, args.prompt)
        outs = generate_all(model, tok, prompts, args.batch, args.max_new)
        for r, o in zip(rows, outs):
            results.append({"config": cfg_name, "pool": r["pool"], "variant": r["variant"],
                            "question": r["question"], "reference": r["reference"],
                            "context_ids": r["hits"], "output": o})
        with open(args.out, "w", encoding="utf-8") as f:
            for x in results:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{cfg_name} 完成")

    print(f"\n全部完成 {len(results)} 条 → {args.out}")


if __name__ == "__main__":
    main()
