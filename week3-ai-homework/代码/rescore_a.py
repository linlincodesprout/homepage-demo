"""只重算 A 的判分（不重新调大模型），用于改判分规则后快速刷新判分表。"""

import json

import config
from eval_a import score, write_summary

if __name__ == "__main__":
    reports = json.loads((config.OUT_DIR / "A_评测结果.json").read_text("utf-8"))
    score(reports)
    write_summary(reports)
    for r in reports:
        s = r["score"]
        print(f"{r['id']} {r['type']}: {s['hit_numbers']}/{s['ref_numbers']} "
              f"{'✅' if s['correct'] else '❌'} 缺 {s['missing'][:3]} "
              f"{'（写了未覆盖）' if s['said_not_covered'] else ''}")
    print("判分表 →", config.OUT_DIR / "A_判分表.md")
