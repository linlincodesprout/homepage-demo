"""B 方向语料：贵州茅台 2021–2025 年年报 + 2021–2026 年半年报，共 11 份全文。

和 A 方向用同一个数据源（巨潮资讯网）、同一套下载逻辑，只是换了公告类别和时间窗。

用法：
    python fetch_reports_dongmi.py
输出：
    data/dongmi/pdf/*.pdf + manifest.json
"""

from __future__ import annotations

import json
import time

import requests

import config
from fetch_reports import (HEADERS, STATIC, UA, download, find_org_id,
                           list_announcements, pick_full_report)

CODE = "600519"
NAME = "贵州茅台"
MARKET = "sse"
OUT_DIR = config.DATA_DIR / "dongmi" / "pdf"

# (类别, 时间窗, 关键词, 期间标签)
WINDOWS = [
    ("category_ndbg_szsh", f"{y}-01-01~{y + 1}-12-31", "年度报告", f"{y}A")
    for y in range(2021, 2026)
] + [
    ("category_bndbg_szsh", f"{y}-06-01~{y + 1}-01-31", "半年度报告", f"{y}H1")
    for y in range(2021, 2027)
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    org_id = find_org_id(session, CODE)
    if not org_id:
        raise SystemExit("找不到 600519 的 orgId")
    print(f"orgId = {org_id}")

    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    for category, se_date, keyword, period in WINDOWS:
        items = list_announcements(session, CODE, org_id, MARKET, category, se_date)
        report = pick_full_report(items, keyword)
        if not report:
            print(f"[跳过] {period} {keyword}：窗口 {se_date} 内没找到（{len(items)} 条公告）")
            continue
        title = report["announcementTitle"].replace("<em>", "").replace("</em>", "")
        url = STATIC + report["adjunctUrl"]
        dest = OUT_DIR / f"{CODE}_{NAME}_{period}.pdf"
        size = download(session, url, dest)
        manifest[period] = {
            "code": CODE, "name": NAME, "period": period, "title": title,
            "url": url, "pdf": str(dest.relative_to(config.ROOT)),
            "size": size, "published": report.get("announcementTime"),
        }
        print(f"[完成] {period} {title} {size/1024:.0f} KB")
        time.sleep(0.6)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n共 {len(manifest)} 份 → {manifest_path}")


if __name__ == "__main__":
    main()
