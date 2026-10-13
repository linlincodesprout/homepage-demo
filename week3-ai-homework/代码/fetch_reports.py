"""从巨潮资讯网（cninfo.com.cn，证监会指定的信息披露平台）下载半年报全文。

上交所/深交所自己的站点对爬虫限制较多（静态文件域名经常超时），
巨潮是交易所指定的法定披露平台，两个市场的公告都能拿到，接口也更稳定。

用法：
    python fetch_reports.py            # 下载 config.COMPANIES 里全部公司
    python fetch_reports.py 600519     # 只下载指定代码
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

import config

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}
TOP_SEARCH = "http://www.cninfo.com.cn/new/information/topSearch/query"
QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC = "http://static.cninfo.com.cn/"


def find_org_id(session: requests.Session, code: str) -> str | None:
    """把股票代码换成巨潮内部的 orgId（公告查询接口必填）。"""
    r = session.post(TOP_SEARCH, headers=HEADERS,
                     data={"keyWord": code, "maxNum": "10"}, timeout=30)
    r.raise_for_status()
    for item in r.json():
        if item.get("code") == code:
            return item.get("orgId")
    return None


def list_announcements(session: requests.Session, code: str, org_id: str,
                       market: str, category: str,
                       se_date: str | None = None) -> list[dict]:
    """列出某家公司某一类别的公告（category_bndbg_szsh = 半年报）。"""
    data = {
        "pageNum": "1",
        "pageSize": "30",
        "column": market,
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{code},{org_id}",
        "searchkey": "",
        "secid": "",
        "category": category,
        "trade": "",
        "seDate": se_date or config.REPORT_SE_DATE,
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    r = session.post(QUERY, headers=HEADERS, data=data, timeout=30)
    r.raise_for_status()
    return r.json().get("announcements") or []


def pick_full_report(items: list[dict], keyword: str | None = None) -> dict | None:
    """在一堆公告里挑出「半年度报告」全文（排除摘要/英文版/更正/已取消）。"""
    bad = ("摘要", "英文", "取消", "补充", "问询", "说明会", "意见", "提示")
    keyword = keyword or config.REPORT_KEYWORD
    cands = []
    for it in items:
        title = (it.get("announcementTitle") or "").replace("<em>", "").replace("</em>", "")
        if keyword not in title:
            continue
        if any(b in title for b in bad):
            continue
        cands.append((it, title))
    if not cands:
        return None
    # 标题最长的通常是最全的那份（"2026年半年度报告"）
    return max(cands, key=lambda x: len(x[1]))[0]


def download(session: requests.Session, url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 200_000:
        return dest.stat().st_size
    r = session.get(url, headers={"User-Agent": UA}, timeout=180, stream=True)
    r.raise_for_status()
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size


def main() -> None:
    only = set(sys.argv[1:])
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = config.PDF_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    session = requests.Session()
    for code, name, market in config.COMPANIES:
        if only and code not in only:
            continue
        try:
            org_id = find_org_id(session, code)
            if not org_id:
                print(f"[跳过] {code} {name}：找不到 orgId")
                continue
            items = list_announcements(session, code, org_id, market, "category_bndbg_szsh")
            report = pick_full_report(items)
            if not report:
                print(f"[跳过] {code} {name}：本期没有半年报全文（共 {len(items)} 条公告）")
                continue
            title = report["announcementTitle"].replace("<em>", "").replace("</em>", "")
            url = STATIC + report["adjunctUrl"]
            dest = config.PDF_DIR / f"{code}_{name}_{config.REPORT_PERIOD}.pdf"
            size = download(session, url, dest)
            manifest[code] = {
                "code": code, "name": name, "market": market, "title": title,
                "url": url, "pdf": str(dest.relative_to(config.ROOT)),
                "size": size, "published": report.get("announcementTime"),
            }
            print(f"[完成] {code} {name} {title} {size/1024:.0f} KB")
            time.sleep(0.6)
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {code} {name}: {exc}")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n共 {len(manifest)} 份报告，清单写入 {manifest_path}")


if __name__ == "__main__":
    main()
