"""把 API Key 存进 Windows 凭据管理器（不落盘、不进代码、不进 git）。

用法（在 PowerShell 里，先 cd 到本目录）：
    ..\\venv\\Scripts\\python.exe save_key.py            # 默认存阿里百炼
    ..\\venv\\Scripts\\python.exe save_key.py dashscope  # 阿里百炼 DashScope
    ..\\venv\\Scripts\\python.exe save_key.py openai     # OpenAI（本作业不用）
    ..\\venv\\Scripts\\python.exe save_key.py deepseek   # DeepSeek（已有可跳过）
    ..\\venv\\Scripts\\python.exe save_key.py --show     # 只看是否已保存，不打印内容

输入时字符不会显示在屏幕上，粘贴后回车即可。
"""

from __future__ import annotations

import getpass
import sys

import keyring

SERVICES = {
    "dashscope": ("DashScopeAPI", "阿里百炼 DashScope", "sk-"),
    "openai": ("OpenAI", "OpenAI", "sk-"),
    "deepseek": ("DeepSeekAPI", "DeepSeek", "sk-"),
}


def show() -> None:
    for name, (service, label, _) in SERVICES.items():
        val = keyring.get_password(service, "default")
        print(f"{label:16s} {'已保存，长度 %d' % len(val) if val else '未保存'}")


def save(name: str) -> None:
    service, label, prefix = SERVICES[name]
    print(f"请输入 {label} 的 API Key（输入时不显示）：")
    key = getpass.getpass("> ").strip()
    if not key:
        print("没有输入内容，未保存。")
        return
    if prefix and not key.startswith(prefix):
        print(f"这个 key 不以 {prefix} 开头，请确认复制完整；已按原样保存，可重跑覆盖。")
    keyring.set_password(service, "default", key)
    print(f"已保存到 Windows 凭据管理器：{service} / default（长度 {len(key)}）")
    print("不要把 key 发到聊天窗口，也不要写进代码。")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--show"]
    if "--show" in sys.argv[1:]:
        show()
        return
    name = args[0].lower() if args else "dashscope"
    if name not in SERVICES:
        print("可选项：" + ", ".join(SERVICES))
        return
    save(name)


if __name__ == "__main__":
    main()
