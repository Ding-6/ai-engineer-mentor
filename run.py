# -*- coding: utf-8 -*-
"""
run.py — AI Engineer Mentor 启动入口
打包后由 PyInstaller 生成的 .exe 调用，避免直接打包 app.py 的 Flask 启动逻辑。
"""
import os
import sys
import threading
import time
import webbrowser

# 把 exe 同级目录加入 sys.path，便于加载模板 / 模块
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def open_browser():
    """延迟 1.2s 自动打开浏览器"""
    time.sleep(1.2)
    try:
        webbrowser.open("http://127.0.0.1:7860")
    except Exception:
        pass


def main():
    # 启动 Flask 在后台线程
    from app import app, init_db, AI_API_KEY
    init_db()

    # 自动打开浏览器（仅当 user 没在命令行加 --no-browser 时）
    no_browser = "--no-browser" in sys.argv
    if not no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    print("=" * 60)
    print("  AI Engineer Mentor  ·  本地版 v2")
    print("=" * 60)
    print("  浏览器已自动打开  http://127.0.0.1:7860")
    print("  关闭本窗口 = 停止服务")
    print("  默认账号: demo / demo123   管理员: admin / admin888")
    print("=" * 60)
    try:
        app.run(host="127.0.0.1", port=7860, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
