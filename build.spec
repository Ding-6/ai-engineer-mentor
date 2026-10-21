# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 配置 — 双击 build.bat 即可一键打包
# 打包结果: dist/AIEngineerMentor/AIEngineerMentor.exe
# 模板文件夹需要作为数据一起打包

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas = [
    ("templates", "templates"),
]
# 收集 chromadb / onnxruntime 等可能的数据文件
datas += collect_data_files("chromadb", includes=["*.py", "*.json", "*.bin", "*.so", "*.pyd"])
datas += collect_data_files("onnxruntime", includes=["*.py", "*.json", "*.bin"])
datas += collect_data_files("tokenizers", includes=["*.py", "*.json"])
datas += collect_data_files("pdfminer", includes=["*.json"])
datas += collect_data_files("uvicorn", includes=["*.py"])
datas += collect_data_files("httpx", includes=["*.py"])

hiddenimports = [
    "app", "rag", "intent", "search", "file_parser",
    "flask", "jinja2", "werkzeug", "click",
    "sqlite3", "json", "urllib.request",
    "chromadb", "onnxruntime", "tokenizers",
    "PyPDF2", "pdfplumber", "pdfminer",
    "docx", "openpyxl", "xlrd",
    "bs4", "requests", "PIL",
    "cryptography", "bcrypt", "huggingface_hub",
]

a = Analysis(
    ["run.py"],
    pathex=[os.getcwd()],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "numpy.tests", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIEngineerMentor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,        # 保留控制台窗口，便于看日志
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AIEngineerMentor",
)
