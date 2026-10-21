#!/bin/bash
# ============================================================
# AI Engineer Mentor — 一键安装脚本
# ============================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        AI Engineer Mentor — 环境安装                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# 创建 conda 环境
if conda info --envs | grep -q "^myai "; then
    echo "✅ conda env 'myai' 已存在"
else
    echo "📦 创建 conda env 'myai' (Python 3.11)..."
    conda create -n myai python=3.11 -y
fi

echo "📥 安装依赖..."
conda run -n myai pip install \
  flask>=3.0 \
  chromadb>=0.5.0 \
  PyPDF2>=3.0 \
  pdfplumber>=0.10 \
  python-docx>=1.1 \
  openpyxl>=3.1 \
  xlrd>=2.0 \
  beautifulsoup4>=4.12 \
  requests>=2.31 \
  pillow>=10.0

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ 安装完成！启动步骤：                             ║"
echo "║                                                      ║"
echo "║  1. 编辑 .env 填入 API Key：                         ║"
echo "║     AI_API_KEY=your_key_here                         ║"
echo "║     TAVILY_API_KEY=tvly-xxxxx  （可选，联网搜索）     ║"
echo "║                                                      ║"
echo "║  2. 启动服务：                                       ║"
echo "║     conda activate myai                              ║"
echo "║     python app.py                                    ║"
echo "║                                                      ║"
echo "║  3. 浏览器访问：http://127.0.0.1:7860                ║"
echo "║     演示账号：demo / demo123                         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
