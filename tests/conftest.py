"""pytest 公共配置
src/ 是平铺模块(import retrieve / import api),不是包,
所以要把 src 加进 sys.path,测试里才能直接 import。
另外把工作目录固定到项目根,保证 load_dotenv() 能找到 .env
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
