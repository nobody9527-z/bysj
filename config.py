"""
统一配置文件 —— 所有模块从这里读取 API 参数，改一处全局生效
"""
import os

# 从 .env 文件加载环境变量
def _load_env():
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())
_load_env()

# DeepSeek API（密钥在 .env 文件中，不提交到代码仓库）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# 评估参数
MBTI_EVAL_TEMPERATURE = 0.35      # 角色生成温度（越低越稳定，角色一致性更高）
JUDGE_TEMPERATURE = 0.1            # 评判温度（0.1=几乎确定，但留一点灵活性避免空输出）
JUDGE_MAX_TOKENS = 200
GENERATE_MAX_TOKENS = 350          # 足够写3-5句话不被截断
API_TIMEOUT = 120                  # 秒
API_MAX_RETRIES = 3                # 重试次数
RETRY_DELAY = 2                    # 重试间隔秒

# COSER 数据集
COSER_FOLDER = "./coser"
MAX_EVAL_SAMPLES = 100
MBTI_TEST_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

# 数据持久化
DATA_DIR = "./data"
