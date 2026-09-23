from openai import OpenAI
import traceback

# ===================== 配置 =====================
import os
API_KEY = os.environ.get("ARK_API_KEY", "")
# ================================================

MODEL_NAME = "doubao-seed-2-0-pro-260215"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

print("🔍 开始初始化API客户端...")
try:
    # 初始化客户端，增加30秒超时避免网络慢
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=30
    )
    print("✅ 客户端初始化成功")

    print("🔍 发起API调用（请求模型回复「你好」）...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "你好"}]
    )
    print("✅ API调用成功！模型回复：")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ 调用出错，详细错误信息：")
    print(f"错误类型：{type(e).__name__}")
    print(f"错误信息：{str(e)}")
    print("\n完整错误堆栈：")
    traceback.print_exc()