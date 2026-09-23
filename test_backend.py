"""后端接口测试 — 测试 /api/chat 和 /api/coser 接口"""
import requests

BACKEND_URL = "http://localhost:8000"

print("=" * 50)
print("测试 1：GET /api/coser")
print("=" * 50)
try:
    resp = requests.get(f"{BACKEND_URL}/api/coser", timeout=10)
    print(f"状态码：{resp.status_code}")
    data = resp.json()
    print(f"返回码：{data.get('code')}")
    roles = data.get("data", [])
    print(f"COSER 角色数量：{len(roles)}")
    if roles:
        print(f"前3个角色：")
        for r in roles[:3]:
            print(f"  - {r['name']}")
except Exception as e:
    print(f"❌ 测试失败：{e}")

print("\n" + "=" * 50)
print("测试 2：POST /api/chat")
print("=" * 50)
data = {
    "message": "你好，请做个自我介绍吧",
    "history": [],
    "system": "你是一个温和友善的大学辅导员，请以这个身份回复。",
}
try:
    resp = requests.post(f"{BACKEND_URL}/api/chat", json=data, timeout=60)
    print(f"状态码：{resp.status_code}")
    result = resp.json()
    reply = result.get("reply", "")
    print(f"模型回复：\n{reply[:300]}")
except Exception as e:
    print(f"❌ 测试失败：{e}")

print("\n" + "=" * 50)
print("测试 3：POST /api/chat（多轮对话）")
print("=" * 50)
data = {
    "message": "我刚才说我叫什么名字？",
    "history": [
        {"role": "user", "content": "你好，我叫小明"},
        {"role": "assistant", "content": "你好小明！很高兴认识你，有什么我可以帮你的吗？"},
    ],
    "system": "你是一个温和友善的大学辅导员，请以这个身份回复。",
}
try:
    resp = requests.post(f"{BACKEND_URL}/api/chat", json=data, timeout=60)
    print(f"状态码：{resp.status_code}")
    result = resp.json()
    reply = result.get("reply", "")
    print(f"模型回复：\n{reply[:300]}")
except Exception as e:
    print(f"❌ 测试失败：{e}")

print("\n" + "=" * 50)
print("测试 4：POST /api/export")
print("=" * 50)
data = {
    "history": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    ],
    "role_name": "测试角色",
    "format": "json",
}
try:
    resp = requests.post(f"{BACKEND_URL}/api/export", json=data, timeout=10)
    print(f"状态码：{resp.status_code}")
    result = resp.json()
    export_data = result.get("data", {})
    print(f"导出消息数：{export_data.get('total_messages')}")
    print(f"角色名：{export_data.get('role_name')}")
except Exception as e:
    print(f"❌ 测试失败：{e}")

print("\n✅ 全部接口测试完成")
