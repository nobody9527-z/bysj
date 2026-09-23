# 基于大语言模型的人物模拟与对话系统

基于 FastAPI + Vue 3 的角色扮演对话系统。可以自定义角色（姓名、MBTI、性格、职业、经历等），让大模型以该角色身份进行对话，并支持对话记忆、历史记录和对话质量评估。

## 功能

- 自定义角色配置（MBTI、性格、职业、简介、经历）
- 基于 DeepSeek 大模型生成符合角色设定的回复
- 内置 COSER 小说角色数据集，可直接选择角色对话
- 对话记忆：自动提取用户信息，下次对话时自动带出
- 对话记录：保存、查看、继续历史对话，支持导出 JSON / TXT
- 评估：MBTI 性格模拟准确率、BLEU / ROUGE-L 对话质量、消融实验、错误分析

## 技术栈

- 后端：Python + FastAPI
- 前端：Vue 3 + Vite
- 大模型：DeepSeek API
- 评估：jieba 分词、NLTK BLEU、ROUGE-L

## 运行

### 后端

```bash
pip install -r requirements.txt
```

在项目根目录新建 `.env` 文件，填入：

```
DEEPSEEK_API_KEY=你的key
```

启动：

```bash
python backend.py
```

后端默认运行在 `http://127.0.0.1:8000`。

### 前端

```bash
cd vue-chat
npm install
npm run dev
```

打开 Vite 提示的地址（默认 `http://localhost:5173`）即可。

### 评估脚本

```bash
python eval_mbti.py              # MBTI 准确率评估
python eval_mbti.py --ablation   # 消融实验
python eval_bleu.py              # BLEU / ROUGE-L 评估
```

> 注：COSER 角色和 BLEU 评估依赖 `coser/` 数据集（小说角色 JSON），该目录未包含在仓库中。

## 项目结构

```
.
├── backend.py       # FastAPI 后端主程序
├── config.py        # 配置中心（API Key、模型、参数）
├── eval_mbti.py     # MBTI 评估 + 消融实验
├── eval_bleu.py     # BLEU / ROUGE-L 评估
├── test_api.py      # 接口测试
├── data/            # 角色、对话、记忆持久化
└── vue-chat/        # Vue 3 前端
```
