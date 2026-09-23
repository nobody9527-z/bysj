<template>
  <div class="app-container">
    <!-- 顶部标题栏 -->
    <header class="top-header">
      <div class="header-content">
        <h1>基于大语言模型的人物模拟与对话系统</h1>
        <p>沈阳航空航天大学 · 人工智能学院 · 毕业设计</p>
      </div>
      <!-- 标签切换 -->
      <nav class="tab-nav">
        <button :class="['tab-btn', { active: activeTab === 'chat' }]" @click="activeTab = 'chat'">💬 对话</button>
        <button :class="['tab-btn', { active: activeTab === 'eval' }]" @click="activeTab = 'eval'">📊 评估</button>
        <button :class="['tab-btn', { active: activeTab === 'history' }]" @click="activeTab = 'history'; loadSavedCharacters(); loadConversations()">📋 记录</button>
      </nav>
    </header>

    <!-- ==================== 对话页 ==================== -->
    <main v-if="activeTab === 'chat'" class="chat-main">
      <div class="chat-header">
        <div class="coser-select-wrapper">
          <label>选择对话角色：</label>
          <select v-model="selectedCoser" @change="selectCoserRole" class="coser-select">
            <option value="">请选择角色</option>
            <option v-for="item in coserList" :key="item.id" :value="item">{{ item.name }}</option>
          </select>
        </div>
        <button class="config-btn" @click="dialogVisible = true">⚙️ 更改人物配置</button>
        <button class="save-btn" @click="saveCurrentCharacter">💾 保存角色</button>
        <button class="memory-btn" @click="extractMemories" :disabled="messages.length < 3">🧠 保存记忆</button>
        <button class="save-conv-btn" @click="saveCurrentConversation" :disabled="messages.length === 0" v-if="currentRoleName">💾 保存对话</button>
        <button class="memory-btn" @click="showMemories = !showMemories" v-if="memoryCount > 0" style="background:#10b981">
          🧠 记忆({{ memoryCount }})
        </button>
        <button class="export-btn" @click="exportChat('json')" :disabled="messages.length === 0">📥 导出JSON</button>
        <button class="export-btn" @click="exportChat('txt')" :disabled="messages.length === 0">📄 导出TXT</button>
      </div>

      <!-- 默认角色快速选择 -->
      <div v-if="!currentRoleName" class="default-char-section">
        <h3>选择一个角色开始对话，或 <a href="#" @click.prevent="dialogVisible = true">自定义创建</a></h3>
        <div class="default-char-grid">
          <div v-for="char in defaultCharacters" :key="char.name"
               :class="['default-char-card', { selected: previewChar === char }]"
               @click="selectDefaultChar(char)">
            <div class="char-card-name">{{ char.name }}</div>
            <div class="char-card-mbti">{{ char.mbti }}</div>
            <div class="char-card-career">{{ char.career }}</div>
            <div class="char-card-intro">{{ char.intro?.slice(0, 40) }}{{ char.intro?.length > 40 ? '...' : '' }}</div>
          </div>
        </div>
      </div>

      <!-- 当前角色信息条 -->
      <div v-if="currentRoleName" class="current-role-bar">
        <span>🤖 当前角色：<strong>{{ currentRoleName }}</strong>（{{ character.mbti }} · {{ character.career }}）</span>
        <button @click="resetCharacter">🔄 更换角色</button>
      </div>

      <!-- 记忆面板 -->
      <div v-if="showMemories" class="memory-panel">
        <div class="memory-panel-head">
          <span>🧠 {{ currentRoleName }} 的记忆</span>
          <button @click="showMemories = false" class="close-icon-sm">×</button>
        </div>
        <div v-if="memories.length === 0" class="memory-empty">暂无记忆，对话3轮后可保存</div>
        <div v-for="(m, i) in memories" :key="i" class="memory-item">
          <span>{{ m }}</span>
          <button @click="deleteMemory(i)" class="memory-del">×</button>
        </div>
      </div>

      <div class="chat-messages no-scrollbar" ref="chatBoxRef">
        <div v-if="messages.length === 0" class="empty-state">
          <div class="empty-icon">💬</div>
          <div class="empty-title">欢迎使用人物模拟对话系统</div>
          <div class="empty-desc">选择COSER角色或自定义配置，即可开始对话~</div>
        </div>
        <div v-for="(msg, index) in messages" :key="index" :class="['message-item', msg.role]">
          <div class="avatar">{{ msg.role === "user" ? "👤" : "🤖" }}</div>
          <div class="message-box">
            <span v-if="msg.role === 'assistant'" class="role-tag">{{ currentRoleName }}</span>
            <div class="message-content">{{ msg.content }}</div>
          </div>
        </div>
        <div v-if="isLoading" class="loading"><span>AI正在生成回复中...</span></div>
      </div>

      <div class="input-footer">
        <textarea v-model="userInput" @keydown.enter.exact="sendMessage"
          :placeholder="`和 ${currentRoleName} 聊点什么吧~`"
          class="message-input no-scrollbar" ref="inputRef"></textarea>
        <button class="send-btn" @click="sendMessage" :disabled="isLoading || !userInput.trim()">发送</button>
      </div>
    </main>

    <!-- ==================== 评估页 ==================== -->
    <main v-if="activeTab === 'eval'" class="eval-main">
      <!-- ========== MBTI 评估 ========== -->
      <div class="eval-section">
        <h2>📊 MBTI 性格模拟准确率评估</h2>
        <p class="eval-desc">任务书技术指标：评价大模型对MBTI性格模拟的效果，采用准确率作为指标</p>
        <div class="eval-actions">
          <button class="eval-run-btn" @click="toggleMbti" :disabled="mbtiRunning">
            {{ mbtiRunning ? '⏳ 评估中...' : mbtiEvalResult && showMbtiDetail ? '▼ 收起MBTI评估结果' : mbtiEvalResult && !showMbtiDetail ? '▶ 展开MBTI评估结果' : '▶ 加载MBTI评估结果' }}
          </button>
          <button class="eval-run-btn" @click="runMbtiEvalStream" :disabled="mbtiRunning" style="background:#10b981">
            {{ mbtiRunning ? '评估中...' : '🔄 在线评估（随机4类型）' }}
          </button>
        </div>
        <div v-if="mbtiRunning" class="eval-progress">⏳ 流式评估中，结果逐类型实时更新...</div>
        <div v-if="mbtiEvalResult && showMbtiDetail" class="eval-result-card">
          <div class="eval-summary">
            <div class="eval-metric">
              <span class="metric-value">{{ mbtiEvalResult.overall_accuracy_pct }}</span>
              <span class="metric-label">MBTI准确率</span>
            </div>
            <div class="eval-metric">
              <span class="metric-value">{{ mbtiEvalResult.total_samples }}</span>
              <span class="metric-label">测试样本数</span>
            </div>
          </div>
          <div class="type-bars" v-if="mbtiTypeAccuracy">
            <div v-for="(stats, mbti) in mbtiTypeAccuracy" :key="mbti" class="type-bar-row">
              <span class="type-label">{{ mbti }}</span>
              <div class="type-bar-track">
                <div class="type-bar-fill" :style="{ width: (stats.accuracy * 100) + '%' }"></div>
              </div>
              <span class="type-score">{{ (stats.accuracy * 100).toFixed(0) }}% ({{ stats.correct }}/{{ stats.total }})</span>
            </div>
          </div>
        </div>
      </div>

      <!-- ========== BLEU 评估 ========== -->
      <div class="eval-section">
        <h2>📈 BLEU 对话质量评估</h2>
        <p class="eval-desc">任务书要求：采用BLEU等作为评价指标，基于COSER数据集评估对话生成质量</p>
        <div class="eval-actions">
          <button class="eval-run-btn" @click="toggleBleu" :disabled="bleuRunning || mbtiRunning">
            {{ bleuRunning ? '⏳ 评估中...' : bleuEvalResult && showBleuDetail ? '▼ 收起BLEU评估结果' : bleuEvalResult && !showBleuDetail ? '▶ 展开BLEU评估结果' : '▶ 加载BLEU评估结果' }}
          </button>
          <button class="eval-run-btn" @click="runBleuEvalStream" :disabled="bleuRunning || mbtiRunning" style="background:#10b981">
            {{ bleuRunning ? '评估中...' : '🔄 在线重新评估' }}
          </button>
          <select v-model="bleuMbtiType" class="bleu-mbti-select">
            <option value="">通用评估（无MBTI约束）</option>
            <option v-for="t in mbtiTypes" :key="t.code" :value="t.code">{{ t.code }} - {{ t.description }}</option>
          </select>
        </div>
        <div v-if="bleuRunning" class="eval-progress">⏳ 流式BLEU评估中，进度实时更新...</div>
        <div v-if="bleuEvalResult && showBleuDetail" class="eval-result-card">
          <div v-if="bleuEvalResult.mbti_type" style="font-size:13px;color:#6366f1;margin-bottom:8px">🎯 评估约束：{{ bleuEvalResult.mbti_type }} 型人格</div>
          <div class="eval-summary">
            <div class="eval-metric">
              <span class="metric-value">{{ bleuEvalResult.avg_bleu ?? bleuEvalResult.corpus_bleu_word ?? '-' }}</span>
              <span class="metric-label">词级BLEU</span>
            </div>
            <div class="eval-metric">
              <span class="metric-value">{{ bleuEvalResult.corpus_bleu_char ?? '-' }}</span>
              <span class="metric-label">字符级BLEU</span>
            </div>
            <div class="eval-metric">
              <span class="metric-value">{{ bleuEvalResult.rouge_l_f1 ?? '-' }}</span>
              <span class="metric-label">ROUGE-L</span>
            </div>
            <div class="eval-metric">
              <span class="metric-value">{{ bleuEvalResult.total_samples }}</span>
              <span class="metric-label">样本数</span>
            </div>
          </div>
        </div>
      </div>

      <!-- ========== 消融实验 ========== -->
      <div class="eval-section">
        <h2>🔬 消融实验</h2>
        <p class="eval-desc">对比验证MBTI风格指令的有效性：A组（有风格指令）vs B组（仅MBTI标签）</p>
        <div class="eval-actions">
          <button class="eval-run-btn" @click="toggleAblation" :disabled="ablationRunning || mbtiRunning || bleuRunning">
            {{ ablationRunning ? '⏳ 评估中...' : ablationResult && showAblationDetail ? '▼ 收起消融实验结果' : ablationResult && !showAblationDetail ? '▶ 展开消融实验结果' : '▶ 加载消融实验结果' }}
          </button>
          <button class="eval-run-btn" @click="runAblationStream" :disabled="ablationRunning || mbtiRunning || bleuRunning" style="background:#10b981">
            {{ ablationRunning ? '评估中...' : '🔄 在线评估（随机4类型）' }}
          </button>
        </div>
        <div v-if="ablationRunning" class="eval-progress">⏳ 流式消融评估中，逐类型对比...</div>
        <div v-if="ablationResult && showAblationDetail" class="eval-result-card" style="margin-top:12px">
          <div class="eval-summary">
            <div class="eval-metric">
              <span class="metric-value">{{ (ablationResult.with_style_accuracy * 100).toFixed(1) }}%</span>
              <span class="metric-label">有风格指令</span>
            </div>
            <div class="eval-metric">
              <span class="metric-value">{{ (ablationResult.without_style_accuracy * 100).toFixed(1) }}%</span>
              <span class="metric-label">无风格指令</span>
            </div>
            <div class="eval-metric">
              <span class="metric-value" style="color:#16a34a">+{{ (ablationResult.improvement * 100).toFixed(1) }}%</span>
              <span class="metric-label">提升幅度</span>
            </div>
          </div>
          <div v-if="ablationTypeComparison" class="type-bars">
            <div v-for="(stats, mbti) in ablationTypeComparison" :key="mbti" class="type-bar-row">
              <span class="type-label">{{ mbti }}</span>
              <span style="font-size:12px;color:#667eea;width:40px;text-align:right">{{ (stats.with_style * 100).toFixed(0) }}%</span>
              <span style="font-size:12px;color:#9ca3af;width:40px;text-align:right">{{ (stats.without_style * 100).toFixed(0) }}%</span>
              <span class="type-score" style="color:#16a34a">+{{ (stats.improvement * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- ========== 错误分析 ========== -->
      <div class="eval-section">
        <h2>🔍 错误分析</h2>
        <p class="eval-desc">MBTI评估中被判为不一致的样本分布（数据来自MBTI评估结果）</p>
        <div class="eval-actions">
          <button class="eval-run-btn" @click="toggleError" style="background:#f59e0b">
            {{ errorAnalysis && showErrorDetail ? '▼ 收起错误分析' : errorAnalysis && !showErrorDetail ? '▶ 展开错误分析' : '▶ 加载错误分析' }}
          </button>
          <span class="eval-hint">运行MBTI评估时实时更新 | 也可手动加载已保存文件</span>
        </div>
        <div v-if="errorAnalysis && showErrorDetail" class="eval-result-card" style="margin-top:12px">
          <p>错误率：<strong>{{ (errorAnalysis.error_rate * 100).toFixed(1) }}%</strong> ({{ errorAnalysis.error_count }}/{{ errorAnalysis.total_samples }})</p>
          <p>按类型：<span v-for="(c, t) in errorAnalysis.by_type" :key="t" class="mbti-badge" style="margin:2px">{{ t }}: {{ c }}</span></p>
          <p>按维度：<span v-for="(c, d) in errorAnalysis.by_dimension" :key="d" class="mbti-badge" style="margin:2px;background:#f59e0b">{{ d }}: {{ c }}</span></p>
        </div>
      </div>
    </main>

    <!-- ==================== 记录页 ==================== -->
    <main v-if="activeTab === 'history'" class="history-main">
      <div class="history-section">
        <h2>💾 已保存角色配置</h2>
        <button class="eval-run-btn" @click="loadSavedCharacters">🔄 刷新</button>
        <div v-if="savedCharacters.length === 0" class="empty-state small">
          <div class="empty-desc">暂无保存的角色配置</div>
        </div>
        <div v-for="char in savedCharacters" :key="char.id" class="history-card">
          <div class="history-card-header">
            <strong>{{ char.name }}</strong>
            <span class="mbti-badge">{{ char.mbti }}</span>
            <span class="time-text">{{ char.created_at?.slice(0, 10) }}</span>
          </div>
          <div class="history-card-body">
            <span>性格：{{ char.personality }}</span>
            <span>职业：{{ char.career }}</span>
          </div>
          <div class="history-card-actions">
            <button @click="loadCharacter(char)">📥 加载</button>
            <button @click="deleteCharacter(char.id)" class="danger">🗑 删除</button>
          </div>
        </div>
      </div>

      <div class="history-section">
        <h2>📋 历史对话记录</h2>
        <button class="eval-run-btn" @click="loadConversations">🔄 刷新</button>
        <div v-if="savedConversations.length === 0" class="empty-state small">
          <div class="empty-desc">暂无保存的对话记录（在对话页中会自动保存）</div>
        </div>
        <div v-for="conv in savedConversations" :key="conv.id" class="history-card">
          <div class="history-card-header">
            <strong>{{ conv.title }}</strong>
            <span class="mbti-badge">{{ conv.character_name }}</span>
            <span class="time-text">{{ conv.message_count }}条消息 | {{ conv.created_at?.slice(0, 10) }}</span>
          </div>
          <div class="history-card-actions">
            <button @click="continueConversation(conv.id)">💬 继续对话</button>
            <button @click="viewConversation(conv.id)">👁 查看</button>
            <button @click="deleteConversation(conv.id)" class="danger">🗑 删除</button>
          </div>
        </div>
      </div>

      <!-- 查看对话详情弹窗 -->
      <Teleport to="body">
      <div v-if="viewingConversation" class="view-dialog-mask" @click.self="viewingConversation = null">
        <div class="view-dialog-box">
          <!-- 头部：标题 + 角色信息 + 关闭 -->
          <div class="view-dialog-head">
            <div class="view-dialog-title-row">
              <h3>{{ viewingConversation.title }}</h3>
              <button class="view-close-btn" @click="viewingConversation = null">×</button>
            </div>
            <div v-if="viewingConversation.character?.name" class="view-dialog-meta">
              <span class="view-meta-tag">{{ viewingConversation.character.mbti }}</span>
              <span>{{ viewingConversation.character.name }}</span>
              <span v-if="viewingConversation.character.career">· {{ viewingConversation.character.career }}</span>
              <span class="view-meta-count">共 {{ viewingConversation.history?.length || 0 }} 条消息</span>
            </div>
          </div>
          <!-- 消息列表（可滚动） -->
          <div class="view-dialog-body">
            <div v-for="(msg, i) in viewingConversation.history" :key="i" :class="['view-msg', msg.role]">
              <span class="view-msg-role">{{ msg.role === 'user' ? '👤 用户' : '🤖 ' + (viewingConversation.character?.name || '角色') }}</span>
              <div class="view-msg-text">{{ msg.content }}</div>
            </div>
          </div>
          <!-- 底部 -->
          <div class="view-dialog-foot">
            <button class="view-foot-btn primary" @click="continueConversation(viewingConversation.id); viewingConversation = null">💬 继续这段对话</button>
            <button class="view-foot-btn" @click="viewingConversation = null">关闭</button>
          </div>
        </div>
      </div>
      </Teleport>
    </main>

    <!-- 人物配置弹窗 -->
    <CharacterConfigDialog :show="dialogVisible" :config="character"
      @close="dialogVisible = false" @save="handleSaveConfig" />
  </div>
</template>

<script setup>
import { ref, reactive, nextTick, onMounted, watch } from "vue";
import axios from "axios";
import CharacterConfigDialog from "./components/CharacterConfigDialog.vue";

const API_BASE = "http://localhost:8000";

// ===================== 标签页 =====================
const activeTab = ref("chat");

// DOM
const chatBoxRef = ref(null);
const inputRef = ref(null);
const dialogVisible = ref(false);

// MBTI类型列表
const mbtiTypes = [
  { code: "INTJ", description: "建筑师" }, { code: "INTP", description: "逻辑学家" },
  { code: "ENTJ", description: "指挥官" }, { code: "ENTP", description: "辩论家" },
  { code: "INFJ", description: "提倡者" }, { code: "INFP", description: "调停者" },
  { code: "ENFJ", description: "主人公" }, { code: "ENFP", description: "竞选者" },
  { code: "ISTJ", description: "物流师" }, { code: "ISFJ", description: "守卫者" },
  { code: "ESTJ", description: "总经理" }, { code: "ESFJ", description: "执政官" },
  { code: "ISTP", description: "鉴赏家" }, { code: "ISFP", description: "探险家" },
  { code: "ESTP", description: "企业家" }, { code: "ESFP", description: "表演者" },
];

// ===================== 角色配置 =====================
const character = reactive({
  name: "", mbti: "", intro: "", personality: "", career: "", experience: "",
});
const activeSystemPrompt = ref("");
const currentRoleName = ref("");  // 空字符串 = 未选择角色

// ===================== 默认角色 =====================
const defaultCharacters = ref([]);
const previewChar = ref(null);

const loadDefaultCharacters = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/characters/defaults`);
    if (res.data.code === 200) {
      defaultCharacters.value = res.data.data.filter(c => c._default !== "custom");
    }
  } catch (err) { console.error("加载默认角色失败：", err); }
};

const selectDefaultChar = (charData) => {
  previewChar.value = charData;
  // 将默认角色数据填入 character 配置
  Object.assign(character, {
    name: charData.name,
    mbti: charData.mbti,
    intro: charData.intro || "",
    personality: charData.personality || "",
    career: charData.career || "",
    experience: charData.experience || "",
  });
  // 生成系统提示词
  activeSystemPrompt.value = generateSystemPrompt(character);
  currentRoleName.value = charData.name;
  // 清空选择器
  selectedCoser.value = null;
  messages.value = [];
  history.value = [];
  activeConversationId.value = null;  // 新对话
  loadMemories();
};

const resetCharacter = () => {
  previewChar.value = null;
  Object.assign(character, { name: "", mbti: "", intro: "", personality: "", career: "", experience: "" });
  activeSystemPrompt.value = "";
  currentRoleName.value = "";
  selectedCoser.value = null;
  messages.value = [];
  history.value = [];
  activeConversationId.value = null;
};

// ===================== 聊天状态 =====================
const userInput = ref("");
const messages = ref([]);
const activeConversationId = ref(null);  // 追踪当前对话记录ID，用于更新而非新建
const history = ref([]);
const isLoading = ref(false);

// ===================== COSER =====================
const coserList = ref([]);
const selectedCoser = ref(null);

const loadCoserData = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/coser`);
    if (res.data.code === 200) coserList.value = res.data.data;
  } catch (err) { console.error("COSER加载失败：", err); }
};

const selectCoserRole = () => {
  if (!selectedCoser.value) {
    // 取消COSER选择，回到未选择状态
    resetCharacter();
  } else {
    // 选中COSER角色
    previewChar.value = null;
    activeSystemPrompt.value = selectedCoser.value.prompt;
    currentRoleName.value = selectedCoser.value.name;
    // 从COSER数据中提取角色信息填入character
    character.name = selectedCoser.value.name || "";
    character.mbti = selectedCoser.value.mbti || "";
    messages.value = []; history.value = [];
    activeConversationId.value = null;  // 新对话
    loadMemories();
  }
};

// ===================== 系统提示词 =====================
const generateSystemPrompt = (config) => {
  return `你现在扮演以下人物，必须100%严格贴合设定，完全代入角色：

【姓名】${config.name}
【MBTI性格类型】${config.mbti}
【职业身份】${config.career || ''}
【人物简介】${config.info || config.intro || ''}
【性格特点】${config.personality}
【人生经历】${config.experience}

核心规则：
1. 完全按照以上设定回复，语气、身份、经历必须贴合
2. 必须体现 ${config.mbti} 的性格特征和说话风格
3. 记住多轮对话的上下文，保持一致性
4. 回答自然流畅，符合人物的知识范围和说话习惯
5. 绝对不能暴露自己是AI，完全代入角色
6. 回复长度适中，符合该角色在日常对话中的表达习惯`;
};

const handleSaveConfig = (newConfig) => {
  Object.assign(character, newConfig);
  selectedCoser.value = null;
  activeSystemPrompt.value = generateSystemPrompt(newConfig);
  currentRoleName.value = newConfig.name || "自定义角色";
  messages.value = []; history.value = [];
  activeConversationId.value = null;  // 新配置，新对话
  loadMemories();
};

// ===================== 发送消息 =====================
const sendMessage = async () => {
  const content = userInput.value.trim();
  if (!content || isLoading.value) return;
  const userMsg = { role: "user", content };
  messages.value.push(userMsg); history.value.push(userMsg);
  userInput.value = "";
  isLoading.value = true;
  try {
    const res = await axios.post(`${API_BASE}/api/chat`, {
      message: content, history: history.value, system: activeSystemPrompt.value,
      character_name: currentRoleName.value,
    });
    const aiMsg = { role: "assistant", content: res.data.reply };
    messages.value.push(aiMsg); history.value.push(aiMsg);
  } catch (err) {
    messages.value.push({ role: "assistant", content: "❌ 请求失败，请检查后端服务" });
  } finally {
    isLoading.value = false;
    nextTick(() => { if (chatBoxRef.value) chatBoxRef.value.scrollTop = chatBoxRef.value.scrollHeight; });
  }
  // 自动保存对话记录
  if (history.value.length > 0 && history.value.length % 4 === 0) {
    autoSaveConversation();
  }
};

const autoSaveConversation = async () => {
  try {
    const payload = {
      character: { name: currentRoleName.value, mbti: character.mbti, personality: character.personality, career: character.career, intro: character.intro, experience: character.experience },
      history: history.value,
      title: `${ character.name || currentRoleName.value }的对话_${new Date().toLocaleDateString()}`,
    };
    if (activeConversationId.value) {
      // 继续历史对话 → 更新已有记录
      await axios.put(`${API_BASE}/api/conversations/${activeConversationId.value}`, payload);
    } else {
      // 新对话 → 创建新记录
      const res = await axios.post(`${API_BASE}/api/conversations`, payload);
      if (res.data.code === 200) {
        activeConversationId.value = res.data.data.id;  // 记录ID以便后续更新
      }
    }
  } catch (err) { /* 静默失败 */ }
};

const saveCurrentConversation = async () => {
  try {
    await autoSaveConversation();
    alert("✅ 对话已保存");
  } catch (err) {
    alert("保存失败：" + err.message);
  }
};

// ===================== 导出 =====================
const exportChat = async (format) => {
  try {
    const res = await axios.post(`${API_BASE}/api/export`, {
      history: history.value, role_name: currentRoleName.value, format,
    });
    const blob = format === "txt"
      ? new Blob([res.data], { type: "text/plain;charset=utf-8" })
      : new Blob([JSON.stringify(res.data.data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `对话记录_${currentRoleName.value}_${new Date().toISOString().slice(0,10)}.${format}`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) { alert("导出失败，请检查后端服务"); }
};

// ===================== 角色保存/加载 =====================
const saveCurrentCharacter = async () => {
  try {
    await axios.post(`${API_BASE}/api/characters`, {
      name: character.name, mbti: character.mbti, personality: character.personality,
      career: character.career || '',
      intro: character.intro || '',
      experience: character.experience,
    });
    alert("✅ 角色配置已保存");
  } catch (err) { alert("保存失败"); }
};

const savedCharacters = ref([]);
const loadSavedCharacters = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/characters`);
    if (res.data.code === 200) savedCharacters.value = res.data.data;
  } catch (err) { console.error(err); }
};

const loadCharacter = (char) => {
  character.name = char.name; character.mbti = char.mbti;
  character.personality = char.personality; character.career = char.career;
  character.intro = char.intro; character.experience = char.experience;
  activeSystemPrompt.value = generateSystemPrompt(character);
  currentRoleName.value = char.name;
  selectedCoser.value = null;
  messages.value = []; history.value = [];
  loadMemories();
  activeTab.value = "chat";
  alert(`✅ 已加载角色：${char.name}`);
};

const deleteCharacter = async (id) => {
  if (!confirm("确认删除此角色配置？")) return;
  await axios.delete(`${API_BASE}/api/characters/${id}`);
  loadSavedCharacters();
};

// ===================== 对话记录 =====================
const savedConversations = ref([]);
const viewingConversation = ref(null);

const loadConversations = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/conversations`);
    if (res.data.code === 200) savedConversations.value = res.data.data;
  } catch (err) { console.error(err); }
};

const viewConversation = async (id) => {
  try {
    const res = await axios.get(`${API_BASE}/api/conversations/${id}`);
    if (res.data.code === 200) viewingConversation.value = res.data.data;
  } catch (err) { alert("加载失败"); }
};

const continueConversation = async (id) => {
  try {
    const res = await axios.get(`${API_BASE}/api/conversations/${id}`);
    if (res.data.code !== 200) { alert("加载对话失败"); return; }
    const conv = res.data.data;
    // 恢复角色配置
    const charData = conv.character || {};
    Object.assign(character, {
      name: charData.name || "",
      mbti: charData.mbti || "",
      personality: charData.personality || "",
      career: charData.career || "",
      intro: charData.intro || "",
      experience: charData.experience || "",
    });
    // 生成系统提示词
    activeSystemPrompt.value = generateSystemPrompt(character);
    currentRoleName.value = charData.name || "历史角色";
    // 恢复对话消息
    messages.value = [...(conv.history || [])];
    history.value = [...(conv.history || [])];
    // 清除COSER选择
    selectedCoser.value = null;
    // 记录当前对话ID，后续保存时更新而非新建
    activeConversationId.value = id;
    // 切换到对话页
    activeTab.value = "chat";
    // 加载该角色的记忆
    loadMemories();
    // 滚动到底部
    nextTick(() => { if (chatBoxRef.value) chatBoxRef.value.scrollTop = chatBoxRef.value.scrollHeight; });
  } catch (err) { alert("继续对话失败：" + err.message); }
};

const deleteConversation = async (id) => {
  if (!confirm("确认删除此对话记录？")) return;
  await axios.delete(`${API_BASE}/api/conversations/${id}`);
  loadConversations();
};

// ===================== 对话记忆 =====================
const memoryCount = ref(0);
const memories = ref([]);
const showMemories = ref(false);

const loadMemories = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/memory/${encodeURIComponent(currentRoleName.value)}`);
    if (res.data.code === 200) {
      memories.value = res.data.data;
      memoryCount.value = res.data.data.length;
    }
  } catch (err) { memoryCount.value = 0; memories.value = []; }
};

const extractMemories = async () => {
  if (history.value.length < 3) { alert("对话轮数不足，至少需要3轮对话"); return; }
  try {
    const res = await axios.post(`${API_BASE}/api/memory/extract`, {
      character_name: currentRoleName.value,
      history: history.value,
    });
    if (res.data.code === 200) {
      const data = res.data.data;
      memories.value = data.memories;
      memoryCount.value = data.memories.length;
      showMemories.value = true;
      if (data.added > 0) {
        alert(`✅ 成功提取 ${data.added} 条新记忆`);
      } else {
        alert("未提取到新的记忆信息");
      }
    }
  } catch (err) { alert("记忆保存失败：" + (err.response?.data?.message || err.message)); }
};

const deleteMemory = async (index) => {
  memories.value.splice(index, 1);
  memoryCount.value = memories.value.length;
  // 同步删除后端
  try {
    await axios.delete(`${API_BASE}/api/memory/${encodeURIComponent(currentRoleName.value)}/${index}`);
  } catch (err) { /* 静默 */ }
};

// ===================== 评估 =====================
const mbtiRunning = ref(false);
const bleuRunning = ref(false);
const mbtiEvalResult = ref(null);
const mbtiTypeAccuracy = ref(null);
const bleuEvalResult = ref(null);
const bleuMbtiType = ref("");

// toggleMbti：加载/收起/展开三态切换
const toggleMbti = () => {
  if (mbtiRunning.value) return;
  if (mbtiEvalResult.value) {
    // 已有数据，切换显示/隐藏
    showMbtiDetail.value = !showMbtiDetail.value;
    return;
  }
  // 首次加载
  loadMbtiResult();
};
const loadMbtiResult = async () => {
  mbtiRunning.value = true; mbtiEvalResult.value = null; mbtiTypeAccuracy.value = null;
  try {
    const latestRes = await axios.get(`${API_BASE}/api/eval/mbti/latest`);
    if (latestRes.data.code === 200 && latestRes.data.data) {
      mbtiEvalResult.value = latestRes.data.data.summary;
      mbtiTypeAccuracy.value = latestRes.data.data.type_accuracy;
      showMbtiDetail.value = true;
      // 同步加载错误分析
      try {
        const errRes = await axios.get(`${API_BASE}/api/eval/mbti/error_analysis`);
        if (errRes.data.code === 200) errorAnalysis.value = errRes.data.data;
      } catch (err) { /* 无错误分析文件 */ }
      mbtiRunning.value = false;
      return;
    }
  } catch (err) { /* 无已保存结果 */ }
  try {
    const response = await fetch(`${API_BASE}/api/eval/mbti/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'progress') {
            mbtiTypeAccuracy.value = { ...mbtiTypeAccuracy.value, ...data.type_accuracy };
            if (data.error) errorAnalysis.value = data.error;
          } else if (data.type === 'complete') {
            mbtiEvalResult.value = data.summary;
            mbtiTypeAccuracy.value = data.type_accuracy;
            showMbtiDetail.value = true;
            if (data.error) errorAnalysis.value = data.error;
          }
        }
      }
    }
  } catch (err) { alert("MBTI评估失败：" + err.message); }
  finally { mbtiRunning.value = false; }
};

// 强制在线重新评估（跳过缓存）
const runMbtiEvalStream = async () => {
  mbtiRunning.value = true; mbtiEvalResult.value = null; mbtiTypeAccuracy.value = null; errorAnalysis.value = null;
  try {
    const response = await fetch(`${API_BASE}/api/eval/mbti/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'progress') {
            mbtiTypeAccuracy.value = { ...mbtiTypeAccuracy.value, ...data.type_accuracy };
            if (data.error) errorAnalysis.value = data.error;
          } else if (data.type === 'complete') {
            mbtiEvalResult.value = data.summary;
            mbtiTypeAccuracy.value = data.type_accuracy;
            showMbtiDetail.value = true;
            // 流式完成后从文件加载错误分析（后端已保存）
            try {
              const errRes = await axios.get(`${API_BASE}/api/eval/mbti/error_analysis`);
              if (errRes.data.code === 200) errorAnalysis.value = errRes.data.data;
            } catch (err) { /* 无文件 */ }
          }
        }
      }
    }
  } catch (err) { alert("MBTI评估失败：" + err.message); }
  finally { mbtiRunning.value = false; }
};

// toggleBleu：加载/收起/展开三态切换
const toggleBleu = () => {
  if (bleuRunning.value) return;
  if (bleuEvalResult.value) {
    showBleuDetail.value = !showBleuDetail.value;
    return;
  }
  loadBleuResult();
};
const loadBleuResult = async () => {
  bleuRunning.value = true; bleuEvalResult.value = null;
  const selectedMbti = bleuMbtiType.value || null;
  if (!selectedMbti) {
    try {
      const latestRes = await axios.get(`${API_BASE}/api/eval/bleu/latest`);
      if (latestRes.data.code === 200 && latestRes.data.data) {
        bleuEvalResult.value = latestRes.data.data.summary; showBleuDetail.value = true;
        bleuRunning.value = false;
        return;
      }
    } catch (err) { /* 无已保存结果 */ }
  }
  try {
    const response = await fetch(`${API_BASE}/api/eval/bleu/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_samples: 20, mbti_type: selectedMbti }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'progress') {
            bleuEvalResult.value = { avg_bleu: data.avg_bleu, current: data.current, total: data.total };
          } else if (data.type === 'complete') {
            bleuEvalResult.value = data.summary; showBleuDetail.value = true;
          }
        }
      }
    }
  } catch (err) { alert("BLEU评估失败：" + err.message); }
  finally { bleuRunning.value = false; }
};

// 强制在线重新评估（跳过缓存，尊重MBTI类型选择）
const runBleuEvalStream = async () => {
  bleuRunning.value = true; bleuEvalResult.value = null;
  const selectedMbti = bleuMbtiType.value || null;
  try {
    const response = await fetch(`${API_BASE}/api/eval/bleu/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_samples: 20, mbti_type: selectedMbti }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'progress') {
            bleuEvalResult.value = { avg_bleu: data.avg_bleu, current: data.current, total: data.total, mbti_type: selectedMbti };
          } else if (data.type === 'complete') {
            bleuEvalResult.value = { ...data.summary, mbti_type: selectedMbti }; showBleuDetail.value = true;
          }
        }
      }
    }
  } catch (err) { alert("BLEU评估失败：" + err.message); }
  finally { bleuRunning.value = false; }
};

// 消融实验
const ablationRunning = ref(false);
const ablationResult = ref(null);
const ablationTypeComparison = ref(null);

// 折叠状态——初始收起，点击按钮加载后才展开
const showMbtiDetail = ref(false);
const showBleuDetail = ref(false);
const showAblationDetail = ref(false);

// toggleAblation：加载/收起/展开三态切换
const toggleAblation = () => {
  if (ablationRunning.value) return;
  if (ablationResult.value) {
    showAblationDetail.value = !showAblationDetail.value;
    return;
  }
  loadAblationResult();
};
const loadAblationResult = async () => {
  ablationRunning.value = true; ablationResult.value = null; ablationTypeComparison.value = null;
  try {
    const res = await axios.get(`${API_BASE}/api/eval/ablation`);
    if (res.data.code === 200) {
      ablationResult.value = res.data.data.summary;
      ablationTypeComparison.value = res.data.data.type_comparison;
      showAblationDetail.value = true;
    } else {
      alert(res.data.message || "请先在终端运行：python eval_mbti.py --ablation");
    }
  } catch (err) { alert("加载失败：" + err.message); }
  finally { ablationRunning.value = false; }
};

// 消融在线流式评估
const runAblationStream = async () => {
  ablationRunning.value = true; ablationResult.value = null; ablationTypeComparison.value = null;
  showAblationDetail.value = true;
  try {
    const response = await fetch(`${API_BASE}/api/eval/ablation/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'progress') {
            ablationTypeComparison.value = { ...ablationTypeComparison.value, ...data.type_comparison };
          } else if (data.type === 'complete') {
            ablationResult.value = data.summary;
            ablationTypeComparison.value = data.type_comparison;
            showAblationDetail.value = true;
          }
        }
      }
    }
  } catch (err) { alert("消融评估失败：" + err.message); }
  finally { ablationRunning.value = false; }
};

// 错误分析
const errorAnalysis = ref(null);
const showErrorDetail = ref(false);

const toggleError = () => {
  if (errorAnalysis.value) {
    showErrorDetail.value = !showErrorDetail.value;
    return;
  }
  loadErrorAnalysis();
};
const loadErrorAnalysis = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/eval/mbti/error_analysis`);
    if (res.data.code === 200) {
      errorAnalysis.value = res.data.data;
      showErrorDetail.value = true;
    } else {
      alert(res.data.message || "请先运行MBTI评估生成错误分析数据");
    }
  } catch (err) { alert("加载失败"); }
};

const loadEvalHistory = async () => {
  try {
    const mbtiRes = await axios.get(`${API_BASE}/api/eval/mbti/latest`);
    if (mbtiRes.data.code === 200) {
      const d = mbtiRes.data.data;
      mbtiEvalResult.value = { ...d.summary, total_samples: d.summary.total_samples };
      mbtiTypeAccuracy.value = d.type_accuracy || null;
    }
  } catch (err) { /* 静默 */ }
  try {
    const bleuRes = await axios.get(`${API_BASE}/api/eval/bleu/latest`);
    if (bleuRes.data.code === 200) {
      bleuEvalResult.value = bleuRes.data.data.summary;
    }
  } catch (err) { /* 静默 */ }
};

// ===================== 初始化 =====================
onMounted(() => {
  loadDefaultCharacters();
  loadCoserData();
});
</script>

<style scoped>
* { margin: 0; padding: 0; box-sizing: border-box; font-family: "Microsoft YaHei", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
.no-scrollbar::-webkit-scrollbar { display: none; }

/* ==================== 全局 ==================== */
.app-container { width: 100vw; height: 100vh; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%); display: flex; flex-direction: column; overflow: hidden; }

.top-header { padding: 20px 24px 12px; text-align: center; color: white; }
.top-header h1 { font-size: 26px; font-weight: 700; letter-spacing: 1px; margin-bottom: 4px; text-shadow: 0 2px 4px rgba(0,0,0,0.15); }
.top-header p { font-size: 13px; opacity: 0.85; margin-bottom: 12px; font-weight: 300; }

.tab-nav { display: flex; gap: 8px; justify-content: center; }
.tab-btn { padding: 9px 26px; border: 2px solid rgba(255,255,255,0.35); background: transparent; color: white; border-radius: 24px; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.25s; backdrop-filter: blur(4px); }
.tab-btn:hover { background: rgba(255,255,255,0.18); border-color: rgba(255,255,255,0.6); transform: translateY(-1px); }
.tab-btn.active { background: white; color: #6366f1; border-color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15); transform: translateY(-1px); }

/* ==================== 对话页 ==================== */
.chat-main { width: 92%; max-width: 1100px; margin: 0 auto; flex: 1; background: rgba(255,255,255,0.97); backdrop-filter: blur(10px); border-radius: 20px 20px 0 0; box-shadow: 0 -8px 30px rgba(0,0,0,0.12); display: flex; flex-direction: column; overflow: hidden; }
.chat-header { padding: 14px 22px; border-bottom: 1px solid #f1f5f9; background: #fafbfc; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.coser-select-wrapper { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #475569; flex: 1; min-width: 200px; }
.coser-select { padding: 8px 14px; border: 1.5px solid #e2e8f0; border-radius: 8px; outline: none; font-size: 13px; background: #fff; cursor: pointer; transition: border-color 0.2s; max-width: 250px; }
.coser-select:focus { border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,0.1); }
.config-btn, .export-btn, .save-btn, .memory-btn { padding: 8px 16px; border: none; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 500; color: white; transition: all 0.2s; white-space: nowrap; letter-spacing: 0.3px; }
.config-btn { background: #6366f1; }
.save-btn { background: #f59e0b; }
.memory-btn { background: #8b5cf6; }
.memory-btn:disabled { background: #c4b5fd; cursor: not-allowed; opacity: 0.7; }
.memory-btn:hover:not(:disabled) { background: #7c3aed; transform: translateY(-1px); }
.save-conv-btn { padding: 8px 16px; border: none; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 500; color: white; background: #06b6d4; transition: all 0.2s; white-space: nowrap; letter-spacing: 0.3px; }
.save-conv-btn:disabled { background: #67e8f9; cursor: not-allowed; opacity: 0.7; }
.save-conv-btn:hover:not(:disabled) { background: #0891b2; transform: translateY(-1px); }
.export-btn { background: #10b981; }
.export-btn:disabled { background: #6ee7b7; cursor: not-allowed; opacity: 0.7; }
.config-btn:hover { background: #4f46e5; transform: translateY(-1px); }
.save-btn:hover { background: #d97706; transform: translateY(-1px); }
.export-btn:hover:not(:disabled) { background: #059669; transform: translateY(-1px); }

.chat-messages { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; background: #f8fafc; }
.empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; }
.empty-icon { font-size: 56px; opacity: 0.4; }
.empty-title { font-size: 18px; font-weight: 600; color: #334155; }
.empty-desc { font-size: 14px; color: #94a3b8; }
.message-item { display: flex; gap: 10px; max-width: 75%; animation: fadeIn 0.25s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.message-item.user { margin-left: auto; flex-direction: row-reverse; }
.avatar { width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #e0e7ff, #c7d2fe); display: flex; align-items: center; justify-content: center; font-size: 17px; flex-shrink: 0; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
.message-box { display: flex; flex-direction: column; gap: 4px; }
.role-tag { font-size: 11px; color: #6366f1; font-weight: 600; padding-left: 6px; }
.message-content { padding: 12px 16px; border-radius: 18px; font-size: 14px; line-height: 1.65; word-break: break-word; white-space: pre-wrap; }
.message-item.assistant .message-content { background: white; color: #334155; border-bottom-left-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.message-item.user .message-content { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; border-bottom-right-radius: 6px; }
.loading { align-self: flex-start; padding: 10px 16px; background: white; border-radius: 14px; font-size: 13px; color: #94a3b8; display: flex; align-items: center; gap: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.loading::before { content: ""; width: 8px; height: 8px; background: #6366f1; border-radius: 50%; animation: pulse 1s infinite; }
@keyframes pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
.input-footer { display: flex; gap: 12px; padding: 16px 22px; border-top: 1px solid #f1f5f9; background: white; align-items: flex-end; }
.message-input { flex: 1; padding: 12px 16px; border: 1.5px solid #e2e8f0; border-radius: 24px; font-size: 14px; line-height: 1.5; min-height: 46px; max-height: 120px; overflow-y: auto; resize: none; outline: none; transition: border-color 0.2s, box-shadow 0.2s; }
.message-input:focus { border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,0.08); }
.send-btn { padding: 0 26px; height: 46px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; border: none; border-radius: 24px; font-size: 14px; font-weight: 600; cursor: pointer; white-space: nowrap; transition: all 0.2s; letter-spacing: 0.5px; }
.send-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(99,102,241,0.3); }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

/* 记忆面板 */
.memory-panel { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-bottom: 2px solid #6ee7b7; padding: 14px 22px; max-height: 200px; overflow-y: auto; }
.memory-panel-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; font-weight: 600; color: #065f46; font-size: 14px; }
.close-icon-sm { width: 26px; height: 26px; border: none; background: rgba(255,255,255,0.8); border-radius: 50%; cursor: pointer; font-size: 14px; color: #065f46; transition: background 0.2s; }
.close-icon-sm:hover { background: #6ee7b7; }
.memory-empty { font-size: 13px; color: #94a3b8; text-align: center; padding: 12px; }
.memory-item { display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: white; border-radius: 8px; margin: 4px 0; font-size: 13px; color: #374151; box-shadow: 0 1px 2px rgba(0,0,0,0.03); transition: transform 0.15s; }
.memory-item:hover { transform: translateX(2px); }
.memory-del { border: none; background: #fee2e2; color: #ef4444; border-radius: 6px; cursor: pointer; font-size: 12px; padding: 3px 8px; transition: background 0.15s; }
.memory-del:hover { background: #fecaca; }

/* ==================== 评估页 ==================== */
.eval-main { width: 92%; max-width: 1100px; margin: 0 auto; flex: 1; background: rgba(255,255,255,0.97); backdrop-filter: blur(10px); border-radius: 20px 20px 0 0; box-shadow: 0 -8px 30px rgba(0,0,0,0.12); padding: 28px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px; }
.eval-section { background: #f8fafc; border: 1px solid #f1f5f9; border-radius: 16px; padding: 24px; transition: box-shadow 0.2s; }
.eval-section:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.04); }
.eval-section h2 { font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
.eval-desc { font-size: 13px; color: #94a3b8; margin-bottom: 16px; line-height: 1.5; }
.eval-actions { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; flex-wrap: wrap; }
.eval-run-btn { padding: 10px 24px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; letter-spacing: 0.3px; }
.eval-run-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(99,102,241,0.3); }
.eval-run-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.eval-hint { font-size: 12px; color: #cbd5e1; }
.bleu-mbti-select { padding: 9px 14px; border: 1.5px solid #e2e8f0; border-radius: 10px; font-size: 13px; outline: none; background: white; cursor: pointer; transition: border-color 0.2s; }
.bleu-mbti-select:focus { border-color: #6366f1; }
.eval-progress { padding: 14px 18px; background: #fef9c3; border: 1px solid #fde68a; border-radius: 10px; color: #a16207; font-size: 13px; font-weight: 500; animation: pulse 1.5s infinite; }
.eval-result-card { background: white; border: 1px solid #f1f5f9; border-radius: 14px; padding: 22px; margin-top: 12px; }
.eval-summary { display: flex; gap: 48px; margin-bottom: 18px; flex-wrap: wrap; }
.eval-metric { text-align: center; min-width: 80px; }
.metric-value { display: block; font-size: 30px; font-weight: 700; color: #6366f1; }
.metric-label { font-size: 12px; color: #94a3b8; margin-top: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.type-bars { display: flex; flex-direction: column; gap: 6px; }
.type-bar-row { display: flex; align-items: center; gap: 10px; padding: 2px 0; }
.type-label { width: 48px; font-size: 13px; font-weight: 600; color: #475569; text-align: right; }
.type-bar-track { flex: 1; height: 24px; background: #f1f5f9; border-radius: 12px; overflow: hidden; }
.type-bar-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #a855f7); border-radius: 12px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); min-width: 4px; }
.type-score { font-size: 12px; color: #64748b; width: 70px; font-weight: 500; }

/* ==================== 记录页 ==================== */
.history-main { width: 92%; max-width: 1100px; margin: 0 auto; flex: 1; background: rgba(255,255,255,0.97); backdrop-filter: blur(10px); border-radius: 20px 20px 0 0; box-shadow: 0 -8px 30px rgba(0,0,0,0.12); padding: 28px; overflow-y: auto; display: flex; flex-direction: column; gap: 28px; }
.history-section h2 { font-size: 17px; font-weight: 700; color: #1e293b; margin-bottom: 12px; }
.history-card { background: #f8fafc; border: 1px solid #f1f5f9; border-radius: 12px; padding: 16px 20px; margin: 8px 0; transition: box-shadow 0.2s, transform 0.15s; }
.history-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.04); transform: translateY(-1px); }
.history-card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
.history-card-header strong { font-size: 15px; color: #1e293b; }
.history-card-body { display: flex; gap: 20px; font-size: 13px; color: #64748b; margin-bottom: 10px; }
.history-card-actions { display: flex; gap: 8px; }
.history-card-actions button { padding: 6px 14px; border: 1px solid #e2e8f0; background: white; border-radius: 7px; cursor: pointer; font-size: 12px; font-weight: 500; transition: all 0.15s; color: #475569; }
.history-card-actions button:hover { background: #f1f5f9; border-color: #cbd5e1; }
.history-card-actions button:first-child { background: #6366f1; color: white; border-color: #6366f1; }
.history-card-actions button:first-child:hover { background: #4f46e5; transform: translateY(-1px); box-shadow: 0 2px 8px rgba(99,102,241,0.3); }
.history-card-actions button.danger { color: #ef4444; border-color: #fecaca; }
.history-card-actions button.danger:hover { background: #fef2f2; }
.mbti-badge { padding: 3px 10px; background: #6366f1; color: white; border-radius: 12px; font-size: 11px; font-weight: 600; letter-spacing: 0.3px; }
.time-text { font-size: 11px; color: #94a3b8; }

/* 查看对话弹窗 */
.view-dialog-mask { position: fixed; inset: 0; background: rgba(15,23,42,0.55); backdrop-filter: blur(3px); display: flex; align-items: center; justify-content: center; z-index: 99998; animation: maskIn 0.18s ease; }
@keyframes maskIn { from { opacity: 0; } to { opacity: 1; } }
.view-dialog-box { width: 720px; max-width: 94vw; max-height: 90vh; background: #fff; border-radius: 14px; display: flex; flex-direction: column; box-shadow: 0 20px 60px rgba(0,0,0,0.2); overflow: hidden; animation: viewIn 0.22s cubic-bezier(0.16,1,0.3,1); }
@keyframes viewIn { from { opacity: 0; transform: scale(0.94) translateY(10px); } to { opacity: 1; transform: scale(1) translateY(0); } }

/* 头部 */
.view-dialog-head { padding: 18px 24px 14px; border-bottom: 1px solid #f1f5f9; flex-shrink: 0; background: #fafbfc; }
.view-dialog-title-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.view-dialog-title-row h3 { margin: 0; font-size: 17px; font-weight: 700; color: #0f172a; }
.view-close-btn { width: 32px; height: 32px; border: none; background: #f1f5f9; border-radius: 8px; font-size: 18px; color: #64748b; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.15s; line-height: 1; flex-shrink: 0; }
.view-close-btn:hover { background: #e2e8f0; color: #0f172a; }
.view-dialog-meta { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #64748b; flex-wrap: wrap; }
.view-meta-tag { display: inline-block; padding: 2px 10px; background: #6366f1; color: #fff; border-radius: 10px; font-size: 11px; font-weight: 600; }
.view-meta-count { margin-left: auto; font-size: 12px; color: #94a3b8; }

/* 消息列表 */
.view-dialog-body { padding: 16px 24px; overflow-y: auto; flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 10px; }
.view-dialog-body::-webkit-scrollbar { width: 5px; }
.view-dialog-body::-webkit-scrollbar-track { background: transparent; }
.view-dialog-body::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 10px; }
.view-dialog-body::-webkit-scrollbar-thumb:hover { background: #9ca3af; }

.view-msg { padding: 12px 16px; border-radius: 12px; }
.view-msg.user { background: #eff6ff; border: 1px solid #dbeafe; }
.view-msg.assistant { background: #f8fafc; border: 1px solid #f1f5f9; }
.view-msg-role { display: block; font-size: 12px; font-weight: 600; color: #64748b; margin-bottom: 4px; }
.view-msg-text { font-size: 14px; color: #334155; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }

/* 底部 */
.view-dialog-foot { display: flex; align-items: center; justify-content: flex-end; gap: 10px; padding: 14px 24px; border-top: 1px solid #f1f5f9; flex-shrink: 0; background: #fafbfc; }
.view-foot-btn { padding: 9px 20px; border: 1.5px solid #e2e8f0; background: #fff; color: #475569; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.15s; }
.view-foot-btn:hover { background: #f1f5f9; border-color: #cbd5e1; }
.view-foot-btn.primary { background: #6366f1; color: #fff; border-color: #6366f1; font-weight: 600; }
.view-foot-btn.primary:hover { background: #4f46e5; transform: translateY(-1px); box-shadow: 0 2px 8px rgba(99,102,241,0.3); }

/* ==================== 默认角色选择 ==================== */
.default-char-section { width: 92%; max-width: 1100px; margin: 0 auto; padding: 20px 0; }
.default-char-section h3 { font-size: 16px; color: #475569; margin-bottom: 16px; font-weight: 500; }
.default-char-section h3 a { color: #6366f1; font-weight: 600; }
.default-char-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.default-char-card { background: white; border: 2px solid #e2e8f0; border-radius: 14px; padding: 16px; cursor: pointer; transition: all 0.2s; }
.default-char-card:hover { border-color: #6366f1; box-shadow: 0 4px 16px rgba(99,102,241,0.12); transform: translateY(-2px); }
.default-char-card.selected { border-color: #6366f1; background: #eef2ff; }
.char-card-name { font-size: 15px; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
.char-card-mbti { display: inline-block; padding: 2px 8px; background: #6366f1; color: white; border-radius: 8px; font-size: 11px; font-weight: 600; margin-bottom: 6px; }
.char-card-career { font-size: 12px; color: #64748b; margin-bottom: 6px; }
.char-card-intro { font-size: 11px; color: #94a3b8; line-height: 1.5; }

/* ==================== 当前角色信息条 ==================== */
.current-role-bar { width: 92%; max-width: 1100px; margin: 0 auto 8px; padding: 10px 18px; background: linear-gradient(135deg, #eef2ff, #e0e7ff); border: 1px solid #c7d2fe; border-radius: 10px; display: flex; align-items: center; justify-content: space-between; font-size: 13px; color: #4338ca; }
.current-role-bar button { padding: 5px 14px; background: white; border: 1px solid #c7d2fe; border-radius: 7px; color: #6366f1; cursor: pointer; font-size: 12px; font-weight: 500; transition: all 0.15s; }
.current-role-bar button:hover { background: #6366f1; color: white; border-color: #6366f1; }
</style>
