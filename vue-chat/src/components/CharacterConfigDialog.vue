<template>
  <Teleport to="body">
  <div v-if="show" class="d-overlay" @click.self="$emit('close')">
    <div class="d-panel">
      <!-- 头部 -->
      <div class="d-header">
        <h2>人物配置</h2>
        <button class="d-close" @click="$emit('close')">&times;</button>
      </div>

      <!-- 可滚动内容 -->
      <div class="d-body">
        <div class="d-field">
          <label>姓名</label>
          <input v-model="form.name" placeholder="请输入角色姓名" />
        </div>
        <div class="d-field">
          <label>MBTI 性格类型</label>
          <select v-model="form.mbti">
            <option value="">请选择 MBTI 类型</option>
            <option v-for="t in mbtiOptions" :key="t.code" :value="t.code">{{ t.code }} - {{ t.label }}</option>
          </select>
        </div>
        <div class="d-field">
          <label>职业身份</label>
          <input v-model="form.career" placeholder="如：心理咨询师、游戏制作人" />
        </div>
        <div class="d-field">
          <label>性格特点</label>
          <textarea v-model="form.personality" rows="2" placeholder="如：温柔体贴、善于倾听、共情能力强"></textarea>
        </div>
        <div class="d-field">
          <label>人物简介</label>
          <textarea v-model="form.intro" rows="2" placeholder="角色的背景介绍、身份设定等"></textarea>
        </div>
        <div class="d-field">
          <label>人生经历</label>
          <textarea v-model="form.experience" rows="3" placeholder="角色的成长经历、关键事件等"></textarea>
        </div>
      </div>

      <!-- 底部 -->
      <div class="d-footer">
        <button class="d-btn-cancel" @click="$emit('close')">取消</button>
        <button class="d-btn-ok" @click="handleSave">保存配置</button>
      </div>
    </div>
  </div>
  </Teleport>
</template>

<script setup>
import { reactive, watch } from "vue";

const props = defineProps({
  show: { type: Boolean, default: false },
  config: { type: Object, required: true },
});
const emit = defineEmits(["close", "save"]);

const mbtiOptions = [
  { code: "INTJ", label: "建筑师" }, { code: "INTP", label: "逻辑学家" },
  { code: "ENTJ", label: "指挥官" }, { code: "ENTP", label: "辩论家" },
  { code: "INFJ", label: "提倡者" }, { code: "INFP", label: "调停者" },
  { code: "ENFJ", label: "主人公" }, { code: "ENFP", label: "竞选者" },
  { code: "ISTJ", label: "物流师" }, { code: "ISFJ", label: "守卫者" },
  { code: "ESTJ", label: "总经理" }, { code: "ESFJ", label: "执政官" },
  { code: "ISTP", label: "鉴赏家" }, { code: "ISFP", label: "探险家" },
  { code: "ESTP", label: "企业家" }, { code: "ESFP", label: "表演者" },
];

const form = reactive({ name: "", mbti: "", intro: "", personality: "", career: "", experience: "" });

watch(() => props.config, (v) => { if (v) Object.assign(form, v); }, { immediate: true, deep: true });

const handleSave = () => {
  emit("save", { ...form });
  emit("close");
};
</script>

<style scoped>
/* === 遮罩 === */
.d-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 99999;
  padding: 20px;
}

/* === 面板 === */
.d-panel {
  width: 500px;
  max-width: 100%;
  max-height: calc(100vh - 40px);
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 16px 48px rgba(0,0,0,0.18);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* === 头部 === */
.d-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid #eee;
  flex-shrink: 0;
}
.d-header h2 { margin: 0; font-size: 16px; font-weight: 700; color: #111; }
.d-close {
  width: 30px; height: 30px; border: none; background: #f3f4f6;
  border-radius: 6px; font-size: 20px; color: #666; cursor: pointer;
  display: flex; align-items: center; justify-content: center; line-height: 1;
  transition: background 0.15s;
}
.d-close:hover { background: #e5e7eb; color: #111; }

/* === 内容区 === */
.d-body {
  padding: 16px 20px;
  overflow-y: auto;
  flex: 1;
  min-height: 0;
}
.d-body::-webkit-scrollbar { width: 5px; }
.d-body::-webkit-scrollbar-thumb { background: #ccc; border-radius: 10px; }

.d-field {
  margin-bottom: 14px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.d-field:last-child { margin-bottom: 0; }
.d-field label { font-size: 13px; font-weight: 600; color: #374151; }
.d-field input,
.d-field textarea,
.d-field select {
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  font-family: inherit;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.d-field input:focus,
.d-field textarea:focus,
.d-field select:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
}
.d-field textarea { resize: vertical; min-height: 40px; }

/* === 底部 === */
.d-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 12px 20px;
  border-top: 1px solid #eee;
  flex-shrink: 0;
}
.d-btn-cancel {
  padding: 8px 18px;
  border: 1px solid #d1d5db;
  background: #fff;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  color: #374151;
}
.d-btn-cancel:hover { background: #f9fafb; }
.d-btn-ok {
  padding: 8px 20px;
  border: none;
  background: #6366f1;
  color: #fff;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
}
.d-btn-ok:hover { background: #4f46e5; }
</style>
