// YouTube双语字幕翻译器 - 生产版
class YouTubeSubtitleTranslator {
  constructor() {
    this.initLogger();
    this.logger.info('🚀 YouTube双语字幕翻译器启动...');
    
    this.settings = {
      apiUrl: 'https://api.openai.com/v1',
      apiKey: '',
      targetLang: '简体中文',
      model: 'gpt-4o-mini',
      autoTranslate: true
    };
    
    this.currentSubtitleText = '';
    this.translationCache = new Map();
    this.translationContainer = null;
    this.isTranslating = false;
    
    this.init();
  }
  
  initLogger() {
    if (typeof window.debugLogger === 'undefined') {
      window.debugLogger = {
        logs: [],
        log: function(level, message, data) {
          const timestamp = new Date().toISOString();
          const logEntry = { timestamp, level, message, data };
          this.logs.push(logEntry);
          console[level.toLowerCase()](message, data || '');
        },
        info: function(message, data) { this.log('INFO', message, data); },
        warn: function(message, data) { this.log('WARN', message, data); },
        error: function(message, data) { this.log('ERROR', message, data); },
        debug: function(message, data) { this.log('DEBUG', message, data); },
        exportLogs: function() {
          const logText = this.logs.map(log => 
            `[${log.timestamp}] [${log.level}] ${log.message}${log.data ? '\\n' + JSON.stringify(log.data, null, 2) : ''}`
          ).join('\\n\\n');
          const blob = new Blob([logText], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `youtube-subtitle-debug-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }
      };
    }
    this.logger = window.debugLogger;
  }
  
  async init() {
    await this.loadSettings();
    
    if (!this.settings.apiKey) {
      this.logger.error('❌ 未配置API密钥');
      this.showStatusInfo('未配置API密钥！请点击插件图标配置。');
      return;
    }
    
    this.createTranslationContainer();
    this.startSubtitleMonitoring();
    this.setupMessageListener();
    
    this.logger.info('✅ 初始化完成');
    this.showStatusInfo('插件已启动，等待检测字幕...');
  }
  
  async loadSettings() {
    return new Promise((resolve) => {
      chrome.storage.sync.get(['apiUrl', 'apiKey', 'targetLang', 'model', 'autoTranslate'], (result) => {
        this.settings = {
          apiUrl: result.apiUrl || this.settings.apiUrl,
          apiKey: result.apiKey || '',
          targetLang: result.targetLang || this.settings.targetLang,
          model: result.model || this.settings.model,
          autoTranslate: result.autoTranslate !== false
        };
        resolve();
      });
    });
  }
  
  setupMessageListener() {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.type === 'SETTINGS_UPDATED') {
        this.settings = message.settings;
        this.logger.info('🔄 设置已更新');
      }
    });
  }
  
  createTranslationContainer() {
    // 移除现有容器
    const existingContainer = document.getElementById('youtube-bilingual-subtitles');
    if (existingContainer) {
      existingContainer.remove();
    }
    
    // 创建新容器
    this.translationContainer = document.createElement('div');
    this.translationContainer.id = 'youtube-bilingual-subtitles';
    this.translationContainer.innerHTML = `
      <div class="subtitle-container">
        <div class="original-subtitle"></div>
        <div class="translated-subtitle"></div>
      </div>
    `;
    
    // 直接插入到body确保可见
    document.body.appendChild(this.translationContainer);
    
    this.logger.info('✅ 双语字幕容器创建成功');
  }
  
  startSubtitleMonitoring() {
    // 监听字幕变化
    const observer = new MutationObserver((mutations) => {
      let shouldCheck = false;
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList' || mutation.type === 'characterData') {
          shouldCheck = true;
        }
      });
      if (shouldCheck) {
        this.checkSubtitleChange();
      }
    });
    
    // 观察字幕容器
    const subtitleSelectors = [
      '.caption-window',
      '.ytp-caption-segment', 
      '.captions-text',
      '.ytp-caption-window-container'
    ];
    
    const startObserving = () => {
      subtitleSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(element => {
          observer.observe(element, {
            childList: true,
            subtree: true,
            characterData: true
          });
        });
      });
      
      // 观察整个document以防字幕容器动态创建
      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    };
    
    startObserving();
    
    // 定期检查和容器维护
    setInterval(() => {
      this.checkSubtitleChange();
      this.maintainContainer();
    }, 2000);
  }
  
  maintainContainer() {
    const container = document.getElementById('youtube-bilingual-subtitles');
    if (!container && this.translationContainer) {
      this.logger.warn('⚠️ 容器被移除，重新创建');
      this.createTranslationContainer();
    }
  }
  
  async checkSubtitleChange() {
    if (!this.settings.autoTranslate || !this.settings.apiKey) {
      return;
    }
    
    const currentSubtitle = this.getCurrentSubtitle();
    
    if (currentSubtitle && currentSubtitle !== this.currentSubtitleText) {
      this.currentSubtitleText = currentSubtitle;
      this.logger.info('✨ 检测到字幕变化:', currentSubtitle);
      this.showStatusInfo(`检测到字幕: ${currentSubtitle.substring(0, 30)}...`);
      
      this.updateSubtitleDisplay(currentSubtitle, '翻译中...');
      await this.translateSubtitle(currentSubtitle);
      
    } else if (!currentSubtitle && this.currentSubtitleText) {
      this.currentSubtitleText = '';
      this.updateSubtitleDisplay('', '');
    }
  }
  
  getCurrentSubtitle() {
    const selectors = [
      '.caption-window .captions-text',
      '.ytp-caption-segment',
      '.caption-window',
      '.html5-captions-text'
    ];
    
    for (const selector of selectors) {
      const elements = document.querySelectorAll(selector);
      for (const element of elements) {
        if (element && element.textContent && element.textContent.trim()) {
          return element.textContent.trim();
        }
      }
    }
    
    return null;
  }
  
  async translateSubtitle(text) {
    if (this.isTranslating) return;
    
    // 检查缓存
    if (this.translationCache.has(text)) {
      const translation = this.translationCache.get(text);
      this.updateSubtitleDisplay(text, translation);
      return;
    }
    
    this.isTranslating = true;
    this.logger.info('🌍 开始翻译:', text);
    
    try {
      const translation = await this.callTranslationAPI(text);
      this.translationCache.set(text, translation);
      this.updateSubtitleDisplay(text, translation);
      this.showStatusInfo(`翻译完成: ${translation.substring(0, 30)}...`);
      
    } catch (error) {
      this.logger.error('❌ 翻译失败:', error.message);
      this.updateSubtitleDisplay(text, `[翻译失败]`);
      this.showStatusInfo(`翻译失败: ${error.message}`);
    } finally {
      this.isTranslating = false;
    }
  }
  
  async callTranslationAPI(text) {
    const prompt = `你是一个专业的字幕翻译专家。请将以下英文字幕翻译成${this.settings.targetLang}，要求：

1. 保持原意准确
2. 使用自然流畅的${this.settings.targetLang}表达
3. 保持技术术语的准确性
4. 适合字幕显示的简洁表达
5. 直接返回翻译结果，不要解释

请翻译以下字幕：`;
    
    const response = await fetch(this.settings.apiUrl + '/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.settings.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.settings.model,
        messages: [
          { role: 'system', content: prompt },
          { role: 'user', content: text }
        ],
        temperature: 0.7,
        max_tokens: 500
      })
    });
    
    if (!response.ok) {
      throw new Error(`API调用失败: ${response.status}`);
    }
    
    const data = await response.json();
    if (!data.choices?.[0]?.message?.content) {
      throw new Error('API返回格式错误');
    }
    
    return data.choices[0].message.content.trim();
  }
  
  updateSubtitleDisplay(original, translated) {
    if (!this.translationContainer) {
      this.createTranslationContainer();
      return;
    }
    
    // 验证容器是否还在DOM中
    const containerInDOM = document.getElementById('youtube-bilingual-subtitles');
    if (!containerInDOM) {
      this.createTranslationContainer();
      return;
    }
    
    const originalElement = this.translationContainer.querySelector('.original-subtitle');
    const translatedElement = this.translationContainer.querySelector('.translated-subtitle');
    
    if (!originalElement || !translatedElement) {
      this.createTranslationContainer();
      return;
    }
    
    // 更新文本内容
    originalElement.textContent = original;
    translatedElement.textContent = translated;
    
    // 显示/隐藏逻辑
    const shouldShow = (original && original.trim()) || (translated && translated.trim());
    
    if (shouldShow) {
      // 强制显示容器
      this.translationContainer.style.display = 'block';
      this.translationContainer.style.visibility = 'visible';
      this.translationContainer.style.opacity = '1';
      this.translationContainer.classList.add('force-show');
    } else {
      // 隐藏容器
      this.translationContainer.style.display = 'none';
      this.translationContainer.classList.remove('force-show');
    }
  }
  
  showStatusInfo(message) {
    // 显示状态信息
    let statusDiv = document.getElementById('subtitle-status-info');
    if (!statusDiv) {
      statusDiv = document.createElement('div');
      statusDiv.id = 'subtitle-status-info';
      statusDiv.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0,0,0,0.8);
        color: #fff;
        padding: 8px 12px;
        border-radius: 5px;
        z-index: 10000;
        font-size: 12px;
        max-width: 300px;
      `;
      document.body.appendChild(statusDiv);
    }
    
    statusDiv.innerHTML = `
      <div><strong>🔧 字幕翻译状态</strong></div>
      <div>${message}</div>
      <div style="margin-top: 5px; font-size: 10px; opacity: 0.7;">
        ${new Date().toLocaleTimeString()}
      </div>
    `;
  }
}

// 初始化
const initTranslator = () => {
  window.debugLogger?.info('🎬 开始初始化翻译器...');
  new YouTubeSubtitleTranslator();
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initTranslator, 1000);
  });
} else {
  setTimeout(initTranslator, 1000);
}