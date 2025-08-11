// YouTube双语字幕翻译器 - Content Script (带日志版)
class YouTubeSubtitleTranslator {
  constructor() {
    // 初始化日志器
    this.initLogger();
    this.logger.info('🚀 YouTube双语字幕翻译器开始初始化...');
    
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
    
    // 添加导出日志按钮
    this.createLogExportButton();
    
    this.init();
  }
  
  initLogger() {
    // 确保日志器存在
    if (typeof window.debugLogger === 'undefined') {
      // 如果全局日志器不存在，创建一个简单的
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
            `[${log.timestamp}] [${log.level}] ${log.message}${log.data ? '\n' + JSON.stringify(log.data, null, 2) : ''}`
          ).join('\n\n');
          const blob = new Blob([logText], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `youtube-subtitle-debug-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        },
        clearLogs: function() { this.logs = []; }
      };
    }
    this.logger = window.debugLogger;
  }
  
  createLogExportButton() {
    // 创建日志导出按钮
    const logButton = document.createElement('button');
    logButton.innerHTML = '📋 导出调试日志';
    logButton.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      z-index: 10001;
      background: #4CAF50;
      color: white;
      border: none;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    `;
    
    logButton.onclick = () => {
      if (window.debugLogger) {
        window.debugLogger.exportLogs();
      }
    };
    
    document.body.appendChild(logButton);
    this.logger.info('📋 日志导出按钮已创建');
  }
  
  async init() {
    this.logger.info('📋 开始加载设置...');
    // 加载设置
    await this.loadSettings();
    this.logger.info('⚙️ 当前设置:', this.settings);
    
    // 检查API密钥
    if (!this.settings.apiKey) {
      this.logger.error('❌ 未配置API密钥！请在插件设置中配置。');
      this.showDebugInfo('未配置API密钥！请点击插件图标配置。');
      return;
    }
    
    // 创建翻译显示容器
    this.logger.info('🎨 创建翻译显示容器...');
    this.createTranslationContainer();
    
    // 开始监听字幕变化
    this.logger.info('👁️ 开始监听字幕变化...');
    this.startSubtitleMonitoring();
    
    // 监听设置更新
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.type === 'SETTINGS_UPDATED') {
        this.settings = message.settings;
        this.logger.info('🔄 设置已更新:', this.settings);
      }
    });
    
    this.logger.info('✅ YouTube双语字幕翻译器初始化完成');
    this.showDebugInfo('插件已启动，等待检测字幕...');
    
    // 立即检查一次字幕
    setTimeout(() => {
      this.logger.info('🔍 执行首次字幕检查...');
      this.checkSubtitleChange();
    }, 3000);
  }
  
  showDebugInfo(message) {
    // 在页面上显示调试信息
    let debugDiv = document.getElementById('subtitle-debug-info');
    if (!debugDiv) {
      debugDiv = document.createElement('div');
      debugDiv.id = 'subtitle-debug-info';
      debugDiv.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0,0,0,0.8);
        color: #fff;
        padding: 10px;
        border-radius: 5px;
        z-index: 10000;
        font-size: 12px;
        max-width: 300px;
      `;
      document.body.appendChild(debugDiv);
    }
    debugDiv.innerHTML = `
      <div><strong>🔧 插件调试信息</strong></div>
      <div>${message}</div>
      <div style="margin-top: 5px; font-size: 10px; opacity: 0.7;">
        ${new Date().toLocaleTimeString()}
      </div>
      <div style="margin-top: 5px;">
        <button onclick="window.debugLogger && window.debugLogger.exportLogs()" style="background:#007bff;color:white;border:none;padding:3px 6px;border-radius:2px;font-size:10px;cursor:pointer;">导出日志</button>
        <button onclick="window.debugLogger && window.debugLogger.clearLogs()" style="background:#dc3545;color:white;border:none;padding:3px 6px;border-radius:2px;font-size:10px;cursor:pointer;">清空日志</button>
      </div>
    `;
  }
  
  async loadSettings() {
    return new Promise((resolve) => {
      chrome.storage.sync.get(['apiUrl', 'apiKey', 'targetLang', 'model', 'autoTranslate'], (result) => {
        this.settings = {
          apiUrl: result.apiUrl || 'https://api.openai.com/v1',
          apiKey: result.apiKey || '',
          targetLang: result.targetLang || '简体中文',
          model: result.model || 'gpt-4o-mini',
          autoTranslate: result.autoTranslate !== false
        };
        resolve();
      });
    });
  }
  
  createTranslationContainer() {
    // 查找YouTube播放器容器
    const playerSelectors = [
      '#movie_player',
      '.html5-video-player',
      '#player-container',
      '.player-container'
    ];
    
    let playerContainer = null;
    for (const selector of playerSelectors) {
      playerContainer = document.querySelector(selector);
      if (playerContainer) {
        this.logger.info('📺 找到播放器容器:', selector);
        break;
      }
    }
    
    if (!playerContainer) {
      this.logger.error('❌ 未找到YouTube播放器容器，3秒后重试...');
      setTimeout(() => this.createTranslationContainer(), 3000);
      return;
    }
    
    // 创建翻译显示容器
    this.translationContainer = document.createElement('div');
    this.translationContainer.id = 'youtube-bilingual-subtitles';
    this.translationContainer.innerHTML = `
      <div class="subtitle-container">
        <div class="original-subtitle"></div>
        <div class="translated-subtitle"></div>
      </div>
    `;
    
    // 添加样式
    const style = document.createElement('style');
    style.textContent = `
      #youtube-bilingual-subtitles {
        position: absolute !important;
        bottom: 80px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 1000 !important;
        pointer-events: none !important;
        font-family: 'Roboto', 'YouTube Noto', sans-serif !important;
      }
      
      #youtube-bilingual-subtitles .subtitle-container {
        background: rgba(0, 0, 0, 0.8) !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
        max-width: 80vw !important;
        text-align: center !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
      }
      
      #youtube-bilingual-subtitles .original-subtitle {
        color: #ffffff !important;
        font-size: 18px !important;
        line-height: 1.3 !important;
        margin-bottom: 2px !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8) !important;
      }
      
      #youtube-bilingual-subtitles .translated-subtitle {
        color: #ffeb3b !important;
        font-size: 16px !important;
        line-height: 1.3 !important;
        font-weight: 500 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8) !important;
      }
      
      #youtube-bilingual-subtitles:empty {
        display: none !important;
      }
      
      /* 确保在全屏模式下也能正确显示 */
      .html5-video-player.ytp-fullscreen #youtube-bilingual-subtitles {
        bottom: 60px !important;
      }
    `;
    
    document.head.appendChild(style);
    
    // 插入到播放器中
    const videoContainer = playerContainer.querySelector('.html5-video-container') || playerContainer;
    videoContainer.style.position = 'relative';
    videoContainer.appendChild(this.translationContainer);
    
    this.logger.info('✅ 双语字幕容器已创建');
    
    // 测试显示
    this.updateSubtitleDisplay('测试原文', '测试翻译');
    setTimeout(() => this.updateSubtitleDisplay('', ''), 2000);
  }
  
  startSubtitleMonitoring() {
    // 监听YouTube字幕变化的方法
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
      let observedCount = 0;
      subtitleSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(element => {
          observer.observe(element, {
            childList: true,
            subtree: true,
            characterData: true
          });
          observedCount++;
        });
      });
      
      this.logger.info(`👀 开始观察 ${observedCount} 个字幕元素`);
      
      // 也观察整个document，以防字幕容器动态创建
      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    };
    
    // 立即开始观察
    startObserving();
    
    // 定期检查字幕（备用方案）
    setInterval(() => {
      this.logger.debug('🔄 定期检查字幕...');
      this.checkSubtitleChange();
    }, 2000);
  }
  
  async checkSubtitleChange() {
    if (!this.settings.autoTranslate || !this.settings.apiKey) {
      this.logger.debug('⏸️ 翻译已暂停', {
        autoTranslate: this.settings.autoTranslate,
        hasApiKey: !!this.settings.apiKey
      });
      return;
    }
    
    // 获取当前字幕文本
    const currentSubtitle = this.getCurrentSubtitle();
    
    this.logger.debug('📝 当前检测到的字幕:', currentSubtitle);
    
    if (currentSubtitle && currentSubtitle !== this.currentSubtitleText) {
      this.currentSubtitleText = currentSubtitle;
      this.logger.info('✨ 检测到字幕变化:', currentSubtitle);
      this.showDebugInfo(`检测到字幕: ${currentSubtitle.substring(0, 30)}...`);
      
      // 显示原文
      this.updateSubtitleDisplay(currentSubtitle, '翻译中...');
      
      // 翻译字幕
      await this.translateSubtitle(currentSubtitle);
    } else if (!currentSubtitle && this.currentSubtitleText) {
      // 字幕消失
      this.logger.debug('💨 字幕已消失');
      this.currentSubtitleText = '';
      this.updateSubtitleDisplay('', '');
    }
  }
  
  getCurrentSubtitle() {
    // 尝试多种字幕选择器
    const selectors = [
      '.caption-window .captions-text',
      '.ytp-caption-segment',
      '.caption-window',
      '.html5-captions-text',
      '.ytp-caption-window-container .captions-text'
    ];
    
    for (const selector of selectors) {
      const elements = document.querySelectorAll(selector);
      this.logger.debug(`🔍 尝试选择器 "${selector}": 找到 ${elements.length} 个元素`);
      
      for (const element of elements) {
        if (element && element.textContent && element.textContent.trim()) {
          const text = element.textContent.trim();
          this.logger.info(`✅ 从 "${selector}" 获取到字幕:`, text);
          return text;
        }
      }
    }
    
    this.logger.debug('❌ 未检测到任何字幕');
    return null;
  }
  
  async translateSubtitle(text) {
    if (this.isTranslating) {
      this.logger.debug('🔄 翻译进行中，跳过...');
      return;
    }
    
    // 检查缓存
    if (this.translationCache.has(text)) {
      const translation = this.translationCache.get(text);
      this.logger.info('💾 使用缓存翻译:', translation);
      this.updateSubtitleDisplay(text, translation);
      return;
    }
    
    this.isTranslating = true;
    this.logger.info('🌍 开始翻译:', text);
    this.showDebugInfo(`正在翻译: ${text.substring(0, 30)}...`);
    
    try {
      const translation = await this.callTranslationAPI(text);
      this.logger.info('✅ 翻译完成:', translation);
      
      // 缓存翻译结果
      this.translationCache.set(text, translation);
      
      // 更新显示
      this.updateSubtitleDisplay(text, translation);
      this.showDebugInfo(`翻译完成: ${translation.substring(0, 30)}...`);
      
    } catch (error) {
      this.logger.error('❌ 翻译失败:', error.message);
      this.updateSubtitleDisplay(text, `[翻译失败] ${error.message}`);
      this.showDebugInfo(`翻译失败: ${error.message}`);
    } finally {
      this.isTranslating = false;
    }
  }
  
  async callTranslationAPI(text) {
    this.logger.info('📡 调用翻译API...', {
      url: this.settings.apiUrl + '/chat/completions',
      model: this.settings.model,
      targetLang: this.settings.targetLang
    });
    
    const prompt = this.createTranslationPrompt(text);
    
    const response = await fetch(this.settings.apiUrl + '/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.settings.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.settings.model,
        messages: [
          {
            role: 'system',
            content: prompt
          },
          {
            role: 'user', 
            content: text
          }
        ],
        temperature: 0.7,
        max_tokens: 500
      })
    });
    
    this.logger.info('📡 API响应状态:', response.status + ' ' + response.statusText);
    
    if (!response.ok) {
      const errorText = await response.text();
      this.logger.error('❌ API响应错误:', errorText);
      throw new Error(`API调用失败: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    this.logger.debug('📡 API响应数据:', data);
    
    if (!data.choices || !data.choices[0] || !data.choices[0].message) {
      throw new Error('API返回格式错误');
    }
    
    return data.choices[0].message.content.trim();
  }
  
  createTranslationPrompt(text) {
    // 基于原项目的翻译提示词，简化版本
    const langMap = {
      '简体中文': '简体中文',
      '繁体中文': '繁体中文', 
      '日文': '日文',
      '韩文': '韩文',
      'English': 'English'
    };
    
    const targetLang = langMap[this.settings.targetLang] || this.settings.targetLang;
    
    return `你是一个专业的字幕翻译专家。请将以下英文字幕翻译成${targetLang}，要求：

1. 保持原意准确
2. 使用自然流畅的${targetLang}表达
3. 保持技术术语的准确性
4. 适合字幕显示的简洁表达
5. 直接返回翻译结果，不要解释

标准术语对照：
- AI -> AI
- API -> API
- LLM/Large Language Model -> 大语言模型
- ChatGPT -> ChatGPT
- OpenAI -> OpenAI
- Google -> Google
- YouTube -> YouTube

请翻译以下字幕：`;
  }
  
  updateSubtitleDisplay(original, translated) {
    if (!this.translationContainer) {
      this.logger.warn('❌ 翻译容器不存在');
      return;
    }
    
    const originalElement = this.translationContainer.querySelector('.original-subtitle');
    const translatedElement = this.translationContainer.querySelector('.translated-subtitle');
    
    if (originalElement) {
      originalElement.textContent = original;
    }
    
    if (translatedElement) {
      translatedElement.textContent = translated;
    }
    
    // 显示/隐藏容器
    if (original || translated) {
      this.translationContainer.style.display = 'block';
      this.logger.debug('📺 显示字幕:', { original: original.substring(0, 30) + '...', translated: translated.substring(0, 30) + '...' });
    } else {
      this.translationContainer.style.display = 'none';
      this.logger.debug('🙈 隐藏字幕');
    }
  }
}

// 页面加载完成后初始化
window.debugLogger.info('📄 页面状态:', document.readyState);

const initTranslator = () => {
  window.debugLogger.info('🎬 开始初始化翻译器...');
  new YouTubeSubtitleTranslator();
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initTranslator, 2000);
  });
} else {
  setTimeout(initTranslator, 2000);
}