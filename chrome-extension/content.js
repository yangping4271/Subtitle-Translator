// YouTube双语字幕翻译器 - Content Script
class YouTubeSubtitleTranslator {
  constructor() {
    this.settings = {
      apiUrl: 'https://api.openai.com/v1',
      apiKey: '',
      targetLang: '简体中文',
      model: 'gpt-4o-min',
      autoTranslate: true
    };
    
    this.currentSubtitleText = '';
    this.translationCache = new Map();
    this.translationContainer = null;
    this.isTranslating = false;
    
    this.init();
  }
  
  async init() {
    // 加载设置
    await this.loadSettings();
    
    // 创建翻译显示容器
    this.createTranslationContainer();
    
    // 开始监听字幕变化
    this.startSubtitleMonitoring();
    
    // 监听设置更新
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.type === 'SETTINGS_UPDATED') {
        this.settings = message.settings;
        console.log('设置已更新:', this.settings);
      }
    });
    
    console.log('YouTube双语字幕翻译器已启动');
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
    const playerContainer = document.querySelector('#movie_player') || document.querySelector('.html5-video-player');
    if (!playerContainer) {
      setTimeout(() => this.createTranslationContainer(), 1000);
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
        position: absolute;
        bottom: 80px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        pointer-events: none;
        font-family: 'Roboto', 'YouTube Noto', sans-serif;
      }
      
      #youtube-bilingual-subtitles .subtitle-container {
        background: rgba(0, 0, 0, 0.8);
        border-radius: 4px;
        padding: 8px 12px;
        max-width: 80vw;
        text-align: center;
      }
      
      #youtube-bilingual-subtitles .original-subtitle {
        color: #ffffff;
        font-size: 18px;
        line-height: 1.3;
        margin-bottom: 2px;
      }
      
      #youtube-bilingual-subtitles .translated-subtitle {
        color: #ffeb3b;
        font-size: 16px;
        line-height: 1.3;
        font-weight: 500;
      }
      
      #youtube-bilingual-subtitles:empty {
        display: none;
      }
    `;
    
    document.head.appendChild(style);
    
    // 插入到播放器中
    const videoContainer = playerContainer.querySelector('.html5-video-container') || playerContainer;
    videoContainer.style.position = 'relative';
    videoContainer.appendChild(this.translationContainer);
    
    console.log('双语字幕容器已创建');
  }
  
  startSubtitleMonitoring() {
    // 监听YouTube字幕变化的方法
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList' || mutation.type === 'characterData') {
          this.checkSubtitleChange();
        }
      });
    });
    
    // 观察字幕容器
    const subtitleSelectors = [
      '.caption-window',  // 新版YouTube
      '.ytp-caption-segment', // 旧版YouTube
      '.captions-text'  // 备用
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
      
      // 也观察整个document，以防字幕容器动态创建
      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    };
    
    // 立即开始观察
    startObserving();
    
    // 定期检查字幕（备用方案）
    setInterval(() => this.checkSubtitleChange(), 1000);
  }
  
  async checkSubtitleChange() {
    if (!this.settings.autoTranslate || !this.settings.apiKey) {
      return;
    }
    
    // 获取当前字幕文本
    const currentSubtitle = this.getCurrentSubtitle();
    
    if (currentSubtitle && currentSubtitle !== this.currentSubtitleText) {
      this.currentSubtitleText = currentSubtitle;
      console.log('检测到字幕变化:', currentSubtitle);
      
      // 显示原文
      this.updateSubtitleDisplay(currentSubtitle, '');
      
      // 翻译字幕
      await this.translateSubtitle(currentSubtitle);
    } else if (!currentSubtitle && this.currentSubtitleText) {
      // 字幕消失
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
      '.html5-captions-text'
    ];
    
    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (element && element.textContent.trim()) {
        return element.textContent.trim();
      }
    }
    
    return null;
  }
  
  async translateSubtitle(text) {
    if (this.isTranslating) {
      return;
    }
    
    // 检查缓存
    if (this.translationCache.has(text)) {
      const translation = this.translationCache.get(text);
      this.updateSubtitleDisplay(text, translation);
      return;
    }
    
    this.isTranslating = true;
    
    try {
      const translation = await this.callTranslationAPI(text);
      
      // 缓存翻译结果
      this.translationCache.set(text, translation);
      
      // 更新显示
      this.updateSubtitleDisplay(text, translation);
      
    } catch (error) {
      console.error('翻译失败:', error);
      this.updateSubtitleDisplay(text, `[翻译失败] ${error.message}`);
    } finally {
      this.isTranslating = false;
    }
  }
  
  async callTranslationAPI(text) {
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
    
    if (!response.ok) {
      throw new Error(`API调用失败: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    
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
    } else {
      this.translationContainer.style.display = 'none';
    }
  }
}

// 页面加载完成后初始化
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => new YouTubeSubtitleTranslator(), 2000);
  });
} else {
  setTimeout(() => new YouTubeSubtitleTranslator(), 2000);
}