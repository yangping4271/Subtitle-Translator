// YouTube双语字幕翻译器 - Content Script
class YouTubeSubtitleTranslator {
  constructor() {
    this.backendUrl = 'http://127.0.0.1:9009';
    this.currentJobId = null;
    this.currentVideoId = '';
    this.subtitleSegments = [];
    this.translationContainer = null;
    this.currentSubtitleIndex = -1;
    
    this.init();
  }
  
  async init() {
    console.log('YouTube双语字幕翻译器已启动');
    
    // 添加测试按钮到页面
    this.addTestButton();
    
    // 创建翻译显示容器
    this.createTranslationContainer();
    
    // 监听popup消息
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleMessage(message, sender, sendResponse);
    });
    
    // 监听URL变化
    this.watchVideoChange();
    
    // 开始监听YouTube字幕
    this.startSubtitleMonitoring();
    
    // 监听键盘快捷键
    document.addEventListener('keydown', (event) => {
      // Ctrl+B: 切换调试面板显示/隐藏
      if (event.ctrlKey && event.key === 'b') {
        event.preventDefault();
        this.toggleDebugPanel();
      }
    });
    
    // 检查当前视频是否有缓存的翻译
    await this.checkCurrentVideo();
  }
  
  handleMessage(message, sender, sendResponse) {
    switch (message.type) {
      case 'TRANSLATION_COMPLETED':
        this.currentJobId = message.jobId;
        this.loadTranslationFromBackend();
        break;
    }
  }
  
  watchVideoChange() {
    let currentUrl = window.location.href;
    
    // 监听pushState/replaceState
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;
    
    history.pushState = function() {
      originalPushState.apply(history, arguments);
      setTimeout(() => {
        if (window.location.href !== currentUrl) {
          currentUrl = window.location.href;
          this.onVideoChange();
        }
      }, 100);
    }.bind(this);
    
    history.replaceState = function() {
      originalReplaceState.apply(history, arguments);
      setTimeout(() => {
        if (window.location.href !== currentUrl) {
          currentUrl = window.location.href;
          this.onVideoChange();
        }
      }, 100);
    }.bind(this);
    
    // 监听popstate
    window.addEventListener('popstate', () => {
      setTimeout(() => {
        if (window.location.href !== currentUrl) {
          currentUrl = window.location.href;
          this.onVideoChange();
        }
      }, 100);
    });
  }
  
  async onVideoChange() {
    const videoId = this.extractVideoId(window.location.href);
    if (videoId && videoId !== this.currentVideoId) {
      this.currentVideoId = videoId;
      this.subtitleSegments = [];
      this.currentSubtitleIndex = -1;
      this.updateSubtitleDisplay('', '');
      
      console.log('检测到新视频:', videoId);
      await this.checkCurrentVideo();
    }
  }
  
  async checkCurrentVideo() {
    const videoId = this.extractVideoId(window.location.href);
    if (!videoId) return;
    
    this.currentVideoId = videoId;
    
    try {
      // 检查是否有翻译缓存
      const response = await fetch(`${this.backendUrl}/cache/check/${videoId}`);
      const data = await response.json();
      
      if (data.has_translation_cache) {
        console.log('发现翻译缓存，尝试加载...');
        await this.loadCachedTranslation(videoId);
      }
    } catch (error) {
      console.error('检查视频缓存失败:', error);
    }
  }
  
  async loadCachedTranslation(videoId) {
    try {
      // 从目标语言设置中获取当前语言
      const targetLang = await this.getTargetLanguage();
      
      // 尝试通过视频ID直接获取segments
      const response = await fetch(`${this.backendUrl}/video/${videoId}/segments?target_lang=${targetLang}`);
      const data = await response.json();
      
      if (response.ok && data.segments && data.segments.length > 0) {
        this.subtitleSegments = data.segments;
        console.log(`从缓存加载了 ${data.segments.length} 个字幕段`);
        this.startSubtitleMatching();
        return true;
      } else if (data.active_jobs && data.active_jobs.length > 0) {
        // 有活跃的任务，尝试获取结果
        const activeJob = data.active_jobs.find(job => job.status === 'done');
        if (activeJob) {
          await this.loadTranslationSegments(activeJob.job_id);
          return true;
        }
      }
      
      return false;
    } catch (error) {
      console.error('加载缓存翻译失败:', error);
      return false;
    }
  }
  
  async getTargetLanguage() {
    return new Promise((resolve) => {
      chrome.storage.local.get(['targetLang'], (result) => {
        resolve(result.targetLang || 'zh');
      });
    });
  }
  
  async loadTranslationFromBackend() {
    if (!this.currentJobId) return;
    
    try {
      const response = await fetch(`${this.backendUrl}/segments/${this.currentJobId}`);
      const data = await response.json();
      
      if (response.ok && data.segments) {
        this.subtitleSegments = data.segments;
        console.log(`加载了 ${data.segments.length} 个字幕段`);
        
        // 开始实时匹配字幕
        this.startSubtitleMatching();
      }
    } catch (error) {
      console.error('加载翻译段失败:', error);
    }
  }
  
  async loadTranslationSegments(jobId) {
    try {
      const response = await fetch(`${this.backendUrl}/segments/${jobId}`);
      const data = await response.json();
      
      if (response.ok && data.segments) {
        this.subtitleSegments = data.segments;
        console.log(`从缓存加载了 ${data.segments.length} 个字幕段`);
        this.startSubtitleMatching();
      }
    } catch (error) {
      console.error('加载翻译段失败:', error);
    }
  }
  
  extractVideoId(url) {
    const match = url.match(/[?&]v=([^&]+)/);
    return match ? match[1] : null;
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
    this.addStyles();
    
    // 插入到播放器中
    const videoContainer = playerContainer.querySelector('.html5-video-container') || playerContainer;
    videoContainer.style.position = 'relative';
    videoContainer.appendChild(this.translationContainer);
    
    console.log('双语字幕容器已创建');
  }
  
  addStyles() {
    if (document.getElementById('bilingual-subtitle-styles')) {
      return;
    }
    
    const style = document.createElement('style');
    style.id = 'bilingual-subtitle-styles';
    style.textContent = `
      #youtube-bilingual-subtitles {
        position: absolute;
        bottom: 80px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        pointer-events: none;
        font-family: 'Roboto', 'YouTube Noto', sans-serif;
        max-width: 80%;
      }
      
      #youtube-bilingual-subtitles .subtitle-container {
        background: rgba(0, 0, 0, 0.8);
        border-radius: 4px;
        padding: 8px 12px;
        text-align: center;
        backdrop-filter: blur(4px);
      }
      
      #youtube-bilingual-subtitles .original-subtitle {
        color: #ffffff;
        font-size: 18px;
        line-height: 1.3;
        margin-bottom: 2px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
      }
      
      #youtube-bilingual-subtitles .translated-subtitle {
        color: #ffeb3b;
        font-size: 16px;
        line-height: 1.3;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
      }
      
      #youtube-bilingual-subtitles:empty {
        display: none;
      }
      
      /* 隐藏时的过渡效果 */
      #youtube-bilingual-subtitles .subtitle-container {
        transition: opacity 0.2s ease-in-out;
      }
      
      #youtube-bilingual-subtitles.hidden .subtitle-container {
        opacity: 0;
      }
    `;
    
    document.head.appendChild(style);
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
      '.caption-window',
      '.ytp-caption-segment',
      '.captions-text'
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
  
  checkSubtitleChange() {
    const currentSubtitle = this.getCurrentSubtitle();
    
    if (currentSubtitle) {
      if (this.subtitleSegments.length > 0) {
        // 有翻译数据，进行匹配
        this.matchAndDisplaySubtitle(currentSubtitle);
      } else {
        // 没有翻译数据，只显示原文
        this.updateSubtitleDisplay(currentSubtitle, '');
      }
    } else {
      // 字幕消失
      this.updateSubtitleDisplay('', '');
    }
  }
  
  matchAndDisplaySubtitle(currentSubtitle) {
    // 在翻译段中查找匹配的字幕
    let bestMatch = null;
    let bestScore = 0;
    
    for (let i = 0; i < this.subtitleSegments.length; i++) {
      const segment = this.subtitleSegments[i];
      const score = this.calculateSimilarity(currentSubtitle, segment.text);
      
      if (score > bestScore && score > 0.7) { // 相似度阈值
        bestMatch = segment;
        bestScore = score;
        this.currentSubtitleIndex = i;
      }
    }
    
    if (bestMatch) {
      this.updateSubtitleDisplay(bestMatch.text, bestMatch.translation);
    } else {
      // 没有找到匹配，显示原文
      this.updateSubtitleDisplay(currentSubtitle, '');
    }
  }
  
  calculateSimilarity(str1, str2) {
    // 简单的相似度计算
    if (!str1 || !str2) return 0;
    
    str1 = str1.toLowerCase().trim();
    str2 = str2.toLowerCase().trim();
    
    if (str1 === str2) return 1;
    
    // 计算编辑距离
    const matrix = [];
    const len1 = str1.length;
    const len2 = str2.length;
    
    for (let i = 0; i <= len1; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= len2; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= len1; i++) {
      for (let j = 1; j <= len2; j++) {
        if (str1.charAt(i - 1) === str2.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    const distance = matrix[len1][len2];
    const maxLen = Math.max(len1, len2);
    return 1 - distance / maxLen;
  }
  
  startSubtitleMatching() {
    // console.log('开始字幕匹配模式');
  }
  
  getCurrentSubtitle() {
    // 尝试多种字幕选择器
    const selectors = [
      '.caption-window .captions-text',
      '.ytp-caption-segment', 
      '.caption-window',
      '.html5-captions-text',
      '.ytp-caption-window-container .captions-text',
      '[class*="caption"] [class*="text"]'
    ];
    
    for (const selector of selectors) {
      const elements = document.querySelectorAll(selector);
      for (const element of elements) {
        if (element && element.textContent.trim()) {
          const text = element.textContent.trim();
          // console.log(`[字幕检测] 找到字幕: "${text}" (选择器: ${selector})`);
          return text;
        }
      }
    }
    
    return null;
  }
  
  updateSubtitleDisplay(original, translated) {
    console.log('🔄 更新字幕显示:', { original, translated });
    
    if (!this.translationContainer) {
      console.log('❌ 翻译容器不存在，重新创建');
      this.createTranslationContainer();
      return;
    }
    
    const originalElement = this.translationContainer.querySelector('.original-subtitle');
    const translatedElement = this.translationContainer.querySelector('.translated-subtitle');
    
    console.log('🔍 字幕元素:', { 
      originalElement: !!originalElement, 
      translatedElement: !!translatedElement 
    });
    
    if (originalElement) {
      originalElement.textContent = original;
      console.log('✅ 英文字幕已设置:', original);
    }
    
    if (translatedElement) {
      translatedElement.textContent = translated;
      console.log('✅ 中文字幕已设置:', translated);
    }
    
    // 强制显示容器
    if (original || translated) {
      this.translationContainer.style.display = 'block';
      this.translationContainer.style.visibility = 'visible';
      this.translationContainer.style.opacity = '1';
      this.translationContainer.style.position = 'fixed';
      this.translationContainer.style.bottom = '100px';
      this.translationContainer.style.left = '50%';
      this.translationContainer.style.transform = 'translateX(-50%)';
      this.translationContainer.style.zIndex = '999999999';
      this.translationContainer.classList.remove('hidden');
      this.translationContainer.classList.add('force-show');
      
      console.log('✅ 容器强制显示');
      
      // 验证容器在DOM中
      const containerInDOM = document.getElementById('youtube-bilingual-subtitles');
      console.log('🔍 容器在DOM中:', !!containerInDOM);
      
    } else {
      this.translationContainer.style.display = 'none';
      this.translationContainer.classList.add('hidden');
      this.translationContainer.classList.remove('force-show');
      console.log('❌ 容器已隐藏');
    }
  }
  
  addTestButton() {
    // 添加测试按钮
    const testButton = document.createElement('button');
    testButton.textContent = '测试双语字幕';
    testButton.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 999999999;
      background: #ff0000;
      color: white;
      padding: 10px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      font-size: 14px;
    `;
    
    testButton.addEventListener('click', () => {
      console.log('🧪 测试按钮被点击');
      this.testSubtitleDisplay();
    });
    
    // 添加调试面板按钮
    const debugButton = document.createElement('button');
    debugButton.textContent = '显示调试面板';
    debugButton.style.cssText = `
      position: fixed;
      top: 60px;
      right: 10px;
      z-index: 999999999;
      background: #0066cc;
      color: white;
      padding: 10px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      font-size: 14px;
    `;
    
    debugButton.addEventListener('click', () => {
      console.log('🔧 调试面板按钮被点击');
      this.showDebugPanel();
    });
    
    document.body.appendChild(testButton);
    document.body.appendChild(debugButton);
    console.log('🧪 测试按钮和调试按钮已添加');
  }
  
  testSubtitleDisplay() {
    console.log('🧪 开始测试字幕显示');
    
    // 强制显示测试字幕
    this.updateSubtitleDisplay(
      'This is a test subtitle in English',
      '这是一个测试字幕中文翻译'
    );
    
    // 5秒后清除
    setTimeout(() => {
      this.updateSubtitleDisplay('', '');
      console.log('🧪 测试字幕已清除');
    }, 5000);
  }
  
  showDebugPanel() {
    console.log('🔧 显示调试面板');
    
    // 移除已存在的调试面板
    const existingPanel = document.getElementById('subtitle-debug-panel');
    if (existingPanel) {
      existingPanel.remove();
    }
    
    // 创建调试面板
    const debugPanel = document.createElement('div');
    debugPanel.id = 'subtitle-debug-panel';
    debugPanel.style.cssText = `
      position: fixed;
      top: 120px;
      right: 10px;
      width: 400px;
      max-height: 500px;
      background: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 15px;
      border-radius: 8px;
      z-index: 999999999;
      font-size: 12px;
      font-family: monospace;
      overflow-y: auto;
      border: 2px solid #0066cc;
    `;
    
    // 收集调试信息
    const debugInfo = this.getDebugInfo();
    
    debugPanel.innerHTML = `
      <h3 style="margin: 0 0 10px 0; color: #00ff00;">🔧 Chrome扩展调试面板</h3>
      <div style="margin-bottom: 10px;">
        <strong>扩展状态:</strong> ${debugInfo.extensionStatus}
      </div>
      <div style="margin-bottom: 10px;">
        <strong>视频ID:</strong> ${debugInfo.videoId || '未检测'}
      </div>
      <div style="margin-bottom: 10px;">
        <strong>翻译容器:</strong> ${debugInfo.containerStatus}
      </div>
      <div style="margin-bottom: 10px;">
        <strong>字幕段数:</strong> ${debugInfo.segmentCount}
      </div>
      <div style="margin-bottom: 10px;">
        <strong>当前字幕:</strong> ${debugInfo.currentSubtitle || '无'}
      </div>
      <div style="margin-bottom: 10px;">
        <strong>后台连接:</strong> ${debugInfo.backendStatus}
      </div>
      <div style="margin-bottom: 10px;">
        <strong>页面URL:</strong> ${window.location.href}
      </div>
      <button id="refresh-debug" style="background: #ff6600; color: white; padding: 5px 10px; border: none; border-radius: 3px; cursor: pointer; margin-right: 10px;">刷新</button>
      <button id="close-debug" style="background: #cc0000; color: white; padding: 5px 10px; border: none; border-radius: 3px; cursor: pointer;">关闭</button>
    `;
    
    // 添加按钮事件
    debugPanel.querySelector('#refresh-debug').addEventListener('click', () => {
      this.showDebugPanel(); // 重新显示面板
    });
    
    debugPanel.querySelector('#close-debug').addEventListener('click', () => {
      debugPanel.remove();
    });
    
    document.body.appendChild(debugPanel);
    console.log('🔧 调试面板已显示');
  }
  
  getDebugInfo() {
    return {
      extensionStatus: '正常运行',
      videoId: this.currentVideoId,
      containerStatus: this.translationContainer ? 
        `已创建 (${this.translationContainer.style.display})` : '未创建',
      segmentCount: this.subtitleSegments ? this.subtitleSegments.length : 0,
      currentSubtitle: this.currentSubtitleIndex >= 0 && this.subtitleSegments ? 
        this.subtitleSegments[this.currentSubtitleIndex]?.text?.substring(0, 50) + '...' : '无',
      backendStatus: this.backendUrl ? '已配置' : '未配置'
    };
  }
  
  toggleDebugPanel() {
    console.log('🔧 切换调试面板');
    const existingPanel = document.getElementById('subtitle-debug-panel');
    if (existingPanel) {
      existingPanel.remove();
      console.log('❌ 调试面板已隐藏');
    } else {
      this.showDebugPanel();
    }
  }
}

// 页面加载完成后初始化
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('🎬 DOM加载完成，Chrome扩展正在初始化...');
    setTimeout(() => new YouTubeSubtitleTranslator(), 2000);
  });
} else {
  console.log('🎬 DOM已就绪，Chrome扩展正在初始化...');
  setTimeout(() => new YouTubeSubtitleTranslator(), 2000);
}