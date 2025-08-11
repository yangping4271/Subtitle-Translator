// YouTube双语字幕翻译器 - 预加载版
class YouTubeSubtitleTranslator {
  constructor() {
    this.initLogger();
    this.logger.info('🚀 YouTube双语字幕翻译器启动...');
    
    this.settings = {
      apiUrl: 'https://api.openai.com/v1',
      apiKey: '',
      targetLang: '简体中文',
      model: 'gpt-4o-mini',
      autoTranslate: true,
      usePreload: true // 启用预加载模式
    };
    
    this.currentVideoId = null;
    this.currentSubtitleText = '';
    this.translationContainer = null;
    
    // 预加载相关
    this.subtitleFetcher = new YouTubeSubtitleFetcher();
    this.translationProcessor = new SmartTranslationProcessor(this.settings, this.logger);
    this.preloadedTranslations = null;
    this.isPreloading = false;
    
    // 状态追踪变量
    this.splitSegmentCount = 0;
    this.hasSummary = false;
    this.processingProgress = 0;
    
    // 实时翻译相关（备用模式）
    this.translationCache = new Map();
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
          // 收集系统信息
          const systemInfo = {
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href,
            viewport: `${window.innerWidth}x${window.innerHeight}`,
            language: navigator.language
          };
          
          const logText = [
            '=== YouTube双语字幕翻译器 - 调试日志 ===',
            `导出时间: ${systemInfo.timestamp}`,
            `当前页面: ${systemInfo.url}`,
            `浏览器: ${systemInfo.userAgent}`,
            `视口大小: ${systemInfo.viewport}`,
            `语言: ${systemInfo.language}`,
            '',
            '=== 应用日志 ===',
            this.logs.map(log => 
              `[${log.timestamp}] [${log.level}] ${log.message}${log.data ? '\n' + JSON.stringify(log.data, null, 2) : ''}`
            ).join('\n\n'),
            '',
            '=== 系统状态 ===',
            `总日志条数: ${this.logs.length}`,
            `错误日志: ${this.logs.filter(log => log.level === 'ERROR').length}条`,
            `警告日志: ${this.logs.filter(log => log.level === 'WARN').length}条`,
            `调试日志: ${this.logs.filter(log => log.level === 'DEBUG').length}条`,
            '',
            '=== 扩展状态 ===',
            `Chrome版本: ${navigator.appVersion}`,
            `内存使用: ${performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1024 / 1024) + 'MB' : '未知'}`,
            ''
          ].join('\n');
          
          const blob = new Blob([logText], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `youtube-subtitle-debug-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          
          // 记录导出操作
          this.info(`📥 调试日志已导出: ${this.logs.length}条日志, ${this.logs.filter(log => log.level === 'ERROR').length}个错误`);
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
    this.setupMessageListener();
    this.setupVideoChangeDetection();
    
    // 设置翻译进度回调
    this.translationProcessor.setProgressCallback((progress) => {
      this.processingProgress = progress;
      this.showStatusInfo(`正在预翻译: ${progress.toFixed(1)}%`);
    });
    
    this.logger.info('✅ 初始化完成');
    this.showStatusInfo('插件已启动，等待检测视频...');
    
    // 设置全局快捷键
    this.setupGlobalKeyboardShortcuts();
    
    // 立即检查当前视频
    this.checkVideoChange();
  }
  
  async loadSettings() {
    return new Promise((resolve) => {
      chrome.storage.sync.get(['apiUrl', 'apiKey', 'targetLang', 'model', 'autoTranslate', 'usePreload'], (result) => {
        this.settings = {
          apiUrl: result.apiUrl || this.settings.apiUrl,
          apiKey: result.apiKey || '',
          targetLang: result.targetLang || this.settings.targetLang,
          model: result.model || this.settings.model,
          autoTranslate: result.autoTranslate !== false,
          usePreload: result.usePreload !== false  // 确保预加载模式默认启用
        };
        resolve();
      });
    });
  }
  
  // 设置全局快捷键
  setupGlobalKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
      // Ctrl+L: 导出日志
      if (event.ctrlKey && event.key === 'l') {
        event.preventDefault();
        this.logger.exportLogs();
        this.logger.info('🔄 通过快捷键导出调试日志');
        this.showTemporaryMessage('调试日志已导出到下载文件夹！');
      }
      
      // Ctrl+D: 切换调试面板显示/隐藏
      if (event.ctrlKey && event.key === 'd') {
        event.preventDefault();
        this.toggleDebugPanel();
      }
    });
    
    this.logger.info('⌨️ 全局快捷键已设置: Ctrl+L=导出日志, Ctrl+D=切换调试面板');
  }
  
  // 显示临时消息
  showTemporaryMessage(message, duration = 3000) {
    const messageDiv = document.createElement('div');
    messageDiv.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(76, 175, 80, 0.9);
      color: white;
      padding: 12px 20px;
      border-radius: 8px;
      z-index: 10001;
      font-size: 14px;
      font-family: Arial, sans-serif;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;
    messageDiv.textContent = message;
    document.body.appendChild(messageDiv);
    
    setTimeout(() => {
      if (messageDiv.parentNode) {
        messageDiv.parentNode.removeChild(messageDiv);
      }
    }, duration);
  }
  
  // 切换调试面板显示/隐藏
  toggleDebugPanel() {
    const statusDiv = document.getElementById('subtitle-status-info');
    if (statusDiv) {
      const isVisible = statusDiv.style.display !== 'none';
      statusDiv.style.display = isVisible ? 'none' : 'block';
      this.logger.info(`🔧 调试面板${isVisible ? '隐藏' : '显示'}`);
      
      if (!isVisible) {
        // 重新显示时更新状态
        this.showStatusInfo('调试面板已重新显示');
      }
    }
  }
  
  // 设置视频变化检测
  setupVideoChangeDetection() {
    // 监听URL变化
    let currentUrl = window.location.href;
    
    const checkUrlChange = () => {
      if (window.location.href !== currentUrl) {
        currentUrl = window.location.href;
        this.checkVideoChange();
      }
    };
    
    // 监听History API变化
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;
    
    history.pushState = function(...args) {
      originalPushState.apply(history, args);
      setTimeout(checkUrlChange, 100);
    };
    
    history.replaceState = function(...args) {
      originalReplaceState.apply(history, args);
      setTimeout(checkUrlChange, 100);
    };
    
    // 监听popstate事件
    window.addEventListener('popstate', checkUrlChange);
    
    // 定期检查视频变化
    setInterval(() => {
      this.checkVideoChange();
    }, 3000);
    
    this.logger.info('👀 视频变化检测已启动');
  }

  // 检查视频变化
  async checkVideoChange() {
    const hasVideoChanged = this.subtitleFetcher.getCurrentVideoId();
    
    if (hasVideoChanged) {
      this.currentVideoId = this.subtitleFetcher.videoId;
      this.logger.info('🎬 检测到新视频:', this.currentVideoId);
      
      // 清理之前的数据
      this.clearPreviousData();
      
      // 调试设置状态
      this.logger.info('🔧 当前设置:', {
        usePreload: this.settings.usePreload,
        autoTranslate: this.settings.autoTranslate,
        apiKey: this.settings.apiKey ? '***已设置***' : '未设置'
      });
      
      // 如果启用预加载模式，开始预加载字幕
      if (this.settings.usePreload && this.settings.autoTranslate) {
        this.logger.info('🚀 启动预加载翻译模式...');
        this.startSubtitlePreloading();
      } else {
        this.logger.info('🔄 启动实时翻译模式...', {
          reason: !this.settings.usePreload ? '预加载已禁用' : '自动翻译已禁用'
        });
        // 否则启用实时翻译模式
        this.startRealtimeMode();
      }
    }
  }

  // 清理之前的数据
  clearPreviousData() {
    this.preloadedTranslations = null;
    this.isPreloading = false;
    this.currentSubtitleText = '';
    this.translationCache.clear();
    this.subtitleFetcher.clearSubtitles();
    this.updateSubtitleDisplay('', '');
    
    // 重置状态追踪变量
    this.splitSegmentCount = 0;
    this.hasSummary = false;
    this.processingProgress = 0;
    
    this.logger.info('🧹 已清理之前数据和状态');
  }

  // 开始字幕预加载
  async startSubtitlePreloading() {
    if (this.isPreloading) return;
    
    this.isPreloading = true;
    this.showStatusInfo('正在获取字幕数据...');
    
    try {
      // 等待页面稳定
      await this.waitForPageStable();
      
      // 使用增强的字幕获取方法
      this.showStatusInfo('尝试多种方法获取完整字幕...');
      this.logger.info('🚀 开始预加载字幕，尝试所有可用方法...');
      
      const subtitles = await this.subtitleFetcher.getSubtitlesWithEnhancedMethods();
      
      if (!subtitles || subtitles.length === 0) {
        // 提供详细的失败原因
        const failureReason = this.analyzeSubtitleFailure();
        this.logger.warn('❌ 字幕获取失败，原因:', failureReason);
        this.showStatusInfo(`字幕获取失败: ${failureReason}`);
        throw new Error(`所有字幕获取方法都失败 - ${failureReason}`);
      }
      
      const stats = this.subtitleFetcher.getSubtitleStats();
      this.logger.info('📊 字幕统计:', stats);
      this.showStatusInfo(`字幕已获取: ${stats.count}条，开始预翻译...`);
      
      // 开始智能翻译处理
      await this.processPreloadedTranslation(subtitles);
      
    } catch (error) {
      this.logger.error('❌ 预加载失败:', error.message);
      this.showStatusInfo(`预加载失败: ${error.message}`);
      
      // 回退到实时翻译模式
      this.startRealtimeMode();
    } finally {
      this.isPreloading = false;
    }
  }

  // 处理预加载翻译
  async processPreloadedTranslation(subtitles) {
    try {
      // 更新设置到处理器
      this.translationProcessor.settings = this.settings;
      
      // 准备字幕文本数据
      const subtitleTexts = this.subtitleFetcher.getAllSubtitleTexts();
      this.logger.info('📝 获取字幕文本数据:', subtitleTexts.length + '条');

      // 智能断句
      this.showStatusInfo('正在智能断句...');
      const segments = this.translationProcessor.smartSplit(subtitleTexts);
      this.splitSegmentCount = segments.length;
      this.logger.info('✅ 断句完成:', this.splitSegmentCount + '段');
      
      // 生成内容总结
      this.showStatusInfo('正在分析内容...');
      const summary = await this.translationProcessor.generateContentSummary(segments);
      this.hasSummary = true;
      this.logger.info('✅ 内容总结完成');
      
      // 分批翻译
      this.showStatusInfo('开始批量翻译...');
      this.preloadedTranslations = await this.translationProcessor.processBatchTranslation(segments, summary);
      
      this.logger.info('✅ 预翻译完成:', this.preloadedTranslations.length + '条');
      this.showStatusInfo('预翻译完成！开始实时显示...');
      
      // 启动实时显示监听
      this.startRealtimeDisplay();
      
    } catch (error) {
      this.logger.error('❌ 预翻译处理失败:', error.message);
      throw error;
    }
  }

  // 等待页面稳定
  waitForPageStable() {
    return new Promise((resolve) => {
      let stabilityCount = 0;
      let checkCount = 0;
      const maxChecks = 20; // 最多检查10秒
      
      this.logger.info('⏳ 等待页面稳定...');
      
      const checkStability = () => {
        checkCount++;
        
        const hasPlayer = !!document.querySelector('#movie_player, .html5-video-player');
        const hasScripts = document.querySelectorAll('script').length > 10;
        const hasVideoId = !!this.subtitleFetcher.videoId;
        const hasPlayerConfig = !!window.ytInitialPlayerResponse || 
                               !!document.querySelector('script[src*="player"]') ||
                               document.querySelectorAll('script').length > 50;
        
        this.logger.debug(`📊 稳定性检查 ${checkCount}/${maxChecks}:`, {
          hasPlayer,
          hasScripts,
          hasVideoId,
          hasPlayerConfig,
          stabilityCount
        });
        
        if (hasPlayer && hasScripts && hasVideoId && hasPlayerConfig) {
          stabilityCount++;
          if (stabilityCount >= 3) {
            this.logger.info('✅ 页面稳定性检查通过');
            resolve();
            return;
          }
        } else {
          stabilityCount = 0;
        }
        
        if (checkCount >= maxChecks) {
          this.logger.warn('⚠️ 页面稳定性检查超时，继续处理');
          resolve();
          return;
        }
        
        setTimeout(checkStability, 500);
      };
      
      setTimeout(checkStability, 1000);
    });
  }

  // 启动实时显示监听
  startRealtimeDisplay() {
    if (this.displayInterval) {
      clearInterval(this.displayInterval);
    }
    
    // 每秒检查当前播放时间并显示对应字幕
    this.displayInterval = setInterval(() => {
      this.updateDisplayFromPreloaded();
    }, 500);
    
    this.logger.info('⏰ 实时显示监听已启动');
  }

  // 从预加载数据更新显示
  updateDisplayFromPreloaded() {
    if (!this.preloadedTranslations || this.preloadedTranslations.length === 0) {
      return;
    }
    
    // 获取当前播放时间
    const currentTime = this.getCurrentPlayTime();
    if (currentTime === null) return;
    
    // 找到当前时间对应的字幕
    const currentSubtitle = this.findSubtitleAtTime(currentTime);
    
    if (currentSubtitle) {
      const text = `${currentSubtitle.text}`;
      if (text !== this.currentSubtitleText) {
        this.currentSubtitleText = text;
        this.updateSubtitleDisplay(currentSubtitle.text, currentSubtitle.translation);
      }
    } else {
      // 没有找到对应字幕，清空显示
      if (this.currentSubtitleText) {
        this.currentSubtitleText = '';
        this.updateSubtitleDisplay('', '');
      }
    }
  }

  // 获取当前播放时间
  getCurrentPlayTime() {
    const videoElement = document.querySelector('video');
    if (videoElement && !videoElement.paused) {
      return videoElement.currentTime;
    }
    return null;
  }

  // 在预加载翻译中查找指定时间的字幕
  findSubtitleAtTime(currentTime) {
    for (const subtitle of this.preloadedTranslations) {
      if (currentTime >= subtitle.startTime && currentTime <= subtitle.endTime) {
        return subtitle;
      }
    }
    return null;
  }

  // 启动实时翻译模式（备用）
  startRealtimeMode() {
    this.logger.info('🔄 启动实时翻译模式');
    this.showStatusInfo('实时翻译模式已启动');
    this.startSubtitleMonitoring();
  }
  
  setupMessageListener() {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.type === 'SETTINGS_UPDATED') {
        this.settings = message.settings;
        this.logger.info('🔄 设置已更新');
        
        // 更新翻译处理器设置
        if (this.translationProcessor) {
          this.translationProcessor.settings = this.settings;
        }
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
        background: rgba(0,0,0,0.9);
        color: #fff;
        padding: 10px 15px;
        border-radius: 8px;
        z-index: 10000;
        font-size: 11px;
        max-width: 350px;
        font-family: monospace;
        border: 1px solid #333;
      `;
      document.body.appendChild(statusDiv);
    }
    
    // 收集详细状态信息
    const debugInfo = this.getDebugStatus();
    
    statusDiv.innerHTML = `
      <div style="color: #4CAF50; font-weight: bold; margin-bottom: 8px;">🔧 字幕翻译调试状态</div>
      <div style="margin-bottom: 5px;"><strong>当前状态:</strong> ${message}</div>
      <div style="margin-bottom: 5px;"><strong>翻译模式:</strong> ${debugInfo.mode}</div>
      <div style="margin-bottom: 5px;"><strong>视频ID:</strong> ${debugInfo.videoId || '未检测'}</div>
      <div style="margin-bottom: 5px;"><strong>字幕获取:</strong> ${debugInfo.subtitleStatus}</div>
      <div style="margin-bottom: 5px;"><strong>断句状态:</strong> ${debugInfo.splitStatus}</div>
      <div style="margin-bottom: 5px;"><strong>总结分析:</strong> ${debugInfo.summaryStatus}</div>
      <div style="margin-bottom: 5px;"><strong>翻译进度:</strong> ${debugInfo.translationProgress}</div>
      <div style="margin-bottom: 5px;"><strong>预翻译:</strong> ${debugInfo.preloadStatus}</div>
      <div style="margin-bottom: 5px;"><strong>实时显示:</strong> ${debugInfo.displayStatus}</div>
      <div style="margin-bottom: 5px; ${debugInfo.errorCount > 0 ? 'color: #FF5722;' : ''}">
        <strong>错误统计:</strong> ${debugInfo.errorCount}个错误, ${debugInfo.warnCount}个警告
      </div>
      <div style="margin-top: 8px; border-top: 1px solid #444; padding-top: 5px;">
        <button id="export-logs-btn" style="
          background: #4CAF50; 
          color: white; 
          border: none; 
          padding: 4px 8px; 
          border-radius: 4px; 
          font-size: 9px; 
          cursor: pointer;
          margin-right: 8px;
        ">导出日志 (Ctrl+L)</button>
        <span style="font-size: 9px; opacity: 0.6;">${new Date().toLocaleTimeString()}</span>
      </div>
    `;
    
    // 添加导出按钮事件监听
    const exportBtn = statusDiv.querySelector('#export-logs-btn');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => {
        this.logger.exportLogs();
        exportBtn.textContent = '已导出 ✓';
        exportBtn.style.background = '#2196F3';
        setTimeout(() => {
          exportBtn.textContent = '导出日志 (Ctrl+L)';
          exportBtn.style.background = '#4CAF50';
        }, 2000);
      });
    }
  }

  // 获取详细调试状态
  getDebugStatus() {
    const hasSubtitles = this.subtitleFetcher?.hasFullSubtitlesEnhanced();
    const subtitleCount = hasSubtitles ? this.subtitleFetcher.fullSubtitles.length : 0;
    const preloadedCount = this.preloadedTranslations?.length || 0;
    
    // 统计日志中的错误和警告
    const errorCount = this.logger.logs.filter(log => log.level === 'ERROR').length;
    const warnCount = this.logger.logs.filter(log => log.level === 'WARN').length;
    
    return {
      mode: this.settings.usePreload ? '预加载模式 (增强版)' : '实时翻译模式',
      videoId: this.currentVideoId || '未检测',
      subtitleStatus: hasSubtitles ? `✅ 已获取 (${subtitleCount}条)` : '❌ 未获取',
      splitStatus: this.splitSegmentCount ? `✅ 已处理 (${this.splitSegmentCount}段)` : '⏳ 待处理',
      summaryStatus: this.hasSummary ? '✅ 已完成' : '⏳ 待处理',
      translationProgress: this.isPreloading ? `🔄 ${this.processingProgress?.toFixed(1) || 0}%` : 
                          (preloadedCount > 0 ? `✅ 完成 (${preloadedCount}条)` : '⏳ 待开始'),
      preloadStatus: this.preloadedTranslations ? 
                    `✅ 完成 (${preloadedCount}条翻译)` : 
                    (this.isPreloading ? '🔄 处理中' : '❌ 未完成'),
      displayStatus: this.displayInterval ? '✅ 运行中' : '❌ 未启动',
      errorCount: errorCount,
      warnCount: warnCount
    };
  }

  // 分析字幕获取失败的原因
  analyzeSubtitleFailure() {
    // 检查视频是否有字幕按钮
    const subtitleButton = document.querySelector('.ytp-subtitles-button, button[data-tooltip-target-id*="caption"]');
    if (subtitleButton) {
      const isDisabled = subtitleButton.hasAttribute('disabled') || 
                        subtitleButton.getAttribute('aria-pressed') === 'false' ||
                        subtitleButton.textContent.includes('unavailable');
      
      if (isDisabled) {
        return '视频没有提供字幕数据';
      }
    }
    
    // 检查网络连接
    if (!navigator.onLine) {
      return '网络连接问题';
    }
    
    // 检查是否是受限视频
    const errorElements = document.querySelectorAll('[class*="error"], [class*="unavailable"]');
    if (errorElements.length > 0) {
      return '视频不可用或受限';
    }
    
    // 检查是否是直播
    if (document.querySelector('.ytp-live')) {
      return '直播视频可能没有预生成字幕';
    }
    
    return '未知原因，可能是技术限制或YouTube API变化';
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