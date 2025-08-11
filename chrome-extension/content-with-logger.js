// YouTube双语字幕翻译器 - 预加载版
class YouTubeSubtitleTranslator {
  constructor() {
    this.initLogger();
    this.logger.info('🚀 YouTube双语字幕翻译器启动...');
    
    // 检查依赖是否可用
    if (typeof SmartTranslationProcessor === 'undefined') {
      console.error('❌ SmartTranslationProcessor 未定义，请检查 translation-processor.js 是否加载');
      this.logger.error('❌ SmartTranslationProcessor 未定义');
      return;
    }
    
    if (typeof SRTParser === 'undefined') {
      console.error('❌ SRTParser 未定义，请检查 srt-parser.js 是否加载');
      this.logger.error('❌ SRTParser 未定义');
      return;
    }
    
    console.log('✅ 所有依赖已加载');
    
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
    // 移除 YouTube 内置字幕获取，改为后端转录方案
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
    this.setupBackendTranscribe();
    
    // 设置翻译进度回调
    this.translationProcessor.setProgressCallback((progress) => {
      this.processingProgress = progress;
      this.showStatusInfo(`正在预翻译: ${progress.toFixed(1)}%`);
    });
    
            this.logger.info('✅ 初始化完成');
        // 显示简洁的启动提示
        this.showTemporaryMessage('🌐 YouTube双语字幕已启动 (Ctrl+B显示详细状态)', 3000);
    
    // 设置全局快捷键
    this.setupGlobalKeyboardShortcuts();
    
    // 立即检查当前视频
    this.checkVideoChange();
    
    // 处理页面刷新后的初始化延迟
    setTimeout(() => {
      this.ensureProperInitialization();
    }, 1000);
  }

  // 确保正确初始化（处理页面刷新等边缘情况）
  ensureProperInitialization() {
    // 检查容器是否存在且位置正确
    if (!this.translationContainer || !document.getElementById('youtube-bilingual-subtitles')) {
      this.logger.debug('🔄 检测到容器丢失，重新创建');
      this.createTranslationContainer();
      return;
    }
    
    // 检查位置管理器是否正常工作
    if (!this.positionManager) {
      this.logger.debug('🔄 检测到位置管理器丢失，重新初始化');
      this.initializeSmartPositioning();
      return;
    }
    
    // 强制更新位置以处理可能的布局问题
    this.positionManager.updatePosition();
    this.logger.debug('✅ 初始化检查完成');
  }

  setupBackendTranscribe() {
    // 后端配置（可改造成从 storage 读取）
    this.backend = {
      baseUrl: this.settings.transcribeBaseUrl || 'http://127.0.0.1:9009',
      alwaysTranscribe: true,
      prefetchAheadSec: 30
    };
    
    // 测试后端连接
    this.testBackendConnection();
  }
  
  async testBackendConnection() {
    try {
      const url = `${this.backend.baseUrl}/health`;
      const useProxy = this.needsProxy(url);
      const resp = useProxy ? await this.proxyFetch(url, { method: 'GET' }) : await fetch(url);
      
      if (resp.ok) {
        const health = await resp.json();
        this.logger.info('✅ 后端连接正常', { 
          health,
          baseUrl: this.backend.baseUrl 
        });
        this.showStatusInfo(`后端服务正常 (Python ${health.python || 'unknown'})`);
      } else {
        throw new Error(`HTTP ${resp.status}`);
      }
    } catch (e) {
      this.logger.error('❌ 后端连接失败', { 
        error: e.message,
        baseUrl: this.backend.baseUrl 
      });
      this.showStatusInfo(`后端连接失败: ${e.message}`);
    }
  }
  
  async testBackendDownload() {
    try {
      this.logger.info('🧪 测试后端下载功能...');
      const url = `${this.backend.baseUrl}/test_download`;
      const useProxy = this.needsProxy(url);
      const resp = useProxy ? await this.proxyFetch(url, { method: 'POST' }) : await fetch(url, { method: 'POST' });
      
      if (resp.ok) {
        const result = await resp.json();
        this.logger.info('📋 下载测试结果', result);
        
        if (result.success) {
          this.logger.info('✅ 后端下载功能正常');
        } else {
          this.logger.warn('⚠️ 后端下载测试失败', { 
            returncode: result.returncode,
            stderr: result.stderr 
          });
        }
      }
    } catch (e) {
      this.logger.debug('下载测试失败 (这是正常的):', e.message);
    }
  }

  // 判断是否需要通过后台代理（避免CORS）
  needsProxy(url) {
    try {
      const u = new URL(url);
      const host = u.hostname;
      // 只有远程API主机需要通过后台代理，本地服务直接访问
      return host === 'ai-proxy.chatwise.app' || host === 'openrouter.ai';
    } catch { return false; }
  }

  // 通过后台代理执行跨域请求
  async proxyFetch(url, options) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ type: 'PROXY_FETCH', url, options }, (resp) => {
        if (!resp) return reject(new Error('代理请求失败'));
        if (!resp.ok) return reject(new Error(`代理响应失败: ${resp.status}${resp.error ? ' ' + resp.error : ''}`));
        
        // 创建标准化的Response接口
        const mockResponse = {
          ok: resp.ok,
          status: resp.status,
          headers: resp.headers,
          json: async () => {
            try {
              return JSON.parse(resp.bodyText);
            } catch (e) {
              throw new Error(`JSON解析失败: ${e.message}`);
            }
          },
          text: async () => resp.bodyText
        };
        resolve(mockResponse);
      });
    });
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
      
      // Ctrl+B: 切换调试面板显示/隐藏
      if (event.ctrlKey && event.key === 'b') {
        event.preventDefault();
        this.toggleDebugPanel();
      }
    });
    
    this.logger.info('⌨️ 全局快捷键已设置: Ctrl+L=导出日志, Ctrl+B=显示/隐藏调试面板');
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
        this.showStatusInfo('调试面板已显示 (按 Ctrl+B 再次隐藏)');
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
    const v = new URLSearchParams(window.location.search).get('v');
    const hasVideoChanged = v && v !== this.currentVideoId;
    if (hasVideoChanged) {
      this.currentVideoId = v;
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
        // 启动后端转录（始终转录）
        this.startBackendTranscription();
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

  async startBackendTranscription() {
    try {
      const url = `${this.backend.baseUrl}/translate_youtube`;
      const body = { url: window.location.href, target_lang: "zh" };
      const options = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
      const resp = this.needsProxy(url) ? await this.proxyFetch(url, options) : await fetch(url, options);
      if (!resp.ok) throw new Error(`translate_youtube: ${resp.status}`);
      const data = this.needsProxy(url) ? await resp.json() : await resp.json();
      this.currentJobId = data.job_id;
      this.logger.info('✅ 完整翻译任务已提交', { jobId: this.currentJobId });
      this.pollBackendState();
    } catch (e) {
      this.logger.error('❌ 完整翻译任务启动失败', { error: e.message, stack: e.stack });
      this.showStatusInfo(`翻译任务失败: ${e.message}`);
    }
  }

  async pollBackendState() {
    if (!this.currentJobId) return;
    
    // 检查超时 (10分钟)
    if (!this.pollStartTime) {
      this.pollStartTime = Date.now();
    }
    const elapsed = Date.now() - this.pollStartTime;
    const timeoutMs = 10 * 60 * 1000; // 10分钟超时
    
    if (elapsed > timeoutMs) {
      this.logger.error('❌ 后端处理超时', { 
        jobId: this.currentJobId, 
        elapsedMs: elapsed,
        timeoutMs: timeoutMs
      });
      this.showStatusInfo('后端处理超时，请重试');
      return;
    }
    
    const url = `${this.backend.baseUrl}/job_status/${this.currentJobId}`;
    try {
      const useProxy = this.needsProxy(url);
      const resp = useProxy ? await this.proxyFetch(url, { method: 'GET' }) : await fetch(url);
      if (resp.ok) {
        const state = await resp.json();
        const oldStatus = this.backendState?.status;
        this.backendState = state;
        
        // 详细的进度更新
        if (state.progress?.stage) {
          const stage = state.progress.stage;
          const message = state.progress.message || '';
          this.showStatusInfo(`${this.getStageDisplayName(stage)}: ${message}`);
        }
        
        // 状态变化时记录日志
        if (oldStatus !== state.status) {
          this.logger.info('📡 后端状态更新', { 
            oldStatus, 
            newStatus: state.status, 
            jobId: this.currentJobId,
            progress: state.progress,
            eventsCount: state.events_count || 0
          });
          
          if (state.status === 'done') {
            this.logger.info('✅ 后端转录完成', { 
              eventsCount: state.events_count,
              elapsedMs: elapsed
            });
            this.showStatusInfo(`后端转录完成，生成${state.events_count || 0}个字幕段`);
            
            // 重置轮询时间
            this.pollStartTime = null;
            
            // 优先尝试获取SRT文件
            const srtLoaded = await this.tryLoadSRTFiles();
            if (!srtLoaded) {
              // SRT文件不可用，回退到segments方式
              this.logger.info('📄 SRT文件不可用，使用segments方式');
              this.maybeFetchBackendSegments(this.getCurrentPlayTime() || 0);
            }
            
          } else if (state.status === 'error') {
            this.logger.error('❌ 后端转录出错', { 
              error: state.error,
              traceback: state.traceback,
              progress: state.progress
            });
            this.showStatusInfo(`后端转录出错: ${state.error || '未知错误'}`);
            this.pollStartTime = null; // 停止轮询
            return;
          }
        }
        
        // 如果已完成，停止轮询
        if (state.status === 'done' || state.status === 'error') {
          this.pollStartTime = null;
          return;
        }
      } else {
        this.logger.warn('⚠️ 后端状态查询失败', { 
          status: resp.status, 
          url: url 
        });
      }
    } catch (e) {
      this.logger.debug('后端状态查询失败:', e.message);
    }
    
    // 继续轮询（增加间隔到2秒减少服务器压力）
    setTimeout(() => this.pollBackendState(), 2000);
  }
  
  getStageDisplayName(stage) {
    const stageNames = {
      'initializing': '初始化',
      'translation_cache_hit': '翻译缓存命中',
      'downloading': '下载音频',
      'checking_files': '检查文件',
      'fallback_download': '备用下载',
      'transcribing': '转录处理',
      'completed': '处理完成',
      'error': '处理出错'
    };
    return stageNames[stage] || stage;
  }

  // 清理之前的数据
  clearPreviousData() {
    this.preloadedTranslations = null;
    this.isPreloading = false;
    this.currentSubtitleText = '';
    this.translationCache.clear();
    // 移除本地字幕清理（不再使用内置获取）
    this.updateSubtitleDisplay('', '');
    
    // 重置状态追踪变量
    this.splitSegmentCount = 0;
    this.hasSummary = false;
    this.processingProgress = 0;
    
    this.logger.info('🧹 已清理之前数据和状态');
  }

  // 开始字幕预加载（改为从后端获取）
  async startSubtitlePreloading() {
    if (this.isPreloading) return;
    
    this.isPreloading = true;
    this.showStatusInfo('等待后端转录...');
    
    try {
      this.logger.info('🚀 开始预加载字幕（后端模式）...');
      
      // 简化：直接启动实时显示，等待后端完整翻译完成
      this.logger.info('🚀 启动实时显示模式，等待后端完整翻译...');
      this.segments = [];
      this.preloadedTranslations = [];
      this.showStatusInfo('等待后端完成音频下载和翻译...');
      this.startRealtimeDisplay();
      
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
      
      // 改为从后端拉取初始 segments（当前窗口前后）
      const now = this.getCurrentPlayTime() ?? 0;
      await this.maybeFetchBackendSegments(now);
      const subtitleTexts = (this.segments || []).map((s, i) => ({ index: i, text: s.text, startTime: s.startTime, endTime: s.endTime }));
      this.logger.info('📝 获取字幕文本数据(后端):', subtitleTexts.length + '条');

      // 若后端尚未返回任何段落，先启动实时显示并等待后端
      if (subtitleTexts.length === 0) {
        this.segments = [];
        this.preloadedTranslations = [];
        this.showStatusInfo('等待后端转录...');
        this.startRealtimeDisplay();
        return;
      }

      // 智能断句（仅当检测为单词级时提示展示）
      const typeInfoForUi = this.translationProcessor.detectSubtitleType(subtitleTexts);
      if (typeInfoForUi.type !== 'paragraph') {
        this.showStatusInfo('正在智能断句...');
      }
      const segments = this.translationProcessor.smartSplit(subtitleTexts);
      this.splitSegmentCount = segments.length;
      if (typeInfoForUi.type !== 'paragraph') {
        this.logger.info('✅ 断句完成:', this.splitSegmentCount + '段');
      }
      
      // 生成内容总结
      this.showStatusInfo('正在分析内容...');
      const summary = await this.translationProcessor.generateContentSummary(segments);
      this.hasSummary = true;
      this.logger.info('✅ 内容总结完成');
      
      // 懒加载翻译：仅翻译首批，后续按需触发
      this.batchSize = 30;
      this.segments = segments;
      this.summary = summary;

      // 建立段落索引映射与批次元数据
      this.segIndexMap = new Map();
      segments.forEach((s, i) => this.segIndexMap.set(s, i));
      this.batches = this.translationProcessor.createBatches(segments, this.batchSize);
      this.segIndexToBatch = new Array(segments.length);
      this.batchMeta = this.batches.map((batch, bi) => {
        let start = Number.POSITIVE_INFINITY, end = 0;
        batch.forEach(seg => {
          const idx = this.segIndexMap.get(seg);
          this.segIndexToBatch[idx] = bi;
          start = Math.min(start, seg.startTime);
          end = Math.max(end, seg.endTime);
        });
        return { startTime: start, endTime: end };
      });
      this.translatedBatch = new Set();
      this.pendingBatch = new Set();

      // 初始化可显示结构（翻译为空，待填充）
      this.preloadedTranslations = segments.map((seg, i) => ({
        index: i,
        segIndex: i,
        text: seg.text,
        translation: null,
        startTime: seg.startTime,
        endTime: seg.endTime
      }));

      // 先翻译首批
      this.showStatusInfo('翻译首批字幕...');
      await this.translateAndFillBatch(0);
      this.logger.info('✅ 首批翻译完成');
      this.showStatusInfo('预翻译完成（首批）。开始实时显示...');
      
      // 启动实时显示监听
      this.startRealtimeDisplay();
      
    } catch (error) {
      this.logger.error('❌ 预翻译处理失败:', error.message);
      throw error;
    }
  }

  // 触发翻译一个批次并回填
  async translateAndFillBatch(batchIndex) {
    if (batchIndex < 0 || batchIndex >= this.batches.length) return;
    if (this.translatedBatch.has(batchIndex) || this.pendingBatch.has(batchIndex)) return;
    this.pendingBatch.add(batchIndex);
    try {
      const batch = this.batches[batchIndex];
      const translations = await this.translationProcessor.translateBatchReturnList(batch, batchIndex, this.summary);
      batch.forEach((seg, j) => {
        const idx = this.segIndexMap.get(seg);
        const t = translations[j] || '[翻译失败]';
        
        // 确保preloadedTranslations数组和对象存在
        if (this.preloadedTranslations && this.preloadedTranslations[idx]) {
          // 只更新翻译，不要覆盖原文
          this.preloadedTranslations[idx].translation = t;
          // 确保原文不被覆盖
          if (!this.preloadedTranslations[idx].originalText) {
            this.preloadedTranslations[idx].originalText = this.preloadedTranslations[idx].text;
          }
          this.logger.debug(`✅ 批次${batchIndex + 1}第${j + 1}项翻译完成`, {
            original: this.preloadedTranslations[idx].text?.substring(0, 30) + '...',
            translation: t.substring(0, 30) + '...'
          });
        } else {
          this.logger.warn(`⚠️ 无法设置翻译结果，索引${idx}不存在`, { batchIndex, j, totalTranslations: this.preloadedTranslations?.length });
        }
      });
      this.translatedBatch.add(batchIndex);
      this.logger.info(`✅ 批次${batchIndex + 1}翻译完成`);
    } catch (e) {
      this.logger.error('❌ 批次翻译失败:', e.message);
    } finally {
      this.pendingBatch.delete(batchIndex);
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
      // 若后端存在且不在SRT模式，尝试按需拉取首批段落
      if (!this.usingSRTMode) {
        const now = this.getCurrentPlayTime();
        if (now !== null) this.maybeFetchBackendSegments(now);
      }
      return;
    }
    
    // 获取当前播放时间
    const currentTime = this.getCurrentPlayTime();
    if (currentTime === null) return;
    
    // 仅在非SRT模式下进行后端按需拉取
    if (!this.usingSRTMode) {
      this.maybeFetchBackendSegments(currentTime);
    }
    
    // 找到当前时间对应的字幕
    const currentSubtitle = this.findSubtitleAtTime(currentTime);
    
    if (currentSubtitle) {
      const text = `${currentSubtitle.text}`;
      if (text !== this.currentSubtitleText) {
        this.currentSubtitleText = text;
        
        // 确保使用正确的原文和翻译
        let original = '';
        let translation = '';
        
        if (this.usingSRTMode) {
          // SRT模式：originalText是英文，translation是翻译
          original = currentSubtitle.originalText || currentSubtitle.text || '';
          translation = currentSubtitle.translation || '';
        } else {
          // segments模式：text是原文，translation是翻译
          original = currentSubtitle.text || '';
          translation = currentSubtitle.translation || '翻译中...';
        }
        
        // 调试日志：检查数据结构
        this.logger.debug('🔍 字幕数据检查', {
          mode: this.usingSRTMode ? 'SRT' : 'segments',
          hasOriginalText: !!currentSubtitle.originalText,
          hasText: !!currentSubtitle.text,
          hasTranslation: !!currentSubtitle.translation,
          originalPreview: original?.substring(0, 30),
          translationPreview: translation?.substring(0, 30)
        });
        
        this.updateSubtitleDisplay(original, translation);
      }

      // 懒加载触发逻辑（仅在非SRT模式下）
      if (!this.usingSRTMode) {
        this.maybeTriggerOnDemand(currentTime, currentSubtitle);
      }
    } else {
      // 没有找到对应字幕，清空显示
      if (this.currentSubtitleText) {
        this.currentSubtitleText = '';
        this.updateSubtitleDisplay('', '');
      }
    }
  }

  // 按播放进度触发下一批翻译
  maybeTriggerOnDemand(currentTime, currentSubtitle) {
    const segIndex = currentSubtitle?.segIndex;
    if (typeof segIndex !== 'number') return;
    const currentBatch = this.segIndexToBatch?.[segIndex] ?? 0;

    // 当前批未翻译则优先补齐
    if (!this.translatedBatch.has(currentBatch)) {
      this.translateAndFillBatch(currentBatch);
      return;
    }

    // 预取下一批（距离开始时间≤8秒）
    const nextBatch = currentBatch + 1;
    if (nextBatch < (this.batches?.length || 0)) {
      const nextStart = this.batchMeta[nextBatch].startTime;
      if ((nextStart - currentTime) <= 8 && !this.translatedBatch.has(nextBatch) && !this.pendingBatch.has(nextBatch)) {
        this.translateAndFillBatch(nextBatch);
      }
    }
  }

  // 限流并按窗口从后端拉取 segments（首版实现）
  maybeFetchBackendSegments(currentTimeSec) {
    if (!this.currentJobId || !this.backend?.baseUrl) return;
    
    // 如果已经在使用SRT模式，跳过segments获取
    if (this.usingSRTMode) {
      return;
    }
    const now = Date.now();
    if (this._lastSegFetchTs && (now - this._lastSegFetchTs) < 800) return; // 800ms 限流
    this._lastSegFetchTs = now;
    const ahead = this.backend.prefetchAheadSec || 30;
    const fromMs = Math.max(0, Math.floor((currentTimeSec - 2) * 1000));
    const toMs = Math.floor((currentTimeSec + ahead) * 1000);
    const url = `${this.backend.baseUrl}/segments/${this.currentJobId}?start=${Math.floor(currentTimeSec)}&window=60`;
    const useProxy = this.needsProxy(url);
    const p = useProxy ? this.proxyFetch(url, { method: 'GET' }) : fetch(url);
    p.then(r => r.ok ? (useProxy ? r.json() : r.json()) : null).then(data => {
      if (!data || !Array.isArray(data.events) || data.events.length === 0) return;
      this.incorporateBackendSegments(data.events, true); // 标记为翻译数据
    }).catch(() => {});
    
    // 同时获取原始字幕作为原文（如果还没获取过）
    if (!this.originalSubtitlesFetched) {
      this.fetchOriginalSubtitles();
    }
  }
  
  // 获取原始字幕作为原文
  async fetchOriginalSubtitles() {
    if (this.originalSubtitlesFetched) return;
    
    try {
      const vid = this.currentVideoId;
      if (!vid) return;
      
      // 尝试从缓存获取原始字幕
      const cacheUrl = `${this.backend.baseUrl}/cache/check/${vid}`;
      const useProxy = this.needsProxy(cacheUrl);
      const resp = useProxy ? await this.proxyFetch(cacheUrl, { method: 'GET' }) : await fetch(cacheUrl);
      
      if (resp.ok) {
        const cacheInfo = await resp.json();
        if (cacheInfo.has_subtitle_cache) {
          this.logger.info('📖 发现原始字幕缓存，将用作双语显示的原文');
          this.originalSubtitlesFetched = true;
          // 这里可以进一步获取原始字幕内容，但目前先用转录结果作为原文
        }
      }
    } catch (e) {
      this.logger.debug('获取原始字幕失败:', e.message);
    }
  }

  // 获取和处理SRT文件
  async tryLoadSRTFiles() {
    if (!this.currentJobId) return false;
    
    try {
      this.logger.info('📄 尝试获取SRT文件...');
      const url = `${this.backend.baseUrl}/srt_files/${this.currentJobId}`;
      const useProxy = this.needsProxy(url);
      const resp = useProxy ? await this.proxyFetch(url, { method: 'GET' }) : await fetch(url);
      
      if (!resp.ok) {
        this.logger.debug('SRT文件获取失败:', resp.status);
        return false;
      }
      
      const data = await resp.json();
      this.logger.info('📄 SRT文件获取结果:', {
        hasEnglish: data.has_english_srt,
        hasTranslated: data.has_translated_srt,
        videoId: data.video_id,
        source: data.source
      });
      
      // 关键调试：检查返回的SRT内容前几行
      if (data.english_srt) {
        const englishPreview = data.english_srt.split('\n').slice(0, 8).join('\n');
        this.logger.debug('🔍 英文SRT内容预览:', englishPreview);
      }
      
      if (data.translated_srt) {
        const translatedPreview = data.translated_srt.split('\n').slice(0, 8).join('\n');
        this.logger.debug('🔍 翻译SRT内容预览:', translatedPreview);
      }
      
      if (data.has_english_srt && data.has_translated_srt) {
        this.logger.info('✅ 发现完整的双语SRT文件，开始处理...');
        // 注意参数顺序：第一个应该是英文，第二个应该是翻译
        this.logger.debug('🔄 开始加载双语SRT，参数顺序：english_srt, translated_srt');
        this.loadBillingualSRTFiles(data.english_srt, data.translated_srt);
        return true;
      } else if (data.has_english_srt || data.has_translated_srt) {
        this.logger.info('⚠️ 仅发现部分SRT文件，继续使用segments方式');
        return false;
      }
      
      return false;
      
    } catch (e) {
      this.logger.debug('SRT文件获取失败:', e.message);
      return false;
    }
  }
  
  // 处理双语SRT文件
  loadBillingualSRTFiles(englishSRT, translatedSRT) {
    try {
      this.logger.info('🔄 解析双语SRT文件...');
      this.logger.debug('🔍 参数验证:', {
        englishSRTLength: englishSRT?.length || 0,
        translatedSRTLength: translatedSRT?.length || 0,
        englishSRTFirst50: englishSRT?.substring(0, 50),
        translatedSRTFirst50: translatedSRT?.substring(0, 50)
      });
      
      // 解析SRT文件
      const englishSubtitles = SRTParser.parse(englishSRT);
      const translatedSubtitles = SRTParser.parse(translatedSRT);
      
      this.logger.info('📊 SRT解析结果:', {
        englishCount: englishSubtitles.length,
        translatedCount: translatedSubtitles.length
      });
      
      // 调试：检查解析后的内容
      if (englishSubtitles.length > 0) {
        this.logger.debug('🔍 英文SRT首条内容:', {
          text: englishSubtitles[0].text?.substring(0, 50),
          startTime: englishSubtitles[0].startTime,
          endTime: englishSubtitles[0].endTime
        });
      }
      
      if (translatedSubtitles.length > 0) {
        this.logger.debug('🔍 翻译SRT首条内容:', {
          text: translatedSubtitles[0].text?.substring(0, 50),
          startTime: translatedSubtitles[0].startTime,
          endTime: translatedSubtitles[0].endTime
        });
      }
      
      // 验证SRT文件质量
      const englishStats = SRTParser.getStats(englishSubtitles);
      const translatedStats = SRTParser.getStats(translatedSubtitles);
      
      this.logger.info('📈 SRT文件统计:', {
        english: englishStats,
        translated: translatedStats
      });
      
      // 合并双语字幕
      const bilingualSubtitles = SRTParser.mergeBilingualSubtitles(englishSubtitles, translatedSubtitles);
      
      this.logger.info('🔀 双语字幕合并完成:', {
        totalCount: bilingualSubtitles.length,
        withTranslation: bilingualSubtitles.filter(s => s.hasTranslation).length
      });
      
      // 调试：检查合并后的首条数据
      if (bilingualSubtitles.length > 0) {
        const first = bilingualSubtitles[0];
        this.logger.debug('🔍 合并后首条数据:', {
          originalText: first.originalText?.substring(0, 50),
          translation: first.translation?.substring(0, 50),
          hasTranslation: first.hasTranslation,
          text: first.text?.substring(0, 50)
        });
        
        // 检查前3条数据
        for (let i = 0; i < Math.min(3, bilingualSubtitles.length); i++) {
          const item = bilingualSubtitles[i];
          this.logger.debug(`🔍 第${i+1}条合并数据:`, {
            originalText: item.originalText?.substring(0, 30),
            translation: item.translation?.substring(0, 30),
            textField: item.text?.substring(0, 30),
            startTime: item.startTime,
            endTime: item.endTime
          });
        }
      }
      
      // 更新preloadedTranslations结构
      this.preloadedTranslations = bilingualSubtitles.map((subtitle, index) => ({
        index: index,
        segIndex: index,
        text: subtitle.originalText || subtitle.text, // 确保使用原始英文文本
        originalText: subtitle.originalText || subtitle.text, // 英文原文
        translation: subtitle.translation, // 中文翻译
        startTime: subtitle.startTime,
        endTime: subtitle.endTime,
        hasTranslation: subtitle.hasTranslation
      }));
      
      this.logger.info('✅ SRT双语字幕加载完成', {
        totalSubtitles: this.preloadedTranslations.length,
        withTranslations: this.preloadedTranslations.filter(s => s.hasTranslation).length
      });
      
      // 标记为SRT模式，跳过segments处理
      this.usingSRTMode = true;
      this.showStatusInfo(`SRT双语字幕加载完成 (${this.preloadedTranslations.length}条)`);
      
      // 启动实时显示
      this.startRealtimeDisplay();
      
    } catch (e) {
      this.logger.error('❌ SRT文件处理失败:', e.message);
      this.showStatusInfo(`SRT处理失败: ${e.message}`);
      // 回退到segments模式
      this.usingSRTMode = false;
    }
  }

  // 将后端 events 合并为前端 segments，并建立/更新懒加载翻译结构
  async incorporateBackendSegments(events) {
    // 1) 映射为统一 segments（段落字幕，无需断句）
    const backendSegments = [];
    for (const ev of events) {
      const start = (ev.tStartMs || 0) / 1000;
      const dur = (ev.dDurationMs || 0) / 1000;
      const end = start + dur;
      const text = Array.isArray(ev.segs) ? ev.segs.map(s => s.utf8 || '').join('') : (ev.text || '');
      if (text && end > start) backendSegments.push({ text: text.trim(), startTime: start, endTime: end });
    }
    if (backendSegments.length === 0) return;

    // 2) 合并到现有 segments，去重（按 start/end/text）
    const keyOf = (s) => `${s.startTime.toFixed(3)}|${s.endTime.toFixed(3)}|${s.text}`;
    const existing = this.segments || [];
    const seen = new Set(existing.map(keyOf));
    const merged = existing.slice();
    let newSegmentsAdded = 0;
    
    for (const s of backendSegments) {
      const k = keyOf(s);
      if (!seen.has(k)) { 
        merged.push(s); 
        seen.add(k); 
        newSegmentsAdded++;
      }
    }
    
    // 如果没有新段落，跳过重建
    if (newSegmentsAdded === 0) {
      return;
    }
    
    merged.sort((a,b)=> (a.startTime - b.startTime) || (a.endTime - b.endTime));
    this.logger.info(`📝 合并新段落: ${newSegmentsAdded}个新段落，总计${merged.length}个`);

    // 3) 生成/更新懒加载翻译结构
    const oldTranslations = new Map();
    if (this.preloadedTranslations) {
      for (const t of this.preloadedTranslations) {
        if (t && t.text && t.translation) oldTranslations.set(t.text, t.translation);
      }
    }

    // 智能断句：段落字幕检测为 paragraph 时，smartSplit 会直接透传
    // 只对新段落进行断句处理，避免重复
    const segments = this.translationProcessor.smartSplit(merged.map((s, i) => ({ index: i, text: s.text, startTime: s.startTime, endTime: s.endTime })));
    this.segments = segments;

    // 重新构建批次与索引（仅在有新段落时）
    this.batchSize = this.batchSize || 30;
    this.segIndexMap = new Map();
    segments.forEach((s, i) => this.segIndexMap.set(s, i));
    this.batches = this.translationProcessor.createBatches(segments, this.batchSize);
    this.segIndexToBatch = new Array(segments.length);
    this.batchMeta = this.batches.map((batch, bi) => {
      let start = Number.POSITIVE_INFINITY, end = 0;
      batch.forEach(seg => {
        const idx = this.segIndexMap.get(seg);
        this.segIndexToBatch[idx] = bi;
        start = Math.min(start, seg.startTime);
        end = Math.max(end, seg.endTime);
      });
      return { startTime: start, endTime: end };
    });
    this.translatedBatch = this.translatedBatch || new Set();
    this.pendingBatch = this.pendingBatch || new Set();

    // 重建 preloadedTranslations，并尽量带上已有翻译
    this.preloadedTranslations = segments.map((seg, i) => ({
      index: i,
      segIndex: i,
      text: seg.text,  // 确保这是原文
      translation: oldTranslations.get(seg.text) || null,  // 这是翻译
      startTime: seg.startTime,
      endTime: seg.endTime,
      originalText: seg.text  // 额外保存原文，防止被覆盖
    }));
    
    // 确保翻译处理器有最新的设置（只在初次设置时记录日志）
    if (this.translationProcessor && !this.translationProcessorConfigured) {
      this.translationProcessor.settings = this.settings;
      this.translationProcessorConfigured = true;
      this.logger.info('🔧 初始化翻译处理器设置', {
        hasApiKey: !!this.settings.apiKey,
        apiUrl: this.settings.apiUrl,
        model: this.settings.model
      });
    }

    // 若首批尚未翻译，触发首批
    if (!this.translatedBatch.has(0) && !this.pendingBatch.has(0)) {
      this.translateAndFillBatch(0);
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
        this.updateTranslationProcessorSettings();
      }
    });
  }
  
  updateTranslationProcessorSettings() {
    if (this.translationProcessor) {
      this.translationProcessor.settings = this.settings;
      this.logger.info('🔧 翻译处理器设置已更新', {
        hasApiKey: !!this.settings.apiKey,
        apiKeyLength: this.settings.apiKey?.length || 0,
        apiUrl: this.settings.apiUrl,
        model: this.settings.model,
        targetLang: this.settings.targetLang
      });
    }
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
    
    // 确保容器可见
    this.translationContainer.style.display = 'block';
    this.translationContainer.style.visibility = 'visible';
    this.translationContainer.style.opacity = '1';
    
    // 智能选择容器的父元素：全屏时插入到播放器，否则插入到body
    const isFullscreen = document.fullscreenElement !== null || 
                         document.querySelector('.html5-video-player.ytp-fullscreen') !== null;
    const player = document.querySelector('.html5-video-player');
    
    if (isFullscreen && player) {
      // 全屏模式：插入到播放器容器内
      player.appendChild(this.translationContainer);
      this.logger.info('✅ 双语字幕容器已插入到播放器（全屏模式）');
    } else {
      // 普通模式：插入到body
      document.body.appendChild(this.translationContainer);
      this.logger.info('✅ 双语字幕容器已插入到body（普通模式）');
    }
    
    // 初始化智能定位系统
    this.initializeSmartPositioning();
    
    // 检查容器是否正确插入
    const containerCheck = document.getElementById('youtube-bilingual-subtitles');
    this.logger.info('✅ 双语字幕容器创建成功', {
      containerExists: !!containerCheck,
      containerVisible: containerCheck?.style.display,
      containerOpacity: containerCheck?.style.opacity,
      parentElement: containerCheck?.parentElement?.tagName,
      hasChildren: containerCheck?.children?.length || 0
    });
  }

  // 智能定位系统
  initializeSmartPositioning() {
    // 创建位置管理器
    this.positionManager = new SubtitlePositionManager(this.translationContainer, this.logger);
    
    // 添加双击重置功能
    this.positionManager.addDoubleClickReset();
    
    // 监听视频尺寸变化
    this.setupVideoSizeObserver();
    
    // 监听全屏状态变化
    this.setupFullscreenObserver();
    
    // 立即应用初始定位
    setTimeout(() => {
      // 强制重新初始化位置，特别是在页面刷新后
      this.forcePositionReset();
      this.positionManager.updatePosition();
    }, 100);
    
    // 延迟再次确保位置正确（处理页面刷新情况）
    setTimeout(() => {
      this.positionManager.updatePosition();
      this.logger.debug('🔄 延迟位置更新完成');
    }, 500);
  }

  // 强制重置位置（特别是页面刷新后）
  forcePositionReset() {
    if (!this.translationContainer) return;
    
    // 清除所有可能的自定义位置
    this.translationContainer.style.removeProperty('top');
    this.translationContainer.style.removeProperty('bottom');
    this.translationContainer.style.removeProperty('left');
    this.translationContainer.style.removeProperty('right');
    this.translationContainer.style.removeProperty('transform');
    
    // 重置为默认状态
    this.translationContainer.style.position = 'fixed';
    this.translationContainer.style.zIndex = '999999999';
    
    // 如果有位置管理器，也重置它的状态
    if (this.positionManager) {
      this.positionManager.resetToDefault();
    }
    
    this.logger.debug('🔄 位置已强制重置');
  }

  // 监听视频尺寸变化
  setupVideoSizeObserver() {
    const videoElement = document.querySelector('video');
    if (!videoElement) return;

    // 使用ResizeObserver监听视频尺寸变化
    if (window.ResizeObserver) {
      this.videoResizeObserver = new ResizeObserver(() => {
        this.positionManager?.updatePosition();
      });
      this.videoResizeObserver.observe(videoElement);
    }

    // 监听视频播放器容器变化
    const playerContainer = document.querySelector('#player-container, .html5-video-player');
    if (playerContainer && window.ResizeObserver) {
      this.playerResizeObserver = new ResizeObserver(() => {
        this.positionManager?.updatePosition();
      });
      this.playerResizeObserver.observe(playerContainer);
    }
  }

  // 监听全屏状态变化
  setupFullscreenObserver() {
    document.addEventListener('fullscreenchange', () => {
      setTimeout(() => {
        this.handleFullscreenChange();
      }, 100); // 延迟以确保全屏状态已完全应用
    });

    // 监听YouTube特定的全屏事件
    const player = document.querySelector('.html5-video-player');
    if (player) {
      const observer = new MutationObserver(() => {
        setTimeout(() => {
          this.handleFullscreenChange();
        }, 100);
      });
      
      observer.observe(player, {
        attributes: true,
        attributeFilter: ['class']
      });
    }
  }

  // 处理全屏状态变化
  handleFullscreenChange() {
    if (!this.translationContainer) return;

    const isFullscreen = document.fullscreenElement !== null || 
                         document.querySelector('.html5-video-player.ytp-fullscreen') !== null;
    const player = document.querySelector('.html5-video-player');

    this.logger.debug('🔄 全屏状态变化', { isFullscreen });

    // 根据全屏状态调整容器父元素
    if (isFullscreen && player && !player.contains(this.translationContainer)) {
      // 切换到全屏：移动到播放器内
      player.appendChild(this.translationContainer);
      this.logger.debug('📺 字幕容器移动到播放器（全屏模式）');
      
      // 重新初始化拖拽功能（关键修复）
      setTimeout(() => {
        this.positionManager?.cleanup();
        this.positionManager = new SubtitlePositionManager(this.translationContainer, this.logger);
        this.positionManager.addDoubleClickReset();
        
        // 切换到全屏时，重置位置以避免坐标系统混淆
        this.positionManager.resetToDefault();
        
        // 详细检查容器状态
        const containerCheck = {
          exists: !!this.translationContainer,
          inDOM: !!document.getElementById('youtube-bilingual-subtitles'),
          parent: this.translationContainer?.parentElement?.tagName,
          visible: this.translationContainer?.style.display,
          opacity: this.translationContainer?.style.opacity,
          hasSubtitleContainer: !!this.translationContainer?.querySelector('.subtitle-container'),
          hasEventListeners: this.positionManager ? 'yes' : 'no'
        };
        
        this.logger.info('🔄 全屏模式拖拽重新初始化完成', containerCheck);
        console.log('🔄 全屏容器状态检查', containerCheck);
      }, 200); // 延迟确保DOM完全更新
      
    } else if (!isFullscreen && !document.body.contains(this.translationContainer)) {
      // 切换到窗口：移动到body
      document.body.appendChild(this.translationContainer);
      this.logger.debug('🖥️ 字幕容器移动到body（窗口模式）');
      
      // 重新初始化拖拽功能（关键修复）
      setTimeout(() => {
        this.positionManager?.cleanup();
        this.positionManager = new SubtitlePositionManager(this.translationContainer, this.logger);
        this.positionManager.addDoubleClickReset();
        
        // 切换到窗口模式时，重置位置以避免坐标系统混淆
        this.positionManager.resetToDefault();
        
        this.logger.debug('🔄 窗口模式拖拽重新初始化完成');
      }, 200); // 延迟确保DOM完全更新
    }

    // 延迟更新位置，确保DOM完全稳定后再计算
    setTimeout(() => {
      this.positionManager?.updatePosition();
      this.logger.debug('🔄 全屏状态变化后位置已更新', {
        isFullscreen,
        containerParent: this.translationContainer?.parentElement?.tagName
      });
    }, 300);
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
    
    const url = this.settings.apiUrl + '/chat/completions';
    const fetchOptions = {
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
    };

    const response = this.needsProxy(url)
      ? await this.proxyFetch(url, fetchOptions)
      : await fetch(url, fetchOptions);
    
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
    
    // 确保内容不为空
    const displayOriginal = original && original.trim() ? original.trim() : '';
    const displayTranslated = translated && translated.trim() ? translated.trim() : '';
    
    // 更新文本内容
    originalElement.textContent = displayOriginal;
    translatedElement.textContent = displayTranslated;
    
    // 调试日志：检查内容更新
    this.logger.debug('🔍 字幕内容更新', {
      original: displayOriginal,
      translated: displayTranslated,
      originalLength: displayOriginal?.length || 0,
      translatedLength: displayTranslated?.length || 0
    });
    
    // 动态调整背景样式根据内容
    if (displayOriginal) {
      originalElement.style.background = 'rgba(0, 0, 0, 0.75)';
      originalElement.style.padding = '3px 8px';
      originalElement.style.display = 'inline-block';
    } else {
      originalElement.style.background = 'transparent';
      originalElement.style.padding = '0';
      originalElement.style.display = 'none';
    }
    
    if (displayTranslated) {
      if (displayTranslated === '翻译中...') {
        // 翻译中状态：显示但无背景
        translatedElement.style.background = 'transparent';
        translatedElement.style.padding = '0';
        translatedElement.style.display = 'inline-block';
      } else {
        // 正常翻译结果：显示背景
        translatedElement.style.background = 'rgba(0, 0, 0, 0.75)';
        translatedElement.style.padding = '3px 8px';
        translatedElement.style.display = 'inline-block';
      }
    } else {
      translatedElement.style.background = 'transparent';
      translatedElement.style.padding = '0';
      translatedElement.style.display = 'none';
    }
    
    // 调试日志
    if (displayOriginal && displayTranslated && displayTranslated !== '翻译中...') {
      this.logger.debug('🎬 双语字幕显示', {
        original: displayOriginal.substring(0, 50) + '...',
        translated: displayTranslated.substring(0, 50) + '...'
      });
    }
    
    // 显示/隐藏逻辑
    const shouldShow = displayOriginal || displayTranslated;
    
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
    // 显示状态信息（默认隐藏，可通过 Ctrl+D 切换）
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
        display: none;
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
    const subtitleCount = this.preloadedTranslations?.length || 0;
    const preloadedCount = this.preloadedTranslations?.length || 0;
    
    // 统计日志中的错误和警告
    const errorCount = this.logger.logs.filter(log => log.level === 'ERROR').length;
    const warnCount = this.logger.logs.filter(log => log.level === 'WARN').length;
    
    return {
      mode: this.usingSRTMode ? 'SRT双语模式' : 
            (this.settings.usePreload ? '预加载模式 (增强版)' : '实时翻译模式'),
      videoId: this.currentVideoId || '未检测',
      subtitleStatus: subtitleCount > 0 ? `✅ 已获取 (${subtitleCount}条)` : '⏳ 等待后端',
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

// 初始化翻译器（移动到文件末尾避免重复定义）

// 字幕位置管理器（支持智能定位和拖拽）
class SubtitlePositionManager {
  constructor(container, logger) {
    this.container = container;
    this.logger = logger;
    this.lastPosition = null;
    this.isDragging = false;
    this.dragOffset = { x: 0, y: 0 };
    this.customPosition = null; // 用户自定义位置
    
    // 初始化拖拽功能
    this.initializeDragFeature();
  }

  // 初始化拖拽功能
  initializeDragFeature() {
    if (!this.container) return;

    // 创建拖拽提示
    this.createDragIndicator();
    
    // 绑定拖拽事件到整个容器
    this.bindContainerDragEvents();
  }

  // 创建拖拽提示
  createDragIndicator() {
    // 检查是否已经有拖拽提示
    if (this.container.querySelector('.drag-indicator')) return;

    const dragIndicator = document.createElement('div');
    dragIndicator.className = 'drag-indicator';
    dragIndicator.innerHTML = '⋮⋮⋮'; // 三竖点图标
    dragIndicator.title = '鼠标悬停显示拖拽区域\n上下拖拽调整字幕位置\n双击重置到默认位置';
    dragIndicator.style.cssText = `
      position: absolute;
      top: -12px;
      left: 50%;
      transform: translateX(-50%);
      width: 30px;
      height: 12px;
      background: rgba(0, 255, 255, 0.6);
      border-radius: 6px 6px 0 0;
      cursor: ns-resize;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 8px;
      color: rgba(0, 0, 0, 0.8);
      opacity: 0;
      transition: all 0.3s ease;
      z-index: 1000000000;
      user-select: none;
      pointer-events: auto;
      font-weight: bold;
    `;

    // 插入拖拽提示
    const subtitleContainer = this.container.querySelector('.subtitle-container');
    if (subtitleContainer) {
      subtitleContainer.style.position = 'relative';
      subtitleContainer.appendChild(dragIndicator);
    }

    this.dragIndicator = dragIndicator;
  }

  // 绑定拖拽事件到整个容器
  bindContainerDragEvents() {
    if (!this.container) return;

    const subtitleContainer = this.container.querySelector('.subtitle-container');
    if (!subtitleContainer) return;
    
    // 调试：记录绑定信息
    const isFullscreen = document.fullscreenElement !== null || 
                         document.querySelector('.html5-video-player.ytp-fullscreen') !== null;
    this.logger.info('🔗 绑定拖拽事件', {
      isFullscreen,
      containerParent: this.container?.parentElement?.tagName,
      containerInDOM: !!document.getElementById('youtube-bilingual-subtitles'),
      subtitleContainerExists: !!subtitleContainer
    });

    // 添加简单的点击测试
    this.container.addEventListener('click', (e) => {
      console.log('🖱️ 字幕容器被点击', {
        target: e.target.className,
        isFullscreen: document.fullscreenElement !== null || 
                     document.querySelector('.html5-video-player.ytp-fullscreen') !== null
      });
      this.logger.info('🖱️ 字幕容器被点击', {
        target: e.target.className,
        isFullscreen: document.fullscreenElement !== null || 
                     document.querySelector('.html5-video-player.ytp-fullscreen') !== null
      });
    });

    // 显示/隐藏拖拽提示和拖拽区域
    this.container.addEventListener('mouseenter', () => {
      if (this.dragIndicator) {
        this.dragIndicator.style.opacity = '0.8';
      }
      // 为整个容器添加拖拽区域样式
      subtitleContainer.style.boxShadow = '0 0 0 2px rgba(0, 255, 255, 0.3)';
      subtitleContainer.style.borderRadius = '6px';
      
      // 调试：记录鼠标进入事件
      const isFullscreen = document.fullscreenElement !== null || 
                          document.querySelector('.html5-video-player.ytp-fullscreen') !== null;
      this.logger.info('🖱️ 鼠标进入字幕区域', {
        isFullscreen,
        containerParent: this.container?.parentElement?.tagName,
        hasSubtitleContainer: !!subtitleContainer,
        dragIndicatorVisible: this.dragIndicator?.style.opacity,
        containerVisible: this.container.style.display,
        containerOpacity: this.container.style.opacity
      });
    });

    this.container.addEventListener('mouseleave', () => {
      if (!this.isDragging) {
        if (this.dragIndicator) {
          this.dragIndicator.style.opacity = '0';
        }
        subtitleContainer.style.boxShadow = 'none';
      }
    });

    // 鼠标按下开始拖拽（整个容器可拖拽）
    subtitleContainer.addEventListener('mousedown', (e) => {
      // 强制记录所有鼠标按下事件
      console.log('🎯 鼠标按下事件触发', e);
      
      // 检查是否点击在文字上（避免选中文字）
      const target = e.target;
      if (target.classList.contains('original-subtitle') || target.classList.contains('translated-subtitle')) {
        this.logger.info('🚫 点击在文字上，跳过拖拽');
        return; // 点击文字本身时不启动拖拽
      }
      
      const isFullscreen = document.fullscreenElement !== null || 
                          document.querySelector('.html5-video-player.ytp-fullscreen') !== null;
      
      this.logger.info('🎯 拖拽事件触发', {
        isFullscreen: isFullscreen,
        containerParent: this.container?.parentElement?.tagName,
        eventTarget: target.className,
        playerElement: !!document.querySelector('.html5-video-player'),
        fullscreenElement: !!document.fullscreenElement,
        mouseButton: e.button,
        clientX: e.clientX,
        clientY: e.clientY
      });
      
      console.log('🎯 开始拖拽逻辑');
      e.preventDefault();
      e.stopPropagation();
      this.startDragging(e);
    });

    // 鼠标移动
    document.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        this.handleDragging(e);
      }
    });

    // 鼠标松开结束拖拽
    document.addEventListener('mouseup', (e) => {
      if (this.isDragging) {
        this.stopDragging(e);
      }
    });

    // 触摸事件支持
    subtitleContainer.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      this.startDragging(touch);
    });

    document.addEventListener('touchmove', (e) => {
      if (this.isDragging) {
        e.preventDefault();
        const touch = e.touches[0];
        this.handleDragging(touch);
      }
    });

    document.addEventListener('touchend', (e) => {
      if (this.isDragging) {
        this.stopDragging(e);
      }
    });
  }

  // 开始拖拽
  startDragging(event) {
    this.isDragging = true;
    
    // 记录初始鼠标位置（简化偏移计算）
    this.initialMouseY = event.clientY;

    // 添加拖拽时的视觉效果
    this.container.style.transition = 'none'; // 拖拽时关闭过渡动画
    
    // 显示拖拽状态
    if (this.dragIndicator) {
      this.dragIndicator.style.opacity = '1';
      this.dragIndicator.style.background = 'rgba(0, 255, 255, 0.9)';
    }
    
    const subtitleContainer = this.container.querySelector('.subtitle-container');
    if (subtitleContainer) {
      subtitleContainer.style.boxShadow = '0 0 0 3px rgba(0, 255, 255, 0.6)';
    }
    
    this.logger.debug('🎯 开始拖拽字幕', {
      initialMouseY: this.initialMouseY
    });
  }

  // 处理拖拽移动（仅垂直方向，保持水平居中）
  handleDragging(event) {
    if (!this.isDragging) return;

    // 获取当前鼠标位置
    const currentY = event.clientY;

    // 获取视频容器信息来保持居中
    const video = document.querySelector('video');
    const videoRect = video ? video.getBoundingClientRect() : null;
    
    if (!videoRect) return;

    // 检查全屏模式
    const isFullscreen = document.fullscreenElement !== null || 
                         document.querySelector('.html5-video-player.ytp-fullscreen') !== null;

    // 计算新的底部偏移量
    const videoBottom = videoRect.bottom;
    const containerHeight = this.container.offsetHeight || 60;
    
    // 根据全屏状态使用不同的坐标系统
    let newBottomOffset;
    let finalBottom;
    let screenHeight;

    if (isFullscreen) {
      // 全屏模式：使用屏幕高度
      screenHeight = screen.height;
      newBottomOffset = Math.max(20, videoBottom - currentY + containerHeight/2);
      finalBottom = screenHeight - videoBottom + newBottomOffset;
    } else {
      // 窗口模式：使用窗口高度
      screenHeight = window.innerHeight;
      newBottomOffset = Math.max(20, videoBottom - currentY + containerHeight/2);
      finalBottom = screenHeight - videoBottom + newBottomOffset;
    }
    
    // 限制在合理范围内
    const maxOffset = videoRect.height + 100; // 不能超出视频高度太多
    const minOffset = 20; // 最少20px距离
    const constrainedBottomOffset = Math.max(minOffset, Math.min(maxOffset, newBottomOffset));

    // 计算视频中心位置
    const videoCenterX = videoRect.left + videoRect.width / 2;

    // 应用新位置：保持在视频水平居中，只改变垂直位置
    this.container.style.left = `${videoCenterX}px`;
    this.container.style.top = 'auto';
    this.container.style.bottom = `${screenHeight - videoBottom + constrainedBottomOffset}px`;
    this.container.style.transform = 'translateX(-50%) scale(1.02)';
    this.container.style.position = 'fixed';

    // 保存自定义位置（保存相对于视频的偏移）
    this.customPosition = {
      bottomOffset: constrainedBottomOffset,
      isCustom: true
    };
    
    // 调试日志
    this.logger.debug('🔄 拖拽中', {
      mouseY: currentY,
      videoBottom: videoBottom,
      bottomOffset: constrainedBottomOffset,
      finalBottom: screenHeight - videoBottom + constrainedBottomOffset,
      isFullscreen: isFullscreen,
      screenHeight: screenHeight,
      windowHeight: window.innerHeight
    });
  }

  // 结束拖拽
  stopDragging(event) {
    if (!this.isDragging) return;

    this.isDragging = false;
    
    // 恢复拖拽提示样式
    if (this.dragIndicator) {
      this.dragIndicator.style.opacity = '0';
      this.dragIndicator.style.background = 'rgba(0, 255, 255, 0.6)';
    }
    
    // 恢复缩放效果，保持视频居中
    this.container.style.transform = 'translateX(-50%) scale(1)';
    this.container.style.transition = 'transform 0.2s ease';
    
    // 恢复容器样式
    const subtitleContainer = this.container.querySelector('.subtitle-container');
    if (subtitleContainer) {
      subtitleContainer.style.boxShadow = 'none';
    }
    
    // 保存位置到localStorage
    if (this.customPosition) {
      this.saveCustomPosition();
    }

    this.logger.debug('🎯 拖拽结束', {
      position: this.customPosition,
      bottomOffset: this.customPosition?.bottomOffset
    });
  }

  // 计算垂直拖拽边界
  calculateVerticalDragBounds() {
    const video = document.querySelector('video');
    const videoRect = video ? video.getBoundingClientRect() : null;
    const containerHeight = this.container.offsetHeight || 80;

    let minY = 50; // 距离顶部最小距离
    let maxY = window.innerHeight - containerHeight - 50; // 距离底部最小距离

    // 如果有视频，根据视频位置调整边界
    if (videoRect) {
      minY = Math.max(minY, videoRect.top + 20); // 不能超出视频顶部太多
      maxY = Math.min(maxY, videoRect.bottom - containerHeight - 20); // 不能超出视频底部太多
    }

    return {
      minY: minY,
      maxY: maxY
    };
  }

  // 保存自定义位置
  saveCustomPosition() {
    if (!this.customPosition) return;

    try {
      const positionData = {
        ...this.customPosition,
        timestamp: Date.now(),
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        }
      };

      localStorage.setItem('youtube-subtitle-position', JSON.stringify(positionData));
      this.logger.debug('💾 字幕位置已保存', positionData);
    } catch (e) {
      this.logger.error('❌ 保存字幕位置失败', e);
    }
  }

  // 加载自定义位置
  loadCustomPosition() {
    try {
      const saved = localStorage.getItem('youtube-subtitle-position');
      if (!saved) return null;

      const positionData = JSON.parse(saved);
      
      // 检查位置是否仍然有效（视窗尺寸变化检测）
      const currentViewport = {
        width: window.innerWidth,
        height: window.innerHeight
      };

      if (positionData.viewport && 
          Math.abs(positionData.viewport.width - currentViewport.width) < 100 &&
          Math.abs(positionData.viewport.height - currentViewport.height) < 100) {
        
        this.customPosition = positionData;
        return positionData;
      }
    } catch (e) {
      this.logger.error('❌ 加载字幕位置失败', e);
    }

    return null;
  }

  // 重置到默认位置
  resetToDefault() {
    this.customPosition = null;
    localStorage.removeItem('youtube-subtitle-position');
    this.updatePosition();
    this.logger.debug('🔄 字幕位置已重置');
  }

  // 获取YouTube原生字幕位置
  getYouTubeSubtitlePosition() {
    const video = document.querySelector('video');
    const playerContainer = document.querySelector('#player-container, .html5-video-player');
    const isFullscreen = this.isFullscreen();

    if (!video) {
      return this.getDefaultPosition();
    }

    const videoRect = video.getBoundingClientRect();
    
    // 检查视频是否有效显示
    if (videoRect.width === 0 || videoRect.height === 0) {
      this.logger.debug('⚠️ 视频尺寸无效，使用默认位置');
      return this.getDefaultPosition();
    }
    
    // 获取播放器容器信息（更精确的参考）
    const playerRect = playerContainer ? playerContainer.getBoundingClientRect() : null;
    
    // 如果有自定义位置，计算相对于视频的位置
    const savedPosition = this.loadCustomPosition();
    if (savedPosition && savedPosition.isCustom && savedPosition.bottomOffset !== undefined) {
      // 计算相对于视频底部的位置
      const videoBottom = videoRect.bottom;
      const absoluteBottom = Math.max(20, videoBottom - savedPosition.bottomOffset);
      
      // 根据全屏状态使用不同的高度计算
      const screenHeight = isFullscreen ? screen.height : window.innerHeight;
      
      return {
        left: `${videoRect.left + videoRect.width / 2}px`,
        top: 'auto',
        bottom: `${screenHeight - absoluteBottom}px`,
        transform: 'translateX(-50%)',
        isCustom: true,
        videoWidth: videoRect.width,
        videoHeight: videoRect.height,
        isFullscreen: isFullscreen
      };
    }

    // 精确的YouTube字幕位置计算
    // 基于视频容器和播放器容器的相对位置
    let bottomOffset;
    
    if (isFullscreen) {
      // 全屏模式：使用视频高度的8%
      bottomOffset = Math.max(40, videoRect.height * 0.08);
    } else {
      // 窗口模式：根据视频大小调整计算策略
      if (videoRect.width <= 700) {
        // 小视频窗口（≤700px）：使用更小的偏移
        bottomOffset = Math.max(40, Math.min(80, videoRect.height * 0.12));
      } else if (videoRect.width <= 1000) {
        // 中等视频窗口：使用中等偏移
        bottomOffset = Math.max(60, Math.min(100, videoRect.height * 0.13));
      } else {
        // 大视频窗口：使用标准偏移
        bottomOffset = Math.max(80, Math.min(120, videoRect.height * 0.15));
      }
      
      // 如果有播放器容器，做微调
      if (playerRect) {
        const playerBottom = playerRect.bottom;
        const videoBottom = videoRect.bottom;
        const controlBarHeight = playerBottom - videoBottom;
        
        // 对于小视频，控制栏处理更保守
        if (videoRect.width <= 700) {
          // 小视频：控制栏影响较小
          if (controlBarHeight > 40) {
            bottomOffset = Math.max(bottomOffset, controlBarHeight + 20);
          }
        } else {
          // 大视频：标准控制栏处理
          if (controlBarHeight > 60) {
            bottomOffset = Math.max(bottomOffset, controlBarHeight + 30);
          }
        }
        
        this.logger.debug('🎯 窗口模式位置计算', {
          videoSize: `${videoRect.width}x${videoRect.height}`,
          videoCategory: videoRect.width <= 700 ? '小视频' : (videoRect.width <= 1000 ? '中等视频' : '大视频'),
          videoBottom,
          playerBottom,
          controlBarHeight,
          calculatedBottomOffset: bottomOffset,
          finalBottomOffset: bottomOffset
        });
      } else {
        this.logger.debug('🎯 窗口模式位置计算（无播放器容器）', {
          videoSize: `${videoRect.width}x${videoRect.height}`,
          videoCategory: videoRect.width <= 700 ? '小视频' : (videoRect.width <= 1000 ? '中等视频' : '大视频'),
          bottomOffset
        });
      }
    }
    
    // 检查是否有原生字幕显示（仅用于微调位置，避免重叠）
    const nativeSubtitle = document.querySelector('.ytp-caption-window-container, .caption-window, .ytp-caption-segment');
    if (nativeSubtitle) {
      const nativeRect = nativeSubtitle.getBoundingClientRect();
      if (nativeRect.bottom > 0 && nativeRect.bottom <= videoRect.bottom) {
        // 根据视频大小调整原生字幕避让空间
        const avoidanceSpace = videoRect.width <= 700 ? 30 : 50; // 小视频用较小的避让空间
        const nativeOffset = videoRect.bottom - nativeRect.bottom + avoidanceSpace;
        bottomOffset = Math.max(bottomOffset, nativeOffset);
        this.logger.debug('🎯 调整位置避开原生字幕', {
          videoCategory: videoRect.width <= 700 ? '小视频' : '大视频',
          calculatedOffset: bottomOffset,
          nativeOffset: nativeOffset,
          avoidanceSpace: avoidanceSpace,
          finalOffset: bottomOffset
        });
      }
    }

    // 计算相对于视频中心的位置
    const videoCenterX = videoRect.left + videoRect.width / 2;
    const videoBottom = videoRect.bottom;
    
    // 根据全屏状态使用不同的高度计算
    const screenHeight = isFullscreen ? screen.height : window.innerHeight;
    const finalBottom = screenHeight - videoBottom + bottomOffset;
    
    this.logger.info('🎯 字幕位置计算结果', {
      mode: isFullscreen ? '全屏' : '窗口',
      videoSize: `${videoRect.width}x${videoRect.height}`,
      videoPosition: {
        top: videoRect.top,
        bottom: videoBottom,
        left: videoRect.left,
        right: videoRect.right
      },
      playerRect: playerRect ? {
        bottom: playerRect.bottom,
        height: playerRect.height
      } : null,
      calculation: {
        bottomOffset: bottomOffset,
        windowHeight: window.innerHeight,
        screenHeight: screenHeight,
        finalBottom: finalBottom,
        subtitleY: finalBottom
      }
    });
    
    return {
      left: `${videoCenterX}px`,
      top: 'auto', 
      bottom: `${finalBottom}px`,
      transform: 'translateX(-50%)',
      isCustom: false,
      videoWidth: videoRect.width,
      videoHeight: videoRect.height,
      isFullscreen: isFullscreen
    };
  }

  // 检查是否为全屏模式
  isFullscreen() {
    const hasFullscreenElement = document.fullscreenElement !== null;
    const hasFullscreenClass = document.querySelector('.html5-video-player.ytp-fullscreen') !== null;
    const result = hasFullscreenElement || hasFullscreenClass;
    
    // 详细的全屏检测日志
    this.logger.debug('🔍 全屏状态检测', {
      fullscreenElement: !!document.fullscreenElement,
      fullscreenClass: hasFullscreenClass,
      finalResult: result
    });
    
    return result;
  }

  // 根据视频尺寸计算字体大小（优化版，更大更协调）
  calculateFontSize(videoWidth) {
    // 检查是否为全屏模式
    const isFullscreen = document.fullscreenElement !== null || 
                         document.querySelector('.html5-video-player.ytp-fullscreen') !== null;
    
    let fontSize;
    
    if (isFullscreen) {
      // 全屏模式：字体适中大小
      if (videoWidth >= 1400) {
        fontSize = 26; // 超大屏幕：26px
      } else if (videoWidth >= 1200) {
        fontSize = 24; // 大屏幕：24px
      } else if (videoWidth >= 800) {
        fontSize = 22; // 中等屏幕：22px
      } else {
        fontSize = 20; // 小屏幕：20px
      }
    } else {
      // 窗口模式：根据视频大小合理调整
      if (videoWidth >= 1200) {
        fontSize = 22; // 大屏幕：22px
      } else if (videoWidth >= 800) {
        fontSize = 20; // 中等屏幕：20px
      } else if (videoWidth >= 600) {
        fontSize = 18; // 小屏幕：18px
      } else {
        fontSize = 16; // 很小屏幕：16px
      }
    }
    
    this.logger.debug('📏 字体大小计算', {
      videoWidth,
      isFullscreen,
      fontSize,
      fullscreenElement: document.fullscreenElement,
      hasFullscreenClass: document.querySelector('.html5-video-player.ytp-fullscreen') !== null
    });
    
    return { 
      original: fontSize,
      translated: fontSize // 中英文使用相同大小
    };
  }

  // 获取默认位置
  getDefaultPosition() {
    const video = document.querySelector('video');
    if (video) {
      const videoRect = video.getBoundingClientRect();
      
      // 确保视频矩形有效
      if (videoRect.width > 0 && videoRect.height > 0) {
        const videoCenterX = videoRect.left + videoRect.width / 2;
        const videoBottom = videoRect.bottom;
        const isFullscreen = this.isFullscreen();
        
        // 使用与主要位置算法相同的逻辑
        let bottomOffset;
        if (isFullscreen) {
          bottomOffset = Math.max(40, videoRect.height * 0.08);
        } else {
          // 根据视频大小调整
          if (videoRect.width <= 700) {
            bottomOffset = Math.max(40, Math.min(80, videoRect.height * 0.12));
          } else if (videoRect.width <= 1000) {
            bottomOffset = Math.max(60, Math.min(100, videoRect.height * 0.13));
          } else {
            bottomOffset = Math.max(80, Math.min(120, videoRect.height * 0.15));
          }
        }
        
        return {
          left: `${videoCenterX}px`,
          top: 'auto',
          bottom: `${(isFullscreen ? screen.height : window.innerHeight) - videoBottom + bottomOffset}px`,
          transform: 'translateX(-50%)',
          isCustom: false,
          videoWidth: videoRect.width,
          videoHeight: videoRect.height,
          isFullscreen: isFullscreen
        };
      }
    }
    
    // 如果没有有效的视频，使用安全的默认位置
    return {
      bottom: '120px',
      left: '50%',
      top: 'auto',
      transform: 'translateX(-50%)',
      isCustom: false,
      videoWidth: 800,
      videoHeight: 450,
      isFullscreen: false
    };
  }

  // 更新字幕位置和样式
  updatePosition() {
    if (!this.container) return;

    const position = this.getYouTubeSubtitlePosition();
    const fontSize = this.calculateFontSize(position.videoWidth);
    
    // 应用位置样式
    this.container.style.position = 'fixed';
    this.container.style.bottom = position.bottom;
    this.container.style.left = position.left;
    this.container.style.top = position.top;
    this.container.style.transform = position.transform;
    this.container.style.zIndex = '999999999';
    this.container.style.pointerEvents = 'auto'; // 允许拖拽交互
    
    // 获取字幕内容元素
    const originalElement = this.container.querySelector('.original-subtitle');
    const translatedElement = this.container.querySelector('.translated-subtitle');
    
    if (originalElement) {
      // ASS样式：英文字幕（青色，Noto Serif，粗体）
      originalElement.style.color = '#00FFFF';
      // 多重方式确保字体大小生效
      originalElement.style.fontSize = `${fontSize.original}px`;
      originalElement.style.setProperty('font-size', `${fontSize.original}px`, 'important');
      originalElement.setAttribute('style', originalElement.getAttribute('style') + `; font-size: ${fontSize.original}px !important;`);
      originalElement.style.fontFamily = '"Noto Serif", "YouTube Noto", sans-serif';
      originalElement.style.fontWeight = '900'; // 最粗字体
      originalElement.style.lineHeight = '1.4';
      originalElement.style.textShadow = '2px 2px 6px rgba(0, 0, 0, 0.9), 1px 1px 2px rgba(0, 0, 0, 0.8)';
      originalElement.style.marginBottom = '4px';
      originalElement.style.textAlign = 'center';
      originalElement.style.pointerEvents = 'none'; // 文字不干扰拖拽
      // 独立背景样式 - 根据文字宽度自适应
      originalElement.style.background = originalElement.textContent.trim() ? 'rgba(0, 0, 0, 0.75)' : 'transparent';
      originalElement.style.borderRadius = '4px';
      originalElement.style.padding = originalElement.textContent.trim() ? '3px 8px' : '0';
      originalElement.style.display = 'inline-block';
      originalElement.style.maxWidth = 'fit-content';
      originalElement.style.wordWrap = 'break-word';
      originalElement.style.whiteSpace = 'pre-wrap';
      originalElement.style.margin = '0 auto 4px auto';
      
      // 延迟检查字体大小是否真正应用
      setTimeout(() => {
        const computedSize = window.getComputedStyle(originalElement).fontSize;
        this.logger.debug('📝 英文字幕样式应用', {
          setFontSize: `${fontSize.original}px`,
          computedFontSize: computedSize,
          textContent: originalElement.textContent.substring(0, 30),
          styleAttribute: originalElement.getAttribute('style')
        });
        
        // 如果计算出的字体大小不匹配，强制重新设置
        if (computedSize !== `${fontSize.original}px`) {
          this.logger.warn('⚠️ 字体大小未正确应用，强制重设', {
            expected: `${fontSize.original}px`,
            actual: computedSize
          });
          originalElement.style.cssText += `; font-size: ${fontSize.original}px !important;`;
        }
      }, 100);
    }

    if (translatedElement) {
      // ASS样式：翻译字幕（绿色，中文字体，粗体）
      translatedElement.style.color = '#00FF00';
      // 多重方式确保字体大小生效
      translatedElement.style.fontSize = `${fontSize.translated}px`;
      translatedElement.style.setProperty('font-size', `${fontSize.translated}px`, 'important');
      translatedElement.setAttribute('style', translatedElement.getAttribute('style') + `; font-size: ${fontSize.translated}px !important;`);
      // 设置中文字体，优先使用您安装的宋体黑体
      translatedElement.style.fontFamily = '"Songti SC Black", "Songti SC", "STSongti-SC-Black", "宋体-简", "SimSun", "宋体", "黑体", "SimHei", "Heiti SC", "Microsoft YaHei", "微软雅黑", "Noto Sans CJK SC", sans-serif';
      // 强制应用字体设置
      translatedElement.style.setProperty('font-family', '"Songti SC Black", "Songti SC", "STSongti-SC-Black", "宋体-简", "SimSun", "宋体", "黑体", "SimHei", "Heiti SC", "Microsoft YaHei", "微软雅黑", "Noto Sans CJK SC", sans-serif', 'important');
      translatedElement.style.fontWeight = '900'; // 最粗字体
      translatedElement.style.fontStretch = 'condensed'; // 字体拉伸
      // 通过transform创造人工粗体效果
      translatedElement.style.filter = 'contrast(1.2) brightness(1.1)';
      // 使用强制粗体属性
      translatedElement.style.setProperty('font-weight', 'bolder', 'important');
      // 添加文字描边效果（webkit引擎支持）
      translatedElement.style.webkitTextStroke = '0.5px rgba(0, 0, 0, 0.8)';
      translatedElement.style.textStroke = '0.5px rgba(0, 0, 0, 0.8)';
      translatedElement.style.lineHeight = '1.4';
      // 增强文字阴影效果，创造更粗重的视觉效果
      translatedElement.style.textShadow = `
        2px 2px 6px rgba(0, 0, 0, 0.9), 
        1px 1px 2px rgba(0, 0, 0, 0.8),
        -1px -1px 2px rgba(0, 0, 0, 0.5),
        1px -1px 2px rgba(0, 0, 0, 0.5),
        -1px 1px 2px rgba(0, 0, 0, 0.5),
        0 0 3px rgba(0, 0, 0, 0.7)
      `;
      translatedElement.style.textAlign = 'center';
      translatedElement.style.pointerEvents = 'none'; // 文字不干扰拖拽
      // 独立背景样式 - 根据文字宽度自适应
      translatedElement.style.background = translatedElement.textContent.trim() && translatedElement.textContent.trim() !== '翻译中...' ? 'rgba(0, 0, 0, 0.75)' : 'transparent';
      translatedElement.style.borderRadius = '4px';
      translatedElement.style.padding = translatedElement.textContent.trim() && translatedElement.textContent.trim() !== '翻译中...' ? '3px 8px' : '0';
      translatedElement.style.display = 'inline-block';
      translatedElement.style.maxWidth = 'fit-content';
      translatedElement.style.wordWrap = 'break-word';
      translatedElement.style.whiteSpace = 'pre-wrap';
      translatedElement.style.margin = '0 auto';
      
      // 延迟检查字体大小是否真正应用
      setTimeout(() => {
        const computedSize = window.getComputedStyle(translatedElement).fontSize;
        const computedFamily = window.getComputedStyle(translatedElement).fontFamily;
        
        // 检测字体是否被正确应用
        const targetFonts = ['Songti SC Black', 'Songti SC', 'STSongti-SC-Black', '宋体-简', 'SimSun', '宋体', '黑体', 'SimHei', 'Heiti SC'];
        const isTargetFontApplied = targetFonts.some(font => computedFamily.includes(font));
        
        this.logger.debug('📝 中文字幕样式应用', {
          setFontSize: `${fontSize.translated}px`,
          computedFontSize: computedSize,
          setFontFamily: '"Songti SC Black", "Songti SC", "STSongti-SC-Black", "宋体-简", "SimSun", "宋体", "黑体", "SimHei", "Heiti SC", "Microsoft YaHei", "微软雅黑", "Noto Sans CJK SC", sans-serif',
          computedFontFamily: computedFamily,
          isTargetFontApplied: isTargetFontApplied,
          detectedFont: computedFamily.split(',')[0].replace(/['"]/g, ''),
          textContent: translatedElement.textContent.substring(0, 30),
          styleAttribute: translatedElement.getAttribute('style')
        });
        
        // 如果计算出的字体大小不匹配，强制重新设置
        if (computedSize !== `${fontSize.translated}px`) {
          this.logger.warn('⚠️ 中文字体大小未正确应用，强制重设', {
            expected: `${fontSize.translated}px`,
            actual: computedSize
          });
          translatedElement.style.cssText += `; font-size: ${fontSize.translated}px !important;`;
        }
      }, 100);
    }

    // 应用容器样式 - 移除背景，让每行文字有独立背景
    const subtitleContainer = this.container.querySelector('.subtitle-container');
    if (subtitleContainer) {
      subtitleContainer.style.background = 'transparent'; // 移除容器背景
      subtitleContainer.style.borderRadius = '0';
      subtitleContainer.style.padding = '8px 0'; // 仅保持上下间距
      subtitleContainer.style.maxWidth = position.isFullscreen ? '90vw' : '80vw';
      subtitleContainer.style.boxShadow = 'none'; // 移除容器阴影
      subtitleContainer.style.position = 'relative';
      subtitleContainer.style.display = 'flex';
      subtitleContainer.style.flexDirection = 'column';
      subtitleContainer.style.alignItems = 'center';
      subtitleContainer.style.gap = '4px'; // 英文和中文之间的间距
      subtitleContainer.style.cursor = 'ns-resize'; // 整个区域显示拖拽光标
    }

    // 记录位置变化
    if (!this.lastPosition || 
        this.lastPosition.bottom !== position.bottom || 
        this.lastPosition.videoWidth !== position.videoWidth ||
        this.lastPosition.isCustom !== position.isCustom) {
      
      this.logger.debug('🎯 字幕位置已更新', {
        position: position.isCustom ? 'custom' : 'auto',
        fontSize: fontSize,
        videoSize: `${position.videoWidth}x${position.videoHeight}`,
        isFullscreen: position.isFullscreen
      });
      
      this.lastPosition = position;
    }
  }

  // 添加双击重置功能
  addDoubleClickReset() {
    if (!this.container) return;

    this.container.addEventListener('dblclick', (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.resetToDefault();
      
      // 显示重置提示
      this.showResetNotification();
    });
  }

  // 显示重置通知
  showResetNotification() {
    const notification = document.createElement('div');
    notification.textContent = '字幕位置已重置';
    notification.style.cssText = `
      position: fixed;
      top: 50px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(76, 175, 80, 0.9);
      color: white;
      padding: 8px 16px;
      border-radius: 4px;
      z-index: 1000000001;
      font-size: 14px;
      font-family: Arial, sans-serif;
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
      notification.remove();
    }, 2000);
  }

  // 清理资源
  cleanup() {
    if (this.dragIndicator) {
      this.dragIndicator.remove();
    }
    this.container = null;
    this.logger = null;
    this.lastPosition = null;
    this.customPosition = null;
  }
}

// 初始化翻译器
const initTranslator = () => {
  try {
    console.log('🎬 开始初始化YouTube双语字幕翻译器...');
    new YouTubeSubtitleTranslator();
    console.log('✅ 翻译器初始化完成');
  } catch (error) {
    console.error('❌ 翻译器初始化失败:', error);
    console.error('错误堆栈:', error.stack);
  }
};

console.log('📜 YouTube双语字幕脚本已加载');

if (document.readyState === 'loading') {
  console.log('⏳ 等待DOM加载完成...');
  document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ DOM加载完成，1秒后初始化翻译器');
    setTimeout(initTranslator, 1000);
  });
} else {
  console.log('✅ DOM已就绪，1秒后初始化翻译器');
  setTimeout(initTranslator, 1000);
}