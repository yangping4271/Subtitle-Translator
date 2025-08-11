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
        this.showTemporaryMessage('🌐 YouTube双语字幕已启动 (Ctrl+D显示详细状态)', 3000);
    
    // 设置全局快捷键
    this.setupGlobalKeyboardShortcuts();
    
    // 立即检查当前视频
    this.checkVideoChange();
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
      // 这些主机默认通过后台代理，避免CORS
      return host === 'ai-proxy.chatwise.app' || host === 'openrouter.ai' || host === '127.0.0.1' || host === 'localhost';
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
      
      // Ctrl+D: 切换调试面板显示/隐藏
      if (event.ctrlKey && event.key === 'd') {
        event.preventDefault();
        this.toggleDebugPanel();
      }
    });
    
    this.logger.info('⌨️ 全局快捷键已设置: Ctrl+L=导出日志, Ctrl+D=显示/隐藏调试面板');
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
        this.showStatusInfo('调试面板已显示 (按 Ctrl+D 再次隐藏)');
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
    
    // 直接插入到body确保可见
    document.body.appendChild(this.translationContainer);
    
    // 初始化智能定位系统
    this.initializeSmartPositioning();
    
    this.logger.info('✅ 双语字幕容器创建成功');
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
      this.positionManager.updatePosition();
    }, 100);
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
        this.positionManager?.updatePosition();
      }, 100); // 延迟以确保全屏状态已完全应用
    });

    // 监听YouTube特定的全屏事件
    const player = document.querySelector('.html5-video-player');
    if (player) {
      const observer = new MutationObserver(() => {
        setTimeout(() => {
          this.positionManager?.updatePosition();
        }, 100);
      });
      
      observer.observe(player, {
        attributes: true,
        attributeFilter: ['class']
      });
    }
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
      
      // 根据内容调整显示
      if (displayOriginal) {
        originalElement.style.display = 'block';
      } else {
        originalElement.style.display = 'none';
      }
      
      if (displayTranslated && displayTranslated !== '翻译中...') {
        translatedElement.style.display = 'block';
      } else {
        translatedElement.style.display = 'none';
      }
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

// 初始化
const initTranslator = () => {
  window.debugLogger?.info('🎬 开始初始化翻译器...');
  new YouTubeSubtitleTranslator();
};

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

    // 创建拖拽手柄
    this.createDragHandle();
    
    // 绑定拖拽事件
    this.bindDragEvents();
  }

  // 创建拖拽手柄
  createDragHandle() {
    // 检查是否已经有拖拽手柄
    if (this.container.querySelector('.drag-handle')) return;

    const dragHandle = document.createElement('div');
    dragHandle.className = 'drag-handle';
    dragHandle.innerHTML = '⋮⋮'; // 双竖点图标，表示垂直拖拽
    dragHandle.title = '拖拽调整字幕位置（上下移动）\n双击重置到默认位置';
    dragHandle.style.cssText = `
      position: absolute;
      top: -15px;
      right: -15px;
      width: 20px;
      height: 20px;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 50%;
      cursor: ns-resize; /* 上下拖拽光标 */
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      color: #333;
      opacity: 0;
      transition: opacity 0.3s ease;
      z-index: 1000000000;
      user-select: none;
      pointer-events: auto;
      border: 2px solid rgba(0, 255, 255, 0.5); /* 青色边框，呼应字幕颜色 */
    `;

    // 鼠标悬停显示拖拽手柄
    this.container.addEventListener('mouseenter', () => {
      dragHandle.style.opacity = '1';
    });

    this.container.addEventListener('mouseleave', () => {
      if (!this.isDragging) {
        dragHandle.style.opacity = '0';
      }
    });

    // 插入拖拽手柄
    const subtitleContainer = this.container.querySelector('.subtitle-container');
    if (subtitleContainer) {
      subtitleContainer.style.position = 'relative';
      subtitleContainer.appendChild(dragHandle);
    }

    this.dragHandle = dragHandle;
  }

  // 绑定拖拽事件
  bindDragEvents() {
    if (!this.dragHandle) return;

    // 鼠标按下开始拖拽
    this.dragHandle.addEventListener('mousedown', (e) => {
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
    this.dragHandle.addEventListener('touchstart', (e) => {
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
    this.dragHandle.style.opacity = '1';
    
    const containerRect = this.container.getBoundingClientRect();
    this.dragOffset = {
      x: event.clientX - containerRect.left,
      y: event.clientY - containerRect.top
    };

    // 添加拖拽时的视觉效果
    this.container.style.transform = 'scale(1.05)';
    this.container.style.transition = 'transform 0.1s ease';
    
    this.logger.debug('🎯 开始拖拽字幕');
  }

  // 处理拖拽移动（仅垂直方向，保持水平居中）
  handleDragging(event) {
    if (!this.isDragging) return;

    const y = event.clientY - this.dragOffset.y;

    // 获取视频容器信息来保持居中
    const video = document.querySelector('video');
    const videoRect = video ? video.getBoundingClientRect() : null;
    
    // 限制拖拽边界（仅垂直方向）
    const bounds = this.calculateVerticalDragBounds();
    const constrainedY = Math.max(bounds.minY, Math.min(bounds.maxY, y));

    // 计算相对于视频底部的距离
    let bottomOffset = 100; // 默认值
    if (videoRect) {
      bottomOffset = window.innerHeight - constrainedY - this.container.offsetHeight;
      // 确保不会拖到视频外面
      const minBottomOffset = 20; // 最小距离视频底部20px
      const maxBottomOffset = videoRect.height - 50; // 最大不超过视频高度减去字幕高度
      bottomOffset = Math.max(minBottomOffset, Math.min(maxBottomOffset, bottomOffset));
    }

    // 应用新位置：保持水平居中，只改变垂直位置
    this.container.style.left = '50%';
    this.container.style.top = 'auto';
    this.container.style.bottom = `${bottomOffset}px`;
    this.container.style.transform = 'translateX(-50%) scale(1.05)';

    // 保存自定义位置
    this.customPosition = {
      bottomOffset: bottomOffset,
      isCustom: true
    };
  }

  // 结束拖拽
  stopDragging(event) {
    if (!this.isDragging) return;

    this.isDragging = false;
    this.dragHandle.style.opacity = '0';
    
    // 恢复缩放效果，保持居中
    this.container.style.transform = 'translateX(-50%) scale(1)';
    this.container.style.transition = 'transform 0.2s ease';
    
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
    // 如果有自定义位置，优先使用
    const savedPosition = this.loadCustomPosition();
    if (savedPosition && savedPosition.isCustom && savedPosition.bottomOffset !== undefined) {
      return {
        left: '50%',
        top: 'auto',
        bottom: `${savedPosition.bottomOffset}px`,
        transform: 'translateX(-50%)',
        isCustom: true,
        videoWidth: 800,
        videoHeight: 450,
        isFullscreen: this.isFullscreen()
      };
    }

    // 查找YouTube原生字幕容器
    const nativeSubtitle = document.querySelector('.ytp-caption-window-container, .caption-window, .ytp-caption-segment');
    const video = document.querySelector('video');
    const player = document.querySelector('.html5-video-player, #player-container');

    if (!video || !player) {
      return this.getDefaultPosition();
    }

    const videoRect = video.getBoundingClientRect();
    const playerRect = player.getBoundingClientRect();
    
    // 基础位置：视频底部稍上方
    let baseBottom = 80;
    
    // 如果有原生字幕，在其上方留出空间
    if (nativeSubtitle) {
      const nativeRect = nativeSubtitle.getBoundingClientRect();
      if (nativeRect.bottom > 0) {
        const videoBottom = videoRect.bottom;
        const nativeTop = nativeRect.top;
        baseBottom = videoBottom - nativeTop + 60; // 在原生字幕上方60px
      }
    }

    return {
      bottom: `${baseBottom}px`,
      left: '50%',
      top: 'auto',
      transform: 'translateX(-50%)',
      isCustom: false,
      videoWidth: videoRect.width,
      videoHeight: videoRect.height,
      isFullscreen: this.isFullscreen()
    };
  }

  // 检查是否为全屏模式
  isFullscreen() {
    return document.fullscreenElement !== null || 
           document.querySelector('.html5-video-player.ytp-fullscreen') !== null;
  }

  // 根据视频尺寸计算字体大小
  calculateFontSize(videoWidth) {
    if (videoWidth >= 1200) {
      return { original: 20, translated: 18 };
    } else if (videoWidth >= 800) {
      return { original: 18, translated: 16 };
    } else if (videoWidth >= 600) {
      return { original: 16, translated: 14 };
    } else {
      return { original: 14, translated: 12 };
    }
  }

  // 获取默认位置
  getDefaultPosition() {
    return {
      bottom: '100px',
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
      // ASS样式：英文字幕（青色）
      originalElement.style.color = '#00FFFF';
      originalElement.style.fontSize = `${fontSize.original}px`;
      originalElement.style.fontFamily = '"Noto Serif", "YouTube Noto", sans-serif';
      originalElement.style.fontWeight = '500';
      originalElement.style.lineHeight = '1.4';
      originalElement.style.textShadow = '2px 2px 4px rgba(0, 0, 0, 0.9)';
      originalElement.style.marginBottom = '6px';
      originalElement.style.textAlign = 'center';
      originalElement.style.pointerEvents = 'none'; // 文字不干扰拖拽
    }

    if (translatedElement) {
      // ASS样式：翻译字幕（绿色）
      translatedElement.style.color = '#00FF00';
      translatedElement.style.fontSize = `${fontSize.translated}px`;
      translatedElement.style.fontFamily = '"Noto Sans CJK SC", "YouTube Noto", sans-serif';
      translatedElement.style.fontWeight = '600';
      translatedElement.style.lineHeight = '1.4';
      translatedElement.style.textShadow = '2px 2px 4px rgba(0, 0, 0, 0.9)';
      translatedElement.style.textAlign = 'center';
      translatedElement.style.pointerEvents = 'none'; // 文字不干扰拖拽
    }

    // 应用容器样式
    const subtitleContainer = this.container.querySelector('.subtitle-container');
    if (subtitleContainer) {
      subtitleContainer.style.background = 'rgba(0, 0, 0, 0.8)';
      subtitleContainer.style.borderRadius = '4px';
      subtitleContainer.style.padding = '8px 12px';
      subtitleContainer.style.maxWidth = position.isFullscreen ? '90vw' : '80vw';
      subtitleContainer.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.6)';
      subtitleContainer.style.position = 'relative';
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
    if (this.dragHandle) {
      this.dragHandle.remove();
    }
    this.container = null;
    this.logger = null;
    this.lastPosition = null;
    this.customPosition = null;
  }
}

// 初始化翻译器
const initTranslator = () => {
  new YouTubeSubtitleTranslator();
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initTranslator, 1000);
  });
} else {
  setTimeout(initTranslator, 1000);
}