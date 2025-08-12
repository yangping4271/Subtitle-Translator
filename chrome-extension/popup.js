// YouTube双语字幕翻译器 - Popup Script
class PopupController {
  constructor() {
    this.backendUrl = 'http://127.0.0.1:9009';
    this.currentJobId = null;
    this.isTranslating = false;
    this.currentUrl = '';
    this.pollingInterval = null;
    
    this.elements = {
      connectionStatus: document.getElementById('connectionStatus'),
      videoInfo: document.getElementById('videoInfo'),
      videoTitle: document.getElementById('videoTitle'),
      videoUrl: document.getElementById('videoUrl'),
      targetLang: document.getElementById('targetLang'),
      startButton: document.getElementById('startTranslation'),
      stopButton: document.getElementById('stopTranslation'),
      progress: document.getElementById('progress'),
      progressText: document.getElementById('progressText'),
      status: document.getElementById('status')
    };
    
    this.init();
  }
  
  async init() {
    // 加载保存的语言设置
    this.loadSettings();
    
    // 绑定事件
    this.bindEvents();
    
    // 检查后台服务连接
    await this.checkBackendConnection();
    
    // 获取当前标签页信息
    await this.getCurrentTab();
  }
  
  loadSettings() {
    chrome.storage.local.get(['targetLang'], (result) => {
      if (result.targetLang) {
        this.elements.targetLang.value = result.targetLang;
      }
    });
  }
  
  saveSettings() {
    chrome.storage.local.set({
      targetLang: this.elements.targetLang.value
    });
  }
  
  bindEvents() {
    this.elements.targetLang.addEventListener('change', () => {
      this.saveSettings();
    });
    
    this.elements.startButton.addEventListener('click', () => {
      this.startTranslation();
    });
    
    this.elements.stopButton.addEventListener('click', () => {
      this.stopTranslation();
    });
  }
  
  async checkBackendConnection() {
    try {
      this.updateConnectionStatus('checking', '正在连接后台服务...');
      
      const response = await fetch(`${this.backendUrl}/health`);
      const data = await response.json();
      
      if (response.ok && data.status === 'ok') {
        this.updateConnectionStatus('connected', '后台服务已连接');
        this.elements.startButton.disabled = false;
      } else {
        throw new Error('服务不可用');
      }
    } catch (error) {
      this.updateConnectionStatus('disconnected', '后台服务未启动');
      this.showStatus('error', '请确保后台服务正在运行 (端口 9009)');
      this.elements.startButton.disabled = true;
    }
  }
  
  async getCurrentTab() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (tab && tab.url.includes('youtube.com/watch')) {
        this.currentUrl = tab.url;
        
        // 提取视频信息
        const videoId = this.extractVideoId(tab.url);
        if (videoId) {
          this.elements.videoTitle.textContent = tab.title || '当前YouTube视频';
          this.elements.videoUrl.textContent = `视频ID: ${videoId}`;
          this.elements.videoInfo.classList.add('show');
          
          // 检查是否有缓存
          await this.checkVideoCache(videoId);
        }
      } else {
        this.elements.videoTitle.textContent = '未检测到YouTube视频';
        this.elements.videoUrl.textContent = '请在YouTube视频页面使用此扩展';
        this.elements.videoInfo.classList.add('show');
        this.elements.startButton.disabled = true;
        this.showStatus('warning', '请在YouTube视频页面使用此扩展');
      }
    } catch (error) {
      console.error('获取当前标签页失败:', error);
      this.showStatus('error', '获取当前页面信息失败');
    }
  }
  
  extractVideoId(url) {
    const match = url.match(/[?&]v=([^&]+)/);
    return match ? match[1] : null;
  }
  
  async checkVideoCache(videoId) {
    try {
      const response = await fetch(`${this.backendUrl}/cache/check/${videoId}`);
      const data = await response.json();
      
      if (data.has_translation_cache) {
        this.showStatus('info', '检测到翻译缓存，处理将非常快速');
      } else if (data.has_audio_cache) {
        this.showStatus('info', '检测到音频缓存，跳过下载步骤');
      }
    } catch (error) {
      console.error('检查缓存失败:', error);
    }
  }
  
  async startTranslation() {
    if (this.isTranslating || !this.currentUrl) {
      return;
    }
    
    this.isTranslating = true;
    this.elements.startButton.disabled = true;
    this.elements.stopButton.style.display = 'block';
    this.elements.stopButton.disabled = false;
    this.elements.progress.classList.add('show');
    
    try {
      // 发起翻译请求
      const response = await fetch(`${this.backendUrl}/translate_youtube`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: this.currentUrl,
          target_lang: this.elements.targetLang.value
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.job_id) {
        this.currentJobId = data.job_id;
        this.updateProgress('已提交翻译任务，开始处理...');
        this.showStatus('info', '翻译任务已开始');
        
        // 开始轮询状态
        this.startStatusPolling();
      } else {
        throw new Error(data.error || '启动翻译失败');
      }
    } catch (error) {
      console.error('启动翻译失败:', error);
      this.showStatus('error', `启动翻译失败: ${error.message}`);
      this.resetTranslationState();
    }
  }
  
  stopTranslation() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    
    this.currentJobId = null;
    this.resetTranslationState();
    this.showStatus('warning', '翻译已停止');
  }
  
  resetTranslationState() {
    this.isTranslating = false;
    this.elements.startButton.disabled = false;
    this.elements.stopButton.style.display = 'none';
    this.elements.progress.classList.remove('show');
  }
  
  startStatusPolling() {
    this.pollingInterval = setInterval(async () => {
      await this.checkTranslationStatus();
    }, 2000); // 每2秒检查一次
  }
  
  async checkTranslationStatus() {
    if (!this.currentJobId) {
      return;
    }
    
    try {
      const response = await fetch(`${this.backendUrl}/job_status/${this.currentJobId}`);
      const data = await response.json();
      
      if (response.ok) {
        this.updateProgress(this.formatProgressMessage(data.progress));
        
        if (data.status === 'done') {
          // 翻译完成
          clearInterval(this.pollingInterval);
          this.pollingInterval = null;
          
          this.updateProgress(`翻译完成！生成了 ${data.events_count} 个字幕段`);
          this.showStatus('success', '翻译完成，字幕已自动显示在视频上');
          this.resetTranslationState();
          
          // 通知content script刷新字幕
          this.notifyContentScript();
          
        } else if (data.status === 'error') {
          // 翻译失败
          clearInterval(this.pollingInterval);
          this.pollingInterval = null;
          
          this.showStatus('error', `翻译失败: ${data.error}`);
          this.resetTranslationState();
        }
      }
    } catch (error) {
      console.error('检查翻译状态失败:', error);
    }
  }
  
  formatProgressMessage(progress) {
    if (!progress) return '处理中...';
    
    const stage = progress.stage;
    const message = progress.message;
    
    const stageNames = {
      'initializing': '初始化',
      'downloading': '下载中',
      'downloading_audio': '下载音频',
      'downloading_subtitle': '下载字幕',
      'cache_hit': '缓存命中',
      'transcribing': '转录翻译中',
      'completed': '完成'
    };
    
    const stageName = stageNames[stage] || stage;
    return `${stageName}: ${message}`;
  }
  
  async notifyContentScript() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab) {
        chrome.tabs.sendMessage(tab.id, {
          type: 'TRANSLATION_COMPLETED',
          jobId: this.currentJobId
        });
      }
    } catch (error) {
      console.error('通知content script失败:', error);
    }
  }
  
  updateConnectionStatus(status, message) {
    const statusElement = this.elements.connectionStatus;
    const dot = statusElement.querySelector('.status-dot');
    const text = statusElement.querySelector('span:last-child');
    
    // 移除所有状态类
    statusElement.className = 'connection-status';
    dot.className = 'status-dot';
    
    // 添加新状态
    statusElement.classList.add(status);
    if (status === 'connected') {
      dot.classList.add('green');
    } else if (status === 'disconnected') {
      dot.classList.add('red');
    } else {
      dot.classList.add('orange');
    }
    
    text.textContent = message;
  }
  
  updateProgress(message) {
    this.elements.progressText.textContent = message;
  }
  
  showStatus(type, message) {
    const statusElement = this.elements.status;
    statusElement.className = `status ${type}`;
    statusElement.textContent = message;
    statusElement.style.display = 'block';
    
    // 自动隐藏成功消息
    if (type === 'success' || type === 'info') {
      setTimeout(() => {
        statusElement.style.display = 'none';
      }, 5000);
    }
  }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
  new PopupController();
});