class YouTubeSubtitleOverlay {
  constructor() {
    this.subtitleData = [];
    this.englishSubtitles = [];
    this.chineseSubtitles = [];
    this.currentVideo = null;
    this.overlayElement = null;
    this.isEnabled = false;
    
    // 自动加载相关配置
    this.autoLoadEnabled = false;
    this.serverUrl = 'http://127.0.0.1:8888';
    this.currentVideoId = null;
    this.autoLoadAttempted = false;

    // 独立的语言设置（从统一配置中心加载）
    this.englishSettings = getDefaultEnglishSettings();
    this.chineseSettings = getDefaultChineseSettings();

    // DPR 自动补偿配置（从统一配置中心加载）
    const config = getDefaultConfig();
    this.enableDPRCompensation = config.dpr.enabled;
    this.dprCompensationFactor = this.calculateDPRCompensation();

    this.init();
  }

  /**
   * 计算设备像素比(DPR)补偿系数
   * 在高分辨率屏幕上，相同的 CSS 像素会显得更小，需要适当放大
   *
   * @returns {number} 补偿系数
   *
   * 补偿公式：1 + (DPR - 1) * 0.4
   * - DPR = 1 (普通屏幕): 补偿系数 = 1.0 (不补偿)
   * - DPR = 2 (Retina): 补偿系数 = 1.4 (放大 40%)
   * - DPR = 3 (高端屏): 补偿系数 = 1.8 (放大 80%)
   *
   * 示例效果：
   * - 34px 在 DPR=2 屏幕 → 34 × 1.4 = 47.6px
   * - 32px 在 DPR=2 屏幕 → 32 × 1.4 = 44.8px
   */
  calculateDPRCompensation() {
    const dpr = window.devicePixelRatio || 1;

    if (dpr <= 1) {
      // 普通屏幕，不需要补偿
      return 1.0;
    }

    // 使用公式计算补偿系数
    // 系数 0.4 是经过测试的最佳值，可以根据实际效果调整
    const compensationFactor = 1 + (dpr - 1) * 0.4;

    return compensationFactor;
  }

  init() {
    this.createOverlayElement();
    this.observeVideoChanges();
    this.loadSubtitleData();
    this.bindMessageListener();
  }

  bindMessageListener() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      switch (request.action) {
        case 'toggleSubtitle':
          this.toggleSubtitle(request.enabled);
          break;
        case 'loadSubtitle':
          this.loadNewSubtitle(request.subtitleData);
          break;
        case 'loadBilingualSubtitles':
          this.loadBilingualSubtitles(request.englishSubtitles, request.chineseSubtitles);
          break;
        case 'clearData':
          this.clearSubtitleData();
          break;
        case 'forceReset':
          this.forceReset();
          break;
        case 'updateSettings':
          this.updateLanguageSettings(request.language, request.settings);
          break;
        case 'toggleAutoLoad':
          this.toggleAutoLoad(request.enabled);
          break;
        case 'updateServerUrl':
          this.serverUrl = request.url || 'http://127.0.0.1:8888';
          break;
        case 'getVideoInfo':
          const videoId = this.getVideoId();
          const subtitleLoaded = this.englishSubtitles.length > 0 || this.chineseSubtitles.length > 0;
          sendResponse({ 
            videoId: videoId, 
            subtitleLoaded: subtitleLoaded,
            autoLoadEnabled: this.autoLoadEnabled
          });
          break;
        case 'getSubtitleStatus':
          // 🔧 新增：返回详细的字幕状态信息
          const currentVideoId = this.getVideoId();
          const englishCount = this.englishSubtitles.length;
          const chineseCount = this.chineseSubtitles.length;
          const hasSubtitles = englishCount > 0 || chineseCount > 0;
          
          sendResponse({
            videoId: currentVideoId,
            hasSubtitles: hasSubtitles,
            englishCount: englishCount,
            chineseCount: chineseCount,
            autoLoadEnabled: this.autoLoadEnabled,
            subtitleEnabled: this.isEnabled
          });
          break;
      }
    });
  }

  createOverlayElement() {
    this.overlayElement = document.createElement('div');
    this.overlayElement.id = 'youtube-local-subtitle-overlay';
    this.overlayElement.innerHTML = `
      <div class="subtitle-container">
        <div class="english-wrapper">
          <span class="english-subtitle" id="englishSubtitle"></span>
        </div>
        <div class="chinese-wrapper">
          <span class="chinese-subtitle" id="chineseSubtitle"></span>
        </div>
      </div>
    `;
    this.applyStyles();
  }

  applyStyles() {
    // 主容器样式 - 初始为绝对定位
    const mainStyles = {
      position: 'absolute',
      zIndex: '40', // 在视频上方(1-10)，在控制栏下方(50+)
      display: 'none',
      left: '50%',
      transform: 'translateX(-50%)',
      bottom: '60px',
      pointerEvents: 'none',
      userSelect: 'none'
    };
    Object.assign(this.overlayElement.style, mainStyles);

    // 字幕容器样式
    const container = this.overlayElement.querySelector('.subtitle-container');
    this._applyContainerFlexStyles(container);

    // wrapper 样式 - 控制垂直布局和平衡换行
    const englishWrapper = this.overlayElement.querySelector('.english-wrapper');
    const chineseWrapper = this.overlayElement.querySelector('.chinese-wrapper');

    if (englishWrapper) {
      Object.assign(englishWrapper.style, {
        display: 'block',
        textAlign: 'center',
        width: '100%',
        textWrap: 'balance'
      });
    }

    if (chineseWrapper) {
      Object.assign(chineseWrapper.style, {
        display: 'block',
        textAlign: 'center',
        width: '100%',
        textWrap: 'balance'
      });
    }

    // 应用独立的英文字幕样式
    this.applyLanguageStyles('english');

    // 应用独立的中文字幕样式
    this.applyLanguageStyles('chinese');
  }

  // 🔧 私有方法：应用容器 Flex 布局样式
  _applyContainerFlexStyles(container) {
    if (container) {
      Object.assign(container.style, {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0px',
        width: '100%'
      });
    }
  }

  // 应用独立语言样式
  applyLanguageStyles(language) {
    const settings = language === 'english' ? this.englishSettings : this.chineseSettings;
    const elementId = language === 'english' ? '#englishSubtitle' : '#chineseSubtitle';
    const element = this.overlayElement.querySelector(elementId);

    if (element && settings) {
      // 计算应用 DPR 补偿后的字体大小
      const baseFontSize = settings.fontSize;
      const compensatedFontSize = this.enableDPRCompensation
        ? Math.round(baseFontSize * this.dprCompensationFactor)
        : baseFontSize;

      Object.assign(element.style, {
        fontSize: compensatedFontSize + 'px',
        color: settings.fontColor,
        WebkitTextFillColor: settings.fontColor,
        fontFamily: settings.fontFamily,
        fontWeight: settings.fontWeight,
        WebkitTextStroke: settings.textStroke || 'none',
        paintOrder: 'stroke fill',
        textShadow: settings.textShadow !== 'none' ? settings.textShadow : 'none',
        lineHeight: settings.lineHeight,
        padding: '0 6px',
        borderRadius: '3px',
        display: 'inline',
        textAlign: 'center',
        whiteSpace: 'pre-wrap',
        wordBreak: 'normal',
        textWrap: 'balance',
        boxDecorationBreak: 'clone',
        WebkitBoxDecorationBreak: 'clone',
        maxWidth: '100%',
        boxSizing: 'border-box',
        margin: '0'
      });
    }
  }

  observeVideoChanges() {
    let currentUrl = location.href;
    const observer = new MutationObserver(() => {
      if (location.href !== currentUrl) {
        currentUrl = location.href;
        setTimeout(() => this.onVideoChange(), 1000);
      }
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
    setTimeout(() => this.onVideoChange(), 1000);
  }

  onVideoChange() {
    const video = document.querySelector('video');
    const videoElementChanged = video && video !== this.currentVideo;
    const newVideoId = this.getVideoId();
    const videoIdChanged = newVideoId && newVideoId !== this.currentVideoId;

    // 当 <video> 元素发生变化时，重新绑定监听与插入覆盖层
    if (videoElementChanged) {
      this.currentVideo = video;
      this.setupVideoListeners();
      this.insertOverlayToPage();
      this.setupResizeListener();
    }

    // 当视频ID发生变化（即使video元素未变）或元素变更时，刷新字幕缓存并触发自动加载
    if (videoIdChanged || videoElementChanged) {
      // 避免显示旧字幕
      this.hideSubtitle();

      // 清空旧数据，避免残留
      this.subtitleData = [];
      this.englishSubtitles = [];
      this.chineseSubtitles = [];

      // 根据新的视频ID从存储加载已缓存的字幕
      this.loadSubtitleData();

      // 重置自动加载状态，允许重新拉取
      this.autoLoadAttempted = false;

      // 延迟尝试自动加载，确保页面元素稳定
      setTimeout(() => {
        this.attemptAutoLoad();
      }, 500);
    }
  }

  setupVideoListeners() {
    if (!this.currentVideo) return;
    
    if (this.onTimeUpdate) {
      this.currentVideo.removeEventListener('timeupdate', this.onTimeUpdate);
    }
    if (this.onEnded) {
      this.currentVideo.removeEventListener('ended', this.onEnded);
    }
    
    this.onTimeUpdate = () => {
      if (this.isEnabled) {
        if (this.englishSubtitles.length > 0 || this.chineseSubtitles.length > 0 || this.subtitleData.length > 0) {
          this.updateSubtitle();
        }
      }
    };
    
    // 在一个视频播放结束且自动播放即将切换时，先隐藏当前字幕，避免残留
    this.onEnded = () => {
      this.hideSubtitle();
    };
    
    this.currentVideo.addEventListener('timeupdate', this.onTimeUpdate);
    this.currentVideo.addEventListener('ended', this.onEnded);

    if (this.isEnabled) {
      this.updateSubtitle();
    }
  }

  setupResizeListener() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    if (this.scrollListener) {
      window.removeEventListener('scroll', this.scrollListener);
    }
    if (this.fullscreenListener) {
      document.removeEventListener('fullscreenchange', this.fullscreenListener);
    }
    if (this.resizeWindowListener) {
      window.removeEventListener('resize', this.resizeWindowListener);
    }
    
    // 节流函数，避免频繁重新定位
    const throttleReposition = this.throttle(() => {
      if (this.overlayElement && this.isEnabled) {
        this.repositionSubtitle();
      }
    }, 100); // 100ms节流
    
    this.resizeObserver = new ResizeObserver(() => {
      throttleReposition();
    });
    
    this.scrollListener = () => {
      throttleReposition();
    };
    
    this.fullscreenListener = () => {
      // 全屏切换需要立即响应，稍微延迟确保DOM更新完成
      setTimeout(() => {
        if (this.overlayElement && this.isEnabled) {
          this.repositionSubtitle();
        }
      }, 100);
    };
    
    this.resizeWindowListener = () => {
      throttleReposition();
    };
    
    this.setupYouTubeStateListener();
    
    if (this.currentVideo) {
      this.resizeObserver.observe(this.currentVideo);
    }
    window.addEventListener('scroll', this.scrollListener, { passive: true });
    document.addEventListener('fullscreenchange', this.fullscreenListener);
    window.addEventListener('resize', this.resizeWindowListener, { passive: true });
  }

  // 节流函数
  throttle(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  setupYouTubeStateListener() {
    if (this.youtubeStateObserver) {
      this.youtubeStateObserver.disconnect();
    }
    
    this.youtubeStateObserver = new MutationObserver((mutations) => {
      let needsReposition = false;
      
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && 
            (mutation.attributeName === 'class' || mutation.attributeName === 'theater')) {
          needsReposition = true;
        }
        
        if (mutation.type === 'childList') {
          needsReposition = true;
        }
      });
      
      if (needsReposition && this.overlayElement && this.isEnabled) {
        setTimeout(() => this.repositionSubtitle(), 200);
      }
    });
    
    const targets = [
      document.querySelector('#movie_player'),
      document.querySelector('#masthead-container'),
      document.querySelector('#page-manager'),
      document.body
    ].filter(el => el);
    
    targets.forEach(target => {
      this.youtubeStateObserver.observe(target, {
        attributes: true,
        attributeFilter: ['class', 'theater', 'fullscreen'],
        childList: true,
        subtree: false
      });
    });
  }

  repositionSubtitle() {
    if (!this.overlayElement || !this.currentVideo) return;

    // 🔧 强制重新应用容器、wrapper 和语言样式，确保最新设置生效
    const container = this.overlayElement.querySelector('.subtitle-container');
    this._applyContainerFlexStyles(container);

    // 强制刷新 wrapper 样式
    const englishWrapper = this.overlayElement.querySelector('.english-wrapper');
    const chineseWrapper = this.overlayElement.querySelector('.chinese-wrapper');

    if (englishWrapper) {
      Object.assign(englishWrapper.style, {
        display: 'block',
        textAlign: 'center',
        width: '100%',
        textWrap: 'balance'
      });
    }

    if (chineseWrapper) {
      Object.assign(chineseWrapper.style, {
        display: 'block',
        textAlign: 'center',
        width: '100%',
        textWrap: 'balance'
      });
    }

    this.applyLanguageStyles('english');
    this.applyLanguageStyles('chinese');

    // 若当前无任何字幕文本，保持容器隐藏，避免空容器被误显
    const englishEl = this.overlayElement.querySelector('#englishSubtitle');
    const chineseEl = this.overlayElement.querySelector('#chineseSubtitle');
    const hasText = !!((englishEl && englishEl.textContent && englishEl.textContent.trim()) ||
                       (chineseEl && chineseEl.textContent && chineseEl.textContent.trim()));
    if (!hasText) {
      this.overlayElement.style.display = 'none';
    }
    
    const videoRect = this.currentVideo.getBoundingClientRect();
    const isFullscreen = document.fullscreenElement !== null;
    const isTheaterMode = document.querySelector('.ytp-size-large') !== null;
    const isMiniPlayer = document.querySelector('.ytp-miniplayer-active') !== null;
    
    // 获取视频播放器容器
    const playerContainer = document.querySelector('#movie_player');
    const containerRect = playerContainer ? playerContainer.getBoundingClientRect() : videoRect;
    
    if (isFullscreen) {
      // 全屏模式：居中显示
      this.overlayElement.style.position = 'fixed';
      this.overlayElement.style.left = '50%';
      this.overlayElement.style.transform = 'translateX(-50%)';
      this.overlayElement.style.bottom = '80px';
      this.overlayElement.style.width = '80%';
      this.overlayElement.style.maxWidth = 'none';
      this.overlayElement.style.zIndex = '9999';
    } else if (isMiniPlayer) {
      // 迷你播放器：隐藏字幕
      this.overlayElement.style.display = 'none';
      return;
    } else {
      // 普通和剧场模式：相对于视频容器定位
      this.overlayElement.style.display = 'block';
      this.overlayElement.style.position = 'absolute';
      this.overlayElement.style.zIndex = '40';
      
      // 确保字幕容器相对于播放器定位
      if (playerContainer) {
        // 如果播放器容器存在，使用相对定位
        if (playerContainer.style.position !== 'relative') {
          playerContainer.style.position = 'relative';
        }
        
        // 相对于播放器容器居中
        this.overlayElement.style.left = '50%';
        this.overlayElement.style.transform = 'translateX(-50%)';
        this.overlayElement.style.bottom = isTheaterMode ? '70px' : '60px';
        this.overlayElement.style.width = '80%';
        this.overlayElement.style.maxWidth = 'none';
        
        // 确保字幕容器在播放器内部
        if (!playerContainer.contains(this.overlayElement)) {
          playerContainer.appendChild(this.overlayElement);
        }
      } else {
        // 后备方案：使用fixed定位
        this.overlayElement.style.position = 'fixed';
        this.overlayElement.style.left = '50%';
        this.overlayElement.style.transform = 'translateX(-50%)';
        this.overlayElement.style.bottom = (window.innerHeight - videoRect.bottom + 60) + 'px';
        this.overlayElement.style.width = Math.min(videoRect.width * 0.8, 800) + 'px';
        this.overlayElement.style.maxWidth = 'none';
      }
    }
    
    // 字幕定位已优化
  }

  insertOverlayToPage() {
    const existingOverlay = document.getElementById('youtube-local-subtitle-overlay');
    if (existingOverlay) {
      existingOverlay.remove();
    }

    // 优先插入到播放器容器内
    const moviePlayer = document.querySelector('#movie_player');

    if (moviePlayer) {
      // 设置播放器容器为相对定位，确保字幕绝对定位相对于它
      if (moviePlayer.style.position !== 'relative') {
        moviePlayer.style.position = 'relative';
      }

      moviePlayer.appendChild(this.overlayElement);
    } else {
      // 后备方案：插入到body
      document.body.appendChild(this.overlayElement);
    }

    // 初始定位
    this.repositionSubtitle();
  }

  updateSubtitle() {
    if (!this.currentVideo || !this.isEnabled || !this.overlayElement) return;
    
    const currentTime = this.currentVideo.currentTime;
    
    let englishText = '';
    let chineseText = '';
    
    if (this.englishSubtitles.length > 0) {
      const englishSubtitle = this.findCurrentSubtitle(currentTime, this.englishSubtitles);
      if (englishSubtitle) {
        englishText = englishSubtitle.text;
      }
    }
    
    if (this.chineseSubtitles.length > 0) {
      const chineseSubtitle = this.findCurrentSubtitle(currentTime, this.chineseSubtitles);
      if (chineseSubtitle) {
        chineseText = chineseSubtitle.text;
      }
    }
    
    if (this.englishSubtitles.length === 0 && this.chineseSubtitles.length === 0 && this.subtitleData.length > 0) {
      const currentSubtitle = this.findCurrentSubtitle(currentTime, this.subtitleData);
      if (currentSubtitle) {
        chineseText = currentSubtitle.text;
      }
    }
    
    if (englishText || chineseText) {
      this.showBilingualSubtitle(englishText, chineseText);
    } else {
      this.hideSubtitle();
    }
  }

  showBilingualSubtitle(englishText, chineseText) {
    const englishSubtitle = this.overlayElement.querySelector('#englishSubtitle');
    const chineseSubtitle = this.overlayElement.querySelector('#chineseSubtitle');

    if (englishSubtitle) {
      englishSubtitle.textContent = englishText;
      englishSubtitle.style.display = englishText ? 'inline' : 'none';
    }

    if (chineseSubtitle) {
      chineseSubtitle.textContent = chineseText;
      chineseSubtitle.style.display = chineseText ? 'inline' : 'none';
    }

    this.overlayElement.style.display = 'block';
    this.overlayElement.style.position = 'absolute';
    this.overlayElement.style.zIndex = '40'; // 精确的中等层级
    this.overlayElement.style.visibility = 'visible';
    this.overlayElement.style.opacity = '1';

    this.repositionSubtitle();
  }

  showSubtitle(text) {
    this.showBilingualSubtitle('', text);
  }

  hideSubtitle() {
    if (!this.overlayElement) return;
    // 隐藏容器
    this.overlayElement.style.display = 'none';
    this.overlayElement.style.visibility = 'hidden';
    this.overlayElement.style.opacity = '0';
    // 清空文本，避免下一次被误显示残留内容
    const englishSubtitle = this.overlayElement.querySelector('#englishSubtitle');
    const chineseSubtitle = this.overlayElement.querySelector('#chineseSubtitle');
    if (englishSubtitle) {
      englishSubtitle.textContent = '';
      englishSubtitle.style.display = 'none';
    }
    if (chineseSubtitle) {
      chineseSubtitle.textContent = '';
      chineseSubtitle.style.display = 'none';
    }
  }

  findCurrentSubtitle(currentTime, subtitles = null) {
    const dataToSearch = subtitles || this.subtitleData;
    return dataToSearch.find(subtitle => 
      currentTime >= subtitle.startTime && currentTime <= subtitle.endTime
    );
  }

  toggleSubtitle(enabled) {
    this.isEnabled = enabled;

    if (!enabled) {
      this.hideSubtitle();
    } else {
      // 开启时，如果有真实字幕数据则显示
      if (this.englishSubtitles.length > 0 || this.chineseSubtitles.length > 0 || this.subtitleData.length > 0) {
        if (this.currentVideo) {
          this.updateSubtitle();
        }
      }
    }
  }

  loadBilingualSubtitles(englishSubtitles, chineseSubtitles) {
    this.englishSubtitles = englishSubtitles || [];
    this.chineseSubtitles = chineseSubtitles || [];

    // 加载字幕时，同步加载最新的设置
    this.loadSubtitleData();
    
    if (this.isEnabled && this.currentVideo) {
      this.updateSubtitle();
    }
  }

  loadNewSubtitle(subtitleData) {
    this.subtitleData = subtitleData;
  }

  clearSubtitleData() {
    this.subtitleData = [];
    this.englishSubtitles = [];
    this.chineseSubtitles = [];
    // 注意：不再清空设置和开关状态，保持用户的设置不变
    this.hideSubtitle();
  }

  // 强制重置 - 重置所有状态和设置
  forceReset() {
    // 清除所有字幕数据
    this.subtitleData = [];
    this.englishSubtitles = [];
    this.chineseSubtitles = [];
    this.currentVideo = null;
    this.autoLoadAttempted = false;
    
    // 重置为默认设置（从统一配置中心加载）
    this.englishSettings = getDefaultEnglishSettings();
    this.chineseSettings = getDefaultChineseSettings();
    
    // 重置自动加载设置
    this.autoLoadEnabled = false;
    this.serverUrl = 'http://127.0.0.1:8888';
    this.currentVideoId = null;
    
    // 重置显示状态
    this.isEnabled = false;
    this.hideSubtitle();
    
    // 重新应用默认样式
    this.applyStyles();
  }

  updateSettings(settings) {
    this.settings = { ...this.settings, ...settings };

    // 立即更新CSS变量，确保设置实时生效
    this.updateCSSVariables();
    this.applyStyles();
    
    if (this.isEnabled && this.overlayElement && this.overlayElement.style.display !== 'none') {
      this.repositionSubtitle();
    }
  }

  // 更新独立语言设置
  updateLanguageSettings(language, settings) {
    if (language === 'english') {
      this.englishSettings = { ...this.englishSettings, ...settings };
    } else if (language === 'chinese') {
      this.chineseSettings = { ...this.chineseSettings, ...settings };
    }

    // 重新应用对应语言的样式
    this.applyLanguageStyles(language);
  }

  async loadSubtitleData() {
    try {
      const currentVideoId = this.getVideoId();
      
      // 获取全局设置和当前视频的字幕数据
      const result = await chrome.storage.local.get([
        'subtitleEnabled', 
        'englishSettings',
        'chineseSettings',
        'autoLoadEnabled',
        `videoSubtitles_${currentVideoId}` // 基于videoId的字幕数据
      ]);
      
      // 清除之前的字幕数据
      this.subtitleData = [];
      this.englishSubtitles = [];
      this.chineseSubtitles = [];
      
      // 只有当前视频ID存在且有对应字幕数据时才加载
      if (currentVideoId && result[`videoSubtitles_${currentVideoId}`]) {
        const videoSubtitles = result[`videoSubtitles_${currentVideoId}`];
        
        // 加载当前视频的字幕数据
        if (videoSubtitles.englishSubtitles || videoSubtitles.chineseSubtitles) {
          this.englishSubtitles = videoSubtitles.englishSubtitles || [];
          this.chineseSubtitles = videoSubtitles.chineseSubtitles || [];
        } else if (videoSubtitles.subtitleData && videoSubtitles.subtitleData.length > 0) {
          this.subtitleData = videoSubtitles.subtitleData;
        }
      }
      
      if (result.subtitleEnabled !== undefined) {
        this.isEnabled = result.subtitleEnabled;
      }
      
      // 加载自动加载状态
      if (result.autoLoadEnabled !== undefined) {
        this.autoLoadEnabled = result.autoLoadEnabled;
      }

      // 加载独立语言设置
      if (result.englishSettings) {
        this.englishSettings = { ...this.englishSettings, ...result.englishSettings };
      }

      if (result.chineseSettings) {
        // 过滤掉空值，避免覆盖默认设置
        const filteredSettings = {};
        for (const [key, value] of Object.entries(result.chineseSettings)) {
          if (value !== '' && value !== null && value !== undefined) {
            filteredSettings[key] = value;
          }
        }

        this.chineseSettings = { ...this.chineseSettings, ...filteredSettings };
      }

      // 重新应用样式
      if (this.overlayElement) {
        this.applyLanguageStyles('english');
        this.applyLanguageStyles('chinese');
      }
    } catch (error) {
      console.error('加载字幕数据失败:', error);
    }
  }

  // 自动加载相关方法
  toggleAutoLoad(enabled) {
    this.autoLoadEnabled = enabled;

    if (enabled) {
      this.attemptAutoLoad();
    }
  }

  getVideoId() {
    const urlParams = new URLSearchParams(window.location.search);
    const videoId = urlParams.get('v');
    return videoId;
  }

  async attemptAutoLoad() {
    if (!this.autoLoadEnabled) {
      return;
    }

    const videoId = this.getVideoId();
    if (!videoId) {
      return;
    }

    // 检查是否为新的视频ID或页面刷新情况
    const isNewVideo = videoId !== this.currentVideoId;
    
    // 检查当前是否已有字幕数据
    const hasExistingSubtitles = this.englishSubtitles.length > 0 || 
                                this.chineseSubtitles.length > 0 || 
                                this.subtitleData.length > 0;
    
    // 触发自动加载的条件：
    // 1. 新视频 - 总是尝试加载
    // 2. 页面刷新且当前没有字幕数据 - 重新加载
    // 3. 页面刷新但从未尝试过加载 - 首次加载
    const shouldReload = isNewVideo || 
                        (!hasExistingSubtitles && !this.autoLoadAttempted) ||
                        (!hasExistingSubtitles);

    if (!shouldReload) {
      return;
    }

    this.currentVideoId = videoId;
    this.autoLoadAttempted = true;

    try {
      const response = await fetch(`${this.serverUrl}/subtitle/${videoId}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });

      if (!response.ok) {
        return;
      }

      const result = await response.json();
      if (result.success && result.content) {
        await this.processAutoLoadedSubtitle(result.content, result.info);

        // 通知popup更新状态，包含字幕数据
        chrome.runtime.sendMessage({
          action: 'autoLoadSuccess',
          videoId: videoId,
          filename: result.info.filename,
          englishSubtitles: this.englishSubtitles,
          chineseSubtitles: this.chineseSubtitles,
          subtitleData: this.subtitleData,
          englishFileName: this.englishFileName || result.info.filename + ' (英文)',
          chineseFileName: this.chineseFileName || result.info.filename + ' (中文)',
          fileName: this.currentFileName || result.info.filename
        });
      }

    } catch (error) {
      // 通知popup服务器连接失败
      chrome.runtime.sendMessage({
        action: 'autoLoadError',
        error: error.message
      });
    }
  }

  async processAutoLoadedSubtitle(content, info) {
    try {
      const format = info.format.toLowerCase();
      const currentVideoId = this.getVideoId();
      
      if (!currentVideoId) {
        console.error('❌ 无法获取视频ID，跳过字幕保存');
        return;
      }

      if (format === '.ass') {
        // 使用 SubtitleParser 统一解析 ASS 格式
        const assResult = SubtitleParser.parseASS(content);

        if (assResult.english.length > 0 || assResult.chinese.length > 0) {
          this.englishSubtitles = assResult.english;
          this.chineseSubtitles = assResult.chinese;
          
          // 基于videoId保存到本地存储
          await chrome.runtime.sendMessage({
            action: 'saveVideoSubtitles',
            videoId: currentVideoId,
            englishSubtitles: assResult.english,
            chineseSubtitles: assResult.chinese,
            englishFileName: info.filename + ' (英文)',
            chineseFileName: info.filename + ' (中文)'
          });
        }
      } else if (format === '.srt' || format === '.vtt') {
        // 处理SRT/VTT文件
        const subtitleData = format === '.srt' ? 
          SubtitleParser.parseSRT(content) : 
          SubtitleParser.parseVTT(content);
          
        if (subtitleData.length > 0) {
          this.subtitleData = subtitleData;
          
          // 基于videoId保存到本地存储
          await chrome.runtime.sendMessage({
            action: 'saveVideoSubtitles',
            videoId: currentVideoId,
            subtitleData: subtitleData,
            fileName: info.filename
          });
        }
      }

      // 自动启用字幕显示
      if (this.englishSubtitles.length > 0 || this.chineseSubtitles.length > 0 || this.subtitleData.length > 0) {
        this.isEnabled = true;
        
        // 通知background更新状态
        chrome.runtime.sendMessage({
          action: 'setSubtitleEnabled',
          enabled: true
        });
        
        if (this.currentVideo) {
          this.updateSubtitle();
        }
      }

    } catch (error) {
      console.error('处理自动加载的字幕失败:', error);
    }
  }
}

// 初始化
let subtitleOverlayInstance = null;

const initializeSubtitle = () => {
  if (!subtitleOverlayInstance) {
    subtitleOverlayInstance = new YouTubeSubtitleOverlay();
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeSubtitle);
} else {
  initializeSubtitle();
}

// 导出
window.SubtitleParser = SubtitleParser;
