/**
 * YouTube Subtitle Extension - Content Script
 * ============================================
 * 在 YouTube 页面上叠加字幕的核心逻辑
 */

import { getDefaultEnglishSettings, getDefaultChineseSettings, getDefaultConfig } from './config';
import { SubtitleParser } from './subtitle-parser';
import type { SimpleSubtitleEntry, SubtitleStyleSettings, VideoSubtitleData, ASSParseResult } from '../types';

// Chrome API 类型声明
declare const chrome: {
  runtime: {
    sendMessage: (message: unknown) => Promise<{ success: boolean; [key: string]: unknown }>;
    onMessage: {
      addListener: (
        callback: (
          request: ChromeMessage,
          sender: unknown,
          sendResponse: (response: unknown) => void
        ) => boolean | void
      ) => void;
    };
  };
  storage: {
    local: {
      get: (keys: string | string[]) => Promise<Record<string, unknown>>;
    };
  };
};

interface ChromeMessage {
  action: string;
  enabled?: boolean;
  subtitleData?: SimpleSubtitleEntry[];
  englishSubtitles?: SimpleSubtitleEntry[];
  chineseSubtitles?: SimpleSubtitleEntry[];
  language?: 'english' | 'chinese';
  settings?: Partial<SubtitleStyleSettings>;
  videoId?: string;
}

class YouTubeSubtitleOverlay {
  private subtitleData: SimpleSubtitleEntry[] = [];
  private englishSubtitles: SimpleSubtitleEntry[] = [];
  private chineseSubtitles: SimpleSubtitleEntry[] = [];
  private currentVideo: HTMLVideoElement | null = null;
  private overlayElement: HTMLElement | null = null;
  private isEnabled = false;

  private autoLoadEnabled = false;
  private currentVideoId: string | null = null;
  private autoLoadAttempted = false;

  private englishSettings: SubtitleStyleSettings;
  private chineseSettings: SubtitleStyleSettings;

  private enableDPRCompensation: boolean;
  private dprCompensationFactor: number;

  private onTimeUpdate: (() => void) | null = null;
  private onEnded: (() => void) | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private scrollListener: (() => void) | null = null;
  private fullscreenListener: (() => void) | null = null;
  private resizeWindowListener: (() => void) | null = null;
  private youtubeStateObserver: MutationObserver | null = null;

  constructor() {
    this.englishSettings = getDefaultEnglishSettings();
    this.chineseSettings = getDefaultChineseSettings();

    const config = getDefaultConfig();
    this.enableDPRCompensation = config.dpr.enabled;
    this.dprCompensationFactor = this.calculateDPRCompensation();

    this.init();
  }

  private calculateDPRCompensation(): number {
    const dpr = window.devicePixelRatio || 1;
    if (dpr <= 1) return 1.0;
    return 1 + (dpr - 1) * 0.4;
  }

  private init(): void {
    this.createOverlayElement();
    this.observeVideoChanges();
    this.loadSubtitleData();
    this.bindMessageListener();

    window.addEventListener('YTSP_TriggerAutoLoad', () => {
      this.autoLoadEnabled = true;
      this.attemptAutoLoad();
    });

    window.addEventListener('YTSP_StartTranslation', () => {
      this.startTranslationFromPage();
    });
  }

  private async startTranslationFromPage(): Promise<void> {
    try {
      // 检查扩展上下文是否有效
      if (!chrome.runtime?.id) {
        this.showErrorNotification('扩展已重新加载，请刷新页面后再试');
        return;
      }

      const subtitles = await this.fetchYouTubeOfficialSubtitles();
      if (!subtitles || subtitles.length === 0) {
        console.error('无法获取字幕');
        return;
      }

      const videoId = this.getVideoId();
      const result = await chrome.storage.local.get(['apiConfig']);
      const apiConfig = (result.apiConfig as Record<string, string>) || {};

      chrome.runtime.sendMessage({
        action: 'startTranslation',
        subtitles,
        targetLanguage: apiConfig.targetLanguage || 'zh',
        videoId,
        apiConfig,
      });
    } catch (error) {
      console.error('启动翻译失败:', error);
      // 检查是否是扩展上下文失效错误
      if (error instanceof Error && error.message.includes('Extension context invalidated')) {
        this.showErrorNotification('扩展已重新加载，请刷新页面后再试');
      }
    }
  }

  private bindMessageListener(): void {
    chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
      switch (request.action) {
        case 'toggleSubtitle':
          this.toggleSubtitle(request.enabled || false);
          break;
        case 'loadSubtitle':
          this.loadNewSubtitle(request.subtitleData || []);
          break;
        case 'loadBilingualSubtitles':
          this.loadBilingualSubtitles(
            request.englishSubtitles || [],
            request.chineseSubtitles || []
          );
          break;
        case 'clearData':
          this.clearSubtitleData();
          break;
        case 'forceReset':
          this.forceReset();
          break;
        case 'updateSettings':
          if (request.language && request.settings) {
            this.updateLanguageSettings(request.language, request.settings);
          }
          break;
        case 'toggleAutoLoad':
          this.toggleAutoLoad(request.enabled || false);
          break;
        case 'getVideoInfo': {
          const videoId = this.getVideoId();
          const subtitleLoaded =
            this.englishSubtitles.length > 0 || this.chineseSubtitles.length > 0;
          sendResponse({
            videoId,
            subtitleLoaded,
            autoLoadEnabled: this.autoLoadEnabled,
          });
          break;
        }
        case 'getSubtitleStatus': {
          const currentVideoId = this.getVideoId();
          const englishCount = this.englishSubtitles.length;
          const chineseCount = this.chineseSubtitles.length;
          const hasSubtitles = englishCount > 0 || chineseCount > 0;

          sendResponse({
            videoId: currentVideoId,
            hasSubtitles,
            englishCount,
            chineseCount,
            autoLoadEnabled: this.autoLoadEnabled,
            subtitleEnabled: this.isEnabled,
          });
          break;
        }
        case 'getYouTubeSubtitles':
          this.fetchYouTubeOfficialSubtitles()
            .then((subtitles) => {
              sendResponse({ success: true, subtitles });
            })
            .catch((error) => {
              sendResponse({ success: false, error: (error as Error).message });
            });
          return true;
      }
    });
  }

  private async fetchYouTubeOfficialSubtitles(): Promise<SimpleSubtitleEntry[]> {
    const videoId = this.getVideoId();
    if (!videoId) {
      throw new Error('无法获取视频ID');
    }

    const subtitles = await this.getSubtitlesFromTranscriptPanel();
    if (subtitles && subtitles.length > 0) {
      return subtitles;
    }

    throw new Error('无法获取YouTube字幕，请确保视频有可用的字幕并打开文字记录面板');
  }

  private async getSubtitlesFromTranscriptPanel(): Promise<SimpleSubtitleEntry[]> {
    let transcriptSegments = document.querySelectorAll('ytd-transcript-segment-renderer');

    if (!transcriptSegments || transcriptSegments.length === 0) {
      const moreButton = document.querySelector('#expand') as HTMLElement;
      if (moreButton) {
        moreButton.click();
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      const transcriptButtonSelectors = [
        'button[aria-label*="transcript" i]',
        'button[aria-label*="Show transcript" i]',
        'ytd-button-renderer[button-renderer*="transcript" i]',
        '#primary-button:has(yt-formatted-string)',
        '.ytd-video-description-transcript-section-renderer button',
      ];

      for (const selector of transcriptButtonSelectors) {
        const btn = document.querySelector(selector) as HTMLElement;
        if (btn) {
          btn.click();
          await new Promise((resolve) => setTimeout(resolve, 1000));
          break;
        }
      }

      transcriptSegments = document.querySelectorAll('ytd-transcript-segment-renderer');
    }

    if (!transcriptSegments || transcriptSegments.length === 0) {
      throw new Error('Transcript面板未找到字幕');
    }

    const subtitles: SimpleSubtitleEntry[] = [];
    transcriptSegments.forEach((segment, index) => {
      let timestampElement =
        segment.querySelector('.segment-timestamp') ||
        segment.querySelector('[class*="timestamp"]') ||
        segment.querySelector('div[class*="time"]');
      let textElement =
        segment.querySelector('.segment-text') ||
        segment.querySelector('[class*="text"]') ||
        segment.querySelector('yt-formatted-string');

      if (!timestampElement || !textElement) {
        const divs = segment.querySelectorAll('div');
        if (divs.length >= 2) {
          timestampElement = timestampElement || divs[0];
          textElement = textElement || divs[1];
        }
      }

      if (timestampElement && textElement) {
        const timestamp = this.parseTimestamp(timestampElement.textContent || '');
        const text = textElement.textContent?.trim() || '';

        if (text) {
          const nextSegment = transcriptSegments[index + 1];
          let endTime = timestamp + 5;
          if (nextSegment) {
            const nextTimestamp =
              nextSegment.querySelector('.segment-timestamp') ||
              nextSegment.querySelector('[class*="timestamp"]') ||
              nextSegment.querySelector('div');
            if (nextTimestamp) {
              endTime = this.parseTimestamp(nextTimestamp.textContent || '');
            }
          }

          subtitles.push({ startTime: timestamp, endTime, text });
        }
      }
    });

    return subtitles;
  }

  private parseTimestamp(timestampStr: string): number {
    if (!timestampStr) return 0;
    const parts = timestampStr.trim().split(':').map(Number);
    let seconds = 0;

    if (parts.length === 2) {
      seconds = parts[0] * 60 + parts[1];
    } else if (parts.length === 3) {
      seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
    }

    return seconds;
  }

  private createOverlayElement(): void {
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

  private applyStyles(): void {
    if (!this.overlayElement) return;

    const mainStyles: Partial<CSSStyleDeclaration> = {
      position: 'absolute',
      zIndex: '40',
      display: 'none',
      left: '50%',
      transform: 'translateX(-50%)',
      bottom: '60px',
      pointerEvents: 'none',
      userSelect: 'none',
    };
    Object.assign(this.overlayElement.style, mainStyles);

    const container = this.overlayElement.querySelector('.subtitle-container') as HTMLElement;
    this.applyContainerFlexStyles(container);

    const englishWrapper = this.overlayElement.querySelector('.english-wrapper') as HTMLElement;
    const chineseWrapper = this.overlayElement.querySelector('.chinese-wrapper') as HTMLElement;

    this.applyWrapperStyles(englishWrapper);
    this.applyWrapperStyles(chineseWrapper);

    this.applyLanguageStyles('english');
    this.applyLanguageStyles('chinese');
  }

  private applyContainerFlexStyles(container: HTMLElement | null): void {
    if (container) {
      Object.assign(container.style, {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0px',
        width: '100%',
      });
    }
  }

  private applyWrapperStyles(wrapper: HTMLElement | null): void {
    if (wrapper) {
      Object.assign(wrapper.style, {
        display: 'block',
        textAlign: 'center',
        width: '100%',
        textWrap: 'balance',
      });
    }
  }

  private applyLanguageStyles(language: 'english' | 'chinese'): void {
    if (!this.overlayElement) return;

    const settings = language === 'english' ? this.englishSettings : this.chineseSettings;
    const elementId = language === 'english' ? '#englishSubtitle' : '#chineseSubtitle';
    const element = this.overlayElement.querySelector(elementId) as HTMLElement;

    if (element && settings) {
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
        lineHeight: String(settings.lineHeight),
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
        margin: '0',
      });
    }
  }

  private observeVideoChanges(): void {
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

  private onVideoChange(): void {
    const video = document.querySelector('video') as HTMLVideoElement;
    const videoElementChanged = video && video !== this.currentVideo;
    const newVideoId = this.getVideoId();
    const videoIdChanged = newVideoId && newVideoId !== this.currentVideoId;

    if (videoElementChanged) {
      this.currentVideo = video;
      this.setupVideoListeners();
      this.insertOverlayToPage();
      this.setupResizeListener();
    }

    if (videoIdChanged || videoElementChanged) {
      this.hideSubtitle();
      this.subtitleData = [];
      this.englishSubtitles = [];
      this.chineseSubtitles = [];
      this.loadSubtitleData();
      this.autoLoadAttempted = false;

      setTimeout(() => {
        this.attemptAutoLoad();
      }, 500);
    }
  }

  private setupVideoListeners(): void {
    if (!this.currentVideo) return;

    if (this.onTimeUpdate) {
      this.currentVideo.removeEventListener('timeupdate', this.onTimeUpdate);
    }
    if (this.onEnded) {
      this.currentVideo.removeEventListener('ended', this.onEnded);
    }

    this.onTimeUpdate = () => {
      if (this.isEnabled) {
        if (
          this.englishSubtitles.length > 0 ||
          this.chineseSubtitles.length > 0 ||
          this.subtitleData.length > 0
        ) {
          this.updateSubtitle();
        }
      }
    };

    this.onEnded = () => {
      this.hideSubtitle();
    };

    this.currentVideo.addEventListener('timeupdate', this.onTimeUpdate);
    this.currentVideo.addEventListener('ended', this.onEnded);

    if (this.isEnabled) {
      this.updateSubtitle();
    }
  }

  private setupResizeListener(): void {
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

    const throttleReposition = this.throttle(() => {
      if (this.overlayElement && this.isEnabled) {
        this.repositionSubtitle();
      }
    }, 100);

    this.resizeObserver = new ResizeObserver(() => {
      throttleReposition();
    });

    this.scrollListener = () => {
      throttleReposition();
    };

    this.fullscreenListener = () => {
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

  private throttle<T extends (...args: unknown[]) => void>(func: T, wait: number): T {
    let timeout: ReturnType<typeof setTimeout> | null = null;
    return ((...args: unknown[]) => {
      const later = () => {
        timeout = null;
        func(...args);
      };
      if (timeout) clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    }) as T;
  }

  private setupYouTubeStateListener(): void {
    if (this.youtubeStateObserver) {
      this.youtubeStateObserver.disconnect();
    }

    this.youtubeStateObserver = new MutationObserver((mutations) => {
      let needsReposition = false;

      mutations.forEach((mutation) => {
        if (
          mutation.type === 'attributes' &&
          (mutation.attributeName === 'class' || mutation.attributeName === 'theater')
        ) {
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
      document.body,
    ].filter((el): el is Element => el !== null);

    targets.forEach((target) => {
      this.youtubeStateObserver!.observe(target, {
        attributes: true,
        attributeFilter: ['class', 'theater', 'fullscreen'],
        childList: true,
        subtree: false,
      });
    });
  }

  private repositionSubtitle(): void {
    if (!this.overlayElement || !this.currentVideo) return;

    const container = this.overlayElement.querySelector('.subtitle-container') as HTMLElement;
    this.applyContainerFlexStyles(container);

    const englishWrapper = this.overlayElement.querySelector('.english-wrapper') as HTMLElement;
    const chineseWrapper = this.overlayElement.querySelector('.chinese-wrapper') as HTMLElement;

    this.applyWrapperStyles(englishWrapper);
    this.applyWrapperStyles(chineseWrapper);

    this.applyLanguageStyles('english');
    this.applyLanguageStyles('chinese');

    const englishEl = this.overlayElement.querySelector('#englishSubtitle') as HTMLElement;
    const chineseEl = this.overlayElement.querySelector('#chineseSubtitle') as HTMLElement;
    const hasText =
      (englishEl && englishEl.textContent?.trim()) ||
      (chineseEl && chineseEl.textContent?.trim());
    if (!hasText) {
      this.overlayElement.style.display = 'none';
    }

    const isFullscreen = document.fullscreenElement !== null;
    const isTheaterMode = document.querySelector('.ytp-size-large') !== null;
    const isMiniPlayer = document.querySelector('.ytp-miniplayer-active') !== null;
    const videoRect = this.currentVideo.getBoundingClientRect();

    const playerContainer = document.querySelector('#movie_player') as HTMLElement;

    if (isFullscreen) {
      this.overlayElement.style.position = 'fixed';
      this.overlayElement.style.left = '50%';
      this.overlayElement.style.transform = 'translateX(-50%)';
      this.overlayElement.style.bottom = '80px';
      this.overlayElement.style.width = '80%';
      this.overlayElement.style.maxWidth = 'none';
      this.overlayElement.style.zIndex = '9999';
    } else if (isMiniPlayer) {
      this.overlayElement.style.display = 'none';
      return;
    } else {
      this.overlayElement.style.display = 'block';
      this.overlayElement.style.position = 'absolute';
      this.overlayElement.style.zIndex = '40';

      if (playerContainer) {
        if (playerContainer.style.position !== 'relative') {
          playerContainer.style.position = 'relative';
        }

        this.overlayElement.style.left = '50%';
        this.overlayElement.style.transform = 'translateX(-50%)';
        this.overlayElement.style.bottom = isTheaterMode ? '70px' : '60px';
        this.overlayElement.style.width = '80%';
        this.overlayElement.style.maxWidth = 'none';

        if (!playerContainer.contains(this.overlayElement)) {
          playerContainer.appendChild(this.overlayElement);
        }
      } else {
        this.overlayElement.style.position = 'fixed';
        this.overlayElement.style.left = '50%';
        this.overlayElement.style.transform = 'translateX(-50%)';
        this.overlayElement.style.bottom = window.innerHeight - videoRect.bottom + 60 + 'px';
        this.overlayElement.style.width = Math.min(videoRect.width * 0.8, 800) + 'px';
        this.overlayElement.style.maxWidth = 'none';
      }
    }
  }

  private insertOverlayToPage(): void {
    const existingOverlay = document.getElementById('youtube-local-subtitle-overlay');
    if (existingOverlay) {
      existingOverlay.remove();
    }

    const moviePlayer = document.querySelector('#movie_player') as HTMLElement;

    if (moviePlayer) {
      if (moviePlayer.style.position !== 'relative') {
        moviePlayer.style.position = 'relative';
      }

      if (this.overlayElement) {
        moviePlayer.appendChild(this.overlayElement);
      }
    } else {
      if (this.overlayElement) {
        document.body.appendChild(this.overlayElement);
      }
    }

    this.repositionSubtitle();
  }

  private updateSubtitle(): void {
    if (!this.currentVideo || !this.isEnabled || !this.overlayElement) {
      return;
    }

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

    if (
      this.englishSubtitles.length === 0 &&
      this.chineseSubtitles.length === 0 &&
      this.subtitleData.length > 0
    ) {
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

  private showBilingualSubtitle(englishText: string, chineseText: string): void {
    if (!this.overlayElement) return;

    const englishSubtitle = this.overlayElement.querySelector('#englishSubtitle') as HTMLElement;
    const chineseSubtitle = this.overlayElement.querySelector('#chineseSubtitle') as HTMLElement;

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
    this.overlayElement.style.zIndex = '40';
    this.overlayElement.style.visibility = 'visible';
    this.overlayElement.style.opacity = '1';

    this.repositionSubtitle();
  }

  private hideSubtitle(): void {
    if (!this.overlayElement) return;
    this.overlayElement.style.display = 'none';
    this.overlayElement.style.visibility = 'hidden';
    this.overlayElement.style.opacity = '0';

    const englishSubtitle = this.overlayElement.querySelector('#englishSubtitle') as HTMLElement;
    const chineseSubtitle = this.overlayElement.querySelector('#chineseSubtitle') as HTMLElement;
    if (englishSubtitle) {
      englishSubtitle.textContent = '';
      englishSubtitle.style.display = 'none';
    }
    if (chineseSubtitle) {
      chineseSubtitle.textContent = '';
      chineseSubtitle.style.display = 'none';
    }
  }

  private showErrorNotification(message: string): void {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.5);
      z-index: 10000;
      display: flex;
      justify-content: center;
      align-items: center;
    `;

    const modal = document.createElement('div');
    modal.style.cssText = `
      background: #f44336;
      color: white;
      padding: 20px 30px;
      border-radius: 8px;
      font-size: 16px;
      font-weight: bold;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;
    modal.textContent = message;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    setTimeout(() => overlay.remove(), 3000);
  }

  private findCurrentSubtitle(
    currentTime: number,
    subtitles: SimpleSubtitleEntry[]
  ): SimpleSubtitleEntry | undefined {
    return subtitles.find(
      (subtitle) => currentTime >= subtitle.startTime && currentTime <= subtitle.endTime
    );
  }

  private toggleSubtitle(enabled: boolean): void {
    this.isEnabled = enabled;

    if (!enabled) {
      this.hideSubtitle();
    } else {
      if (
        this.englishSubtitles.length > 0 ||
        this.chineseSubtitles.length > 0 ||
        this.subtitleData.length > 0
      ) {
        if (this.currentVideo) {
          this.updateSubtitle();
        }
      }
    }
  }

  private loadBilingualSubtitles(
    englishSubtitles: SimpleSubtitleEntry[],
    chineseSubtitles: SimpleSubtitleEntry[]
  ): void {
    this.englishSubtitles = englishSubtitles || [];
    this.chineseSubtitles = chineseSubtitles || [];

    if (this.englishSubtitles.length > 0 || this.chineseSubtitles.length > 0) {
      this.isEnabled = true;

      if (!this.currentVideo) {
        this.currentVideo = document.querySelector('video');
      }

      if (!this.overlayElement) {
        this.createOverlayElement();
        this.insertOverlayToPage();
      }

      chrome.runtime.sendMessage({
        action: 'setSubtitleEnabled',
        enabled: true,
      });

      if (this.currentVideo && this.overlayElement) {
        this.setupVideoListeners();
        this.updateSubtitle();
      }
    }
  }

  private loadNewSubtitle(subtitleData: SimpleSubtitleEntry[]): void {
    this.subtitleData = subtitleData;
  }

  private clearSubtitleData(): void {
    this.subtitleData = [];
    this.englishSubtitles = [];
    this.chineseSubtitles = [];
    this.hideSubtitle();
  }

  private forceReset(): void {
    this.subtitleData = [];
    this.englishSubtitles = [];
    this.chineseSubtitles = [];
    this.currentVideo = null;
    this.autoLoadAttempted = false;

    this.englishSettings = getDefaultEnglishSettings();
    this.chineseSettings = getDefaultChineseSettings();

    this.autoLoadEnabled = false;
    this.currentVideoId = null;

    this.isEnabled = false;
    this.hideSubtitle();

    this.applyStyles();
  }

  private updateLanguageSettings(
    language: 'english' | 'chinese',
    settings: Partial<SubtitleStyleSettings>
  ): void {
    if (language === 'english') {
      this.englishSettings = { ...this.englishSettings, ...settings };
    } else if (language === 'chinese') {
      this.chineseSettings = { ...this.chineseSettings, ...settings };
    }

    this.applyLanguageStyles(language);
  }

  private async loadSubtitleData(): Promise<void> {
    try {
      const currentVideoId = this.getVideoId();

      const result = await chrome.storage.local.get([
        'subtitleEnabled',
        'englishSettings',
        'chineseSettings',
        'autoLoadEnabled',
        `videoSubtitles_${currentVideoId}`,
      ]);

      this.subtitleData = [];
      this.englishSubtitles = [];
      this.chineseSubtitles = [];

      if (currentVideoId && result[`videoSubtitles_${currentVideoId}`]) {
        const videoSubtitles = result[`videoSubtitles_${currentVideoId}`] as VideoSubtitleData;

        if (videoSubtitles.englishSubtitles || videoSubtitles.chineseSubtitles) {
          this.englishSubtitles = videoSubtitles.englishSubtitles || [];
          this.chineseSubtitles = videoSubtitles.chineseSubtitles || [];
        } else if (videoSubtitles.subtitleData && videoSubtitles.subtitleData.length > 0) {
          this.subtitleData = videoSubtitles.subtitleData;
        }
      }

      if (result.subtitleEnabled !== undefined) {
        this.isEnabled = result.subtitleEnabled as boolean;
      }

      if (result.autoLoadEnabled !== undefined) {
        this.autoLoadEnabled = result.autoLoadEnabled as boolean;
      }

      if (result.englishSettings) {
        this.englishSettings = {
          ...this.englishSettings,
          ...(result.englishSettings as SubtitleStyleSettings),
        };
      }

      if (result.chineseSettings) {
        const filteredSettings: Partial<SubtitleStyleSettings> = {};
        for (const [key, value] of Object.entries(
          result.chineseSettings as SubtitleStyleSettings
        )) {
          if (value !== '' && value !== null && value !== undefined) {
            (filteredSettings as Record<string, unknown>)[key] = value;
          }
        }

        this.chineseSettings = { ...this.chineseSettings, ...filteredSettings };
      }

      if (this.overlayElement) {
        this.applyLanguageStyles('english');
        this.applyLanguageStyles('chinese');
      }

      const hasSubtitles =
        this.englishSubtitles.length > 0 ||
        this.chineseSubtitles.length > 0 ||
        this.subtitleData.length > 0;

      if (hasSubtitles && this.isEnabled) {
        if (!this.currentVideo) {
          this.currentVideo = document.querySelector('video');
        }
        if (this.currentVideo && this.overlayElement) {
          this.setupVideoListeners();
          this.updateSubtitle();
        }
      }
    } catch (error) {
      console.error('加载字幕数据失败:', error);
    }
  }

  private toggleAutoLoad(enabled: boolean): void {
    this.autoLoadEnabled = enabled;

    if (enabled) {
      this.attemptAutoLoad();
    }
  }

  private getVideoId(): string | null {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('v');
  }

  private async attemptAutoLoad(): Promise<void> {
    if (!this.autoLoadEnabled) return;

    const videoId = this.getVideoId();
    if (!videoId) return;

    const isNewVideo = videoId !== this.currentVideoId;
    const hasExistingSubtitles =
      this.englishSubtitles.length > 0 ||
      this.chineseSubtitles.length > 0 ||
      this.subtitleData.length > 0;

    if (!isNewVideo && hasExistingSubtitles) return;

    this.currentVideoId = videoId;
    this.autoLoadAttempted = true;

    await this.loadSubtitleData();
  }

  async processAutoLoadedSubtitle(
    content: string,
    info: { format: string; filename: string }
  ): Promise<void> {
    try {
      const format = info.format.toLowerCase();
      const currentVideoId = this.getVideoId();

      if (!currentVideoId) {
        console.error('❌ 无法获取视频ID，跳过字幕保存');
        return;
      }

      if (format === '.ass') {
        const assResult: ASSParseResult = SubtitleParser.parseASS(content);

        if (assResult.english.length > 0 || assResult.chinese.length > 0) {
          this.englishSubtitles = assResult.english;
          this.chineseSubtitles = assResult.chinese;

          await chrome.runtime.sendMessage({
            action: 'saveVideoSubtitles',
            videoId: currentVideoId,
            englishSubtitles: assResult.english,
            chineseSubtitles: assResult.chinese,
            englishFileName: info.filename + ' (英文)',
            chineseFileName: info.filename + ' (中文)',
          });
        }
      } else if (format === '.srt' || format === '.vtt') {
        const subtitleData =
          format === '.srt'
            ? SubtitleParser.parseSRT(content)
            : SubtitleParser.parseVTT(content);

        if (subtitleData.length > 0) {
          this.subtitleData = subtitleData;

          await chrome.runtime.sendMessage({
            action: 'saveVideoSubtitles',
            videoId: currentVideoId,
            subtitleData,
            fileName: info.filename,
          });
        }
      }

      if (
        this.englishSubtitles.length > 0 ||
        this.chineseSubtitles.length > 0 ||
        this.subtitleData.length > 0
      ) {
        this.isEnabled = true;

        chrome.runtime.sendMessage({
          action: 'setSubtitleEnabled',
          enabled: true,
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
let subtitleOverlayInstance: YouTubeSubtitleOverlay | null = null;

const initializeSubtitle = (): void => {
  if (!subtitleOverlayInstance) {
    subtitleOverlayInstance = new YouTubeSubtitleOverlay();
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeSubtitle);
} else {
  initializeSubtitle();
}
