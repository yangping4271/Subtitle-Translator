/**
 * YouTube Subtitle Extension - Background Service Worker
 * ============================================
 * 处理扩展的后台任务：消息通信、存储管理、翻译任务
 */

import { getDefaultEnglishSettings, getDefaultChineseSettings } from './config';
import { translatorService } from './translator';
import type { SimpleSubtitleEntry, SubtitleStyleSettings, VideoSubtitleData, TranslationProgress } from '../types';

// Chrome API 类型声明
declare const chrome: {
  runtime: {
    onInstalled: {
      addListener: (callback: (details: { reason: string }) => void) => void;
    };
    onMessage: {
      addListener: (
        callback: (
          request: ChromeMessage,
          sender: chrome.runtime.MessageSender,
          sendResponse: (response: unknown) => void
        ) => boolean | void
      ) => void;
    };
  };
  tabs: {
    query: (query: { active?: boolean; currentWindow?: boolean }) => Promise<chrome.tabs.Tab[]>;
    sendMessage: (tabId: number, message: unknown) => Promise<void>;
  };
  storage: {
    local: {
      get: (keys: string | string[] | null) => Promise<Record<string, unknown>>;
      set: (items: Record<string, unknown>) => Promise<void>;
      remove: (keys: string | string[]) => Promise<void>;
      clear: () => Promise<void>;
    };
  };
};

declare namespace chrome.runtime {
  interface MessageSender {
    tab?: chrome.tabs.Tab;
  }
}

declare namespace chrome.tabs {
  interface Tab {
    id?: number;
    url?: string;
  }
}

interface ChromeMessage {
  action: string;
  data?: unknown;
  enabled?: boolean;
  settings?: {
    language: 'english' | 'chinese';
    data: Partial<SubtitleStyleSettings>;
  };
  subtitles?: SimpleSubtitleEntry[];
  targetLanguage?: string;
  videoId?: string;
  apiConfig?: {
    openaiBaseUrl?: string;
    openaiApiKey?: string;
    llmModel?: string;
    targetLanguage?: string;
  };
  englishSubtitles?: SimpleSubtitleEntry[];
  chineseSubtitles?: SimpleSubtitleEntry[];
  englishFileName?: string;
  chineseFileName?: string;
  subtitleData?: SimpleSubtitleEntry[];
  fileName?: string;
}

class SubtitleExtensionBackground {
  constructor() {
    this.init();
  }

  init(): void {
    chrome.runtime.onInstalled.addListener((details) => {
      if (details.reason === 'install') {
        this.onInstall();
      } else if (details.reason === 'update') {
        this.onUpdate();
      }
    });

    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      this.handleMessage(request, sender, sendResponse);
      return true;
    });
  }

  onInstall(): void {
    chrome.storage.local.clear().then(() => {
      chrome.storage.local.set({
        subtitleEnabled: false,
        subtitleData: [],
        englishSubtitles: [],
        chineseSubtitles: [],
        englishFileName: '',
        chineseFileName: '',
        englishSettings: getDefaultEnglishSettings(),
        chineseSettings: getDefaultChineseSettings(),
        autoLoadEnabled: false,
      });
    });
  }

  onUpdate(): void {
    chrome.storage.local.get(['englishSettings']).then((res) => {
      const english = (res.englishSettings as SubtitleStyleSettings) || {};
      const needsFix = !english.fontFamily || english.fontFamily === 'inherit';
      if (needsFix) {
        const fixed: SubtitleStyleSettings = {
          ...getDefaultEnglishSettings(),
          ...english,
          fontFamily: '"Noto Serif", Georgia, serif',
        };
        chrome.storage.local.set({ englishSettings: fixed }).then(() => {
          try {
            this.notifyContentScript('updateSettings', { language: 'english', settings: fixed });
          } catch {
            // 忽略通知错误
          }
        });
      }
    });
  }

  async handleMessage(
    request: ChromeMessage,
    _sender: chrome.runtime.MessageSender,
    sendResponse: (response: unknown) => void
  ): Promise<void> {
    try {
      switch (request.action) {
        case 'getSubtitleData': {
          const data = await this.getSubtitleData();
          sendResponse({ success: true, data });
          break;
        }

        case 'getBilingualSubtitleData': {
          const bilingualData = await this.getBilingualSubtitleData();
          sendResponse({ success: true, data: bilingualData });
          break;
        }

        case 'saveSubtitleData':
          await this.saveSubtitleData(request.data as SimpleSubtitleEntry[]);
          await this.notifyContentScript('loadSubtitle', { subtitleData: request.data });
          sendResponse({ success: true });
          break;

        case 'saveBilingualSubtitles':
          await this.saveBilingualSubtitles(
            request.englishSubtitles || [],
            request.chineseSubtitles || [],
            request.englishFileName || '',
            request.chineseFileName || ''
          );
          await this.notifyContentScript('loadBilingualSubtitles', {
            englishSubtitles: request.englishSubtitles,
            chineseSubtitles: request.chineseSubtitles,
          });
          sendResponse({ success: true });
          break;

        case 'saveVideoSubtitles':
          await this.saveVideoSubtitles(
            request.videoId || '',
            request.englishSubtitles,
            request.chineseSubtitles,
            request.subtitleData,
            request.englishFileName,
            request.chineseFileName,
            request.fileName
          );
          sendResponse({ success: true });
          break;

        case 'toggleSubtitle':
          await this.toggleSubtitle(request.enabled || false);
          sendResponse({ success: true });
          break;

        case 'updateSettings':
          if (request.settings) {
            await this.updateSettings(request.settings);
          }
          sendResponse({ success: true });
          break;

        case 'clearSubtitleData':
          await this.clearSubtitleData();
          sendResponse({ success: true });
          break;

        case 'forceReset':
          await this.forceReset();
          sendResponse({ success: true });
          break;

        case 'setSubtitleEnabled':
          await this.setSubtitleEnabled(request.enabled || false);
          sendResponse({ success: true });
          break;

        case 'autoLoadSuccess':
        case 'autoLoadError':
          break;

        case 'startTranslation':
          this.startBackgroundTranslation(request, sendResponse);
          break;

        case 'cancelTranslation':
          this.cancelBackgroundTranslation(sendResponse);
          break;

        case 'getTranslationStatus': {
          const status = await chrome.storage.local.get(['translationProgress']);
          sendResponse({
            success: true,
            progress: (status.translationProgress as TranslationProgress) || null,
          });
          break;
        }

        default:
          sendResponse({ success: false, error: '未知操作' });
      }
    } catch (error) {
      console.error('处理消息时出错:', error);
      sendResponse({ success: false, error: (error as Error).message });
    }
  }

  async getBilingualSubtitleData(): Promise<{
    subtitleData: SimpleSubtitleEntry[];
    englishSubtitles: SimpleSubtitleEntry[];
    chineseSubtitles: SimpleSubtitleEntry[];
    subtitleEnabled: boolean;
    englishSettings: SubtitleStyleSettings;
    chineseSettings: SubtitleStyleSettings;
    englishFileName: string;
    chineseFileName: string;
  }> {
    const result = await chrome.storage.local.get([
      'subtitleData',
      'englishSubtitles',
      'chineseSubtitles',
      'subtitleEnabled',
      'englishSettings',
      'chineseSettings',
      'englishFileName',
      'chineseFileName',
    ]);
    return {
      subtitleData: (result.subtitleData as SimpleSubtitleEntry[]) || [],
      englishSubtitles: (result.englishSubtitles as SimpleSubtitleEntry[]) || [],
      chineseSubtitles: (result.chineseSubtitles as SimpleSubtitleEntry[]) || [],
      subtitleEnabled: (result.subtitleEnabled as boolean) || false,
      englishSettings: (result.englishSettings as SubtitleStyleSettings) || {},
      chineseSettings: (result.chineseSettings as SubtitleStyleSettings) || {},
      englishFileName: (result.englishFileName as string) || '',
      chineseFileName: (result.chineseFileName as string) || '',
    };
  }

  async getSubtitleData(): Promise<{
    subtitleData: SimpleSubtitleEntry[];
    subtitleEnabled: boolean;
    subtitleSettings: Record<string, unknown>;
  }> {
    const result = await chrome.storage.local.get([
      'subtitleData',
      'subtitleEnabled',
      'subtitleSettings',
    ]);
    return {
      subtitleData: (result.subtitleData as SimpleSubtitleEntry[]) || [],
      subtitleEnabled: (result.subtitleEnabled as boolean) || false,
      subtitleSettings: (result.subtitleSettings as Record<string, unknown>) || {},
    };
  }

  async saveVideoSubtitles(
    videoId: string,
    englishSubtitles?: SimpleSubtitleEntry[],
    chineseSubtitles?: SimpleSubtitleEntry[],
    subtitleData?: SimpleSubtitleEntry[],
    englishFileName?: string,
    chineseFileName?: string,
    fileName?: string
  ): Promise<void> {
    if (!videoId) {
      console.error('❌ 保存字幕失败: 缺少视频ID');
      return;
    }

    const subtitleKey = `videoSubtitles_${videoId}`;
    const videoSubtitleData: VideoSubtitleData = {
      videoId,
      timestamp: new Date().toISOString(),
    };

    if (englishSubtitles || chineseSubtitles) {
      videoSubtitleData.englishSubtitles = englishSubtitles || [];
      videoSubtitleData.chineseSubtitles = chineseSubtitles || [];
      videoSubtitleData.englishFileName = englishFileName || '';
      videoSubtitleData.chineseFileName = chineseFileName || '';
    }

    if (subtitleData) {
      videoSubtitleData.subtitleData = subtitleData;
      videoSubtitleData.fileName = fileName || '';
    }

    await chrome.storage.local.set({ [subtitleKey]: videoSubtitleData });

    if (englishSubtitles || chineseSubtitles) {
      await this.notifyContentScript('loadBilingualSubtitles', {
        englishSubtitles: englishSubtitles || [],
        chineseSubtitles: chineseSubtitles || [],
      });
    }
  }

  async saveBilingualSubtitles(
    englishSubtitles: SimpleSubtitleEntry[],
    chineseSubtitles: SimpleSubtitleEntry[],
    englishFileName: string,
    chineseFileName: string
  ): Promise<void> {
    await chrome.storage.local.set({
      englishSubtitles: englishSubtitles || [],
      chineseSubtitles: chineseSubtitles || [],
      englishFileName: englishFileName || '',
      chineseFileName: chineseFileName || '',
    });
  }

  async saveSubtitleData(data: SimpleSubtitleEntry[]): Promise<void> {
    await chrome.storage.local.set({ subtitleData: data });
  }

  async toggleSubtitle(enabled: boolean): Promise<void> {
    await chrome.storage.local.set({ subtitleEnabled: enabled });
    await this.notifyContentScript('toggleSubtitle', { enabled });
  }

  async setSubtitleEnabled(enabled: boolean): Promise<void> {
    await chrome.storage.local.set({ subtitleEnabled: enabled });
  }

  async updateSettings(settings: {
    language: 'english' | 'chinese';
    data: Partial<SubtitleStyleSettings>;
  }): Promise<void> {
    if (settings.language === 'english') {
      const currentSettings = await chrome.storage.local.get(['englishSettings']);
      const newSettings = {
        ...((currentSettings.englishSettings as SubtitleStyleSettings) || {}),
        ...settings.data,
      };

      await chrome.storage.local.set({ englishSettings: newSettings });
      await this.notifyContentScript('updateSettings', {
        language: 'english',
        settings: newSettings,
      });
    } else if (settings.language === 'chinese') {
      const currentSettings = await chrome.storage.local.get(['chineseSettings']);
      const newSettings = {
        ...((currentSettings.chineseSettings as SubtitleStyleSettings) || {}),
        ...settings.data,
      };

      await chrome.storage.local.set({ chineseSettings: newSettings });
      await this.notifyContentScript('updateSettings', {
        language: 'chinese',
        settings: newSettings,
      });
    }
  }

  async clearSubtitleData(): Promise<void> {
    const allData = await chrome.storage.local.get(null);
    const videoSubtitleKeys = Object.keys(allData).filter((key) =>
      key.startsWith('videoSubtitles_')
    );

    await chrome.storage.local.set({
      subtitleData: [],
      englishSubtitles: [],
      chineseSubtitles: [],
      englishFileName: '',
      chineseFileName: '',
    });

    if (videoSubtitleKeys.length > 0) {
      await chrome.storage.local.remove(videoSubtitleKeys);
    }

    await this.notifyContentScript('clearData', {});
  }

  async forceReset(): Promise<void> {
    await chrome.storage.local.clear();

    await chrome.storage.local.set({
      subtitleEnabled: false,
      subtitleData: [],
      englishSubtitles: [],
      chineseSubtitles: [],
      englishFileName: '',
      chineseFileName: '',
      englishSettings: getDefaultEnglishSettings(),
      chineseSettings: getDefaultChineseSettings(),
      autoLoadEnabled: false,
    });

    await this.notifyContentScript('forceReset', {});
  }

  async notifyContentScript(action: string, data: Record<string, unknown> = {}): Promise<void> {
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      const currentTab = tabs[0];

      if (currentTab && currentTab.id && this.isYouTubePage(currentTab.url)) {
        await chrome.tabs.sendMessage(currentTab.id, {
          action,
          ...data,
        });
      }
    } catch (error) {
      console.error('向content script发送消息失败:', error);
    }
  }

  isYouTubePage(url?: string): boolean {
    return !!url && (url.includes('youtube.com/watch') || url.includes('youtu.be/'));
  }

  async startBackgroundTranslation(
    request: ChromeMessage,
    sendResponse: (response: unknown) => void
  ): Promise<void> {
    const { subtitles, targetLanguage, videoId, apiConfig } = request;

    if (!subtitles || subtitles.length === 0) {
      sendResponse({ success: false, error: '没有字幕数据' });
      return;
    }

    if (apiConfig) {
      await chrome.storage.local.set({ apiConfig });
    }

    sendResponse({ success: true, message: '翻译已在后台启动' });

    try {
      const result = await translatorService.translateFull(
        subtitles,
        targetLanguage || 'zh',
        null
      );

      if (videoId) {
        await this.saveVideoSubtitles(
          videoId,
          result.english,
          result.chinese,
          undefined,
          'YouTube字幕 (英文)',
          'AI翻译 (中文)',
          undefined
        );
      }
    } catch (error) {
      console.error('❌ 后台翻译失败:', error);
      await chrome.storage.local.set({
        translationProgress: {
          isTranslating: false,
          error: (error as Error).message,
          timestamp: Date.now(),
        } as TranslationProgress,
      });
    }
  }

  async cancelBackgroundTranslation(sendResponse: (response: unknown) => void): Promise<void> {
    translatorService.cancelTranslation();
    await chrome.storage.local.remove('translationProgress');
    sendResponse({ success: true });
  }
}

// 初始化 background 服务
new SubtitleExtensionBackground();
