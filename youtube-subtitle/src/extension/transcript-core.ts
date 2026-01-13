/**
 * YouTube SubtitlePlus - 转录核心功能
 * ============================================
 * 处理 YouTube 页面上的转录面板操作
 */

import type { VideoInfo } from '../types';

interface TranscriptSegment {
  timeStr: string;
  text: string;
}

interface UserConfig {
  buttonIcons: {
    download: string;
    copy: string;
    translate: string;
  };
  fileNamingFormat: string;
  includeTimestamps: boolean;
  includeChapterHeaders: boolean;
  settingsGuide: boolean;
  autoOpenTranscript: boolean;
}

const USER_CONFIG: UserConfig = {
  buttonIcons: {
    download: '↓',
    copy: '📋',
    translate: '🚀',
  },
  fileNamingFormat: 'title-channel',
  includeTimestamps: true,
  includeChapterHeaders: true,
  settingsGuide: false,
  autoOpenTranscript: true,
};

function getWatchFlexyElement(): HTMLElement | null {
  return document.querySelector('ytd-watch-flexy');
}

function showNotification(message: string): void {
  const overlay = document.createElement('div');
  overlay.classList.add('YTSP-overlay');

  const modal = document.createElement('div');
  modal.classList.add('YTSP-notification');
  modal.textContent = message;

  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  setTimeout(() => overlay.remove(), 1000);
}

function getVideoInfo(): VideoInfo {
  const watchFlexyElement = getWatchFlexyElement();
  if (!watchFlexyElement) {
    return {
      ytTitle: 'N/A',
      channelName: 'N/A',
      uploadDate: 'N/A',
      videoURL: window.location.href,
      videoId: '',
    };
  }

  const ytTitle =
    watchFlexyElement.querySelector('div#title h1 > yt-formatted-string')?.textContent?.trim() ||
    'N/A';
  const channelName =
    watchFlexyElement
      .querySelector('ytd-video-owner-renderer ytd-channel-name#channel-name yt-formatted-string#text a')
      ?.textContent?.trim() || 'N/A';
  const uploadDate =
    watchFlexyElement
      .querySelector('ytd-video-primary-info-renderer #info-strings yt-formatted-string')
      ?.textContent?.trim() || 'N/A';
  const videoURL = window.location.href;
  const urlParams = new URLSearchParams(window.location.search);
  const videoId = urlParams.get('v') || '';

  return { ytTitle, channelName, uploadDate, videoURL, videoId };
}

function parseTimeSeconds(timeStr: string): number {
  const parts = timeStr.split(':').map(Number);
  let seconds = 0;
  if (parts.length === 3) {
    seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
  } else if (parts.length === 2) {
    seconds = parts[0] * 60 + parts[1];
  }
  return seconds;
}

function formatTimeSRT(seconds: number): string {
  const date = new Date(0);
  date.setSeconds(seconds);
  const iso = date.toISOString().substring(11, 19);
  return `${iso},000`;
}

function getTranscriptSegments(): TranscriptSegment[] {
  const watchFlexyElement = getWatchFlexyElement();
  if (!watchFlexyElement) return [];

  const transcriptContainer = watchFlexyElement.querySelector(
    'ytd-transcript-segment-list-renderer #segments-container'
  );
  if (!transcriptContainer) return [];

  const segments: TranscriptSegment[] = [];
  Array.from(transcriptContainer.children).forEach((element) => {
    if (element.tagName === 'YTD-TRANSCRIPT-SEGMENT-RENDERER') {
      const timeElement = element.querySelector('.segment-timestamp');
      const textElement = element.querySelector('.segment-text');
      if (timeElement && textElement) {
        segments.push({
          timeStr: timeElement.textContent?.trim() || '',
          text: textElement.textContent?.replace(/\s+/g, ' ').trim() || '',
        });
      }
    }
  });
  return segments;
}

function getTranscriptTextOnly(): string {
  const watchFlexyElement = getWatchFlexyElement();
  if (!watchFlexyElement) return '';

  const transcriptContainer = watchFlexyElement.querySelector(
    'ytd-transcript-segment-list-renderer #segments-container'
  );
  if (!transcriptContainer) return '';

  const lines: string[] = [];
  Array.from(transcriptContainer.children).forEach((element) => {
    if (element.tagName === 'YTD-TRANSCRIPT-SECTION-HEADER-RENDERER') {
      if (USER_CONFIG.includeChapterHeaders) {
        const chapterTitle = element.querySelector('h2 > span')?.textContent?.trim();
        if (chapterTitle) lines.push(`\nChapter: ${chapterTitle}`);
      }
    } else if (element.tagName === 'YTD-TRANSCRIPT-SEGMENT-RENDERER') {
      const textElement = element.querySelector('.segment-text');
      if (textElement) {
        lines.push(textElement.textContent?.replace(/\s+/g, ' ').trim() || '');
      }
    }
  });

  return lines.join('\n');
}

function getTranscriptSRT(): string {
  const segments = getTranscriptSegments();
  if (segments.length === 0) return '';

  let srtOutput = '';
  segments.forEach((seg, index) => {
    const startSeconds = parseTimeSeconds(seg.timeStr);
    let endSeconds = startSeconds + 5;

    if (index < segments.length - 1) {
      const nextStart = parseTimeSeconds(segments[index + 1].timeStr);
      if (nextStart > startSeconds) {
        endSeconds = nextStart;
      }
    }

    srtOutput += `${index + 1}\n`;
    srtOutput += `${formatTimeSRT(startSeconds)} --> ${formatTimeSRT(endSeconds)}\n`;
    srtOutput += `${seg.text}\n\n`;
  });

  return srtOutput;
}

function downloadTranscriptAsSRT(): void {
  const srtContent = getTranscriptSRT();
  if (!srtContent) {
    showNotification('字幕为空');
    return;
  }

  const { ytTitle, channelName, videoId } = getVideoInfo();
  const blob = new Blob([srtContent], { type: 'text/plain' });

  const sanitize = (str: string) => str.replace(/[<>:"/\\|?*]+/g, '');
  const fileName = `${sanitize(ytTitle)} - ${sanitize(channelName)}_${videoId}.srt`;

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showNotification('SRT 文件已下载');
}

function selectAndCopyTranscript(): void {
  const finalText = getTranscriptTextOnly();

  if (!finalText) {
    showNotification('Transcript is empty or not loaded.');
    return;
  }

  const { ytTitle, channelName, uploadDate, videoURL } = getVideoInfo();
  const fullContent = `Information about the YouTube Video:\nTitle: ${ytTitle}\nChannel: ${channelName}\nUpload Date: ${uploadDate}\nURL: ${videoURL}\n\n\nYouTube Transcript:\n${finalText.trimStart()}`;

  navigator.clipboard
    .writeText(fullContent)
    .then(() => {
      showNotification('字幕已复制');
    })
    .catch((err) => {
      console.error('Failed to copy: ', err);
      showNotification('复制失败');
    });
}

function openTranscript(): boolean {
  const transcriptButton =
    document.querySelector('#button-container button[aria-label="Show transcript"]') ||
    document.querySelector('button[aria-label="Show transcript"]');

  if (transcriptButton) {
    (transcriptButton as HTMLButtonElement).click();
    return true;
  }

  const engagementPanelSelector =
    'ytd-engagement-panel-section-list-renderer[target-id="engagement-panel-searchable-transcript"]';
  const engagementPanel = document.querySelector(engagementPanelSelector);

  if (engagementPanel) {
    window.dispatchEvent(new CustomEvent('YTSP_OpenTranscript'));
    return true;
  }

  return false;
}

function handleTranscriptAction(callback: () => void): void {
  const watchFlexyElement = getWatchFlexyElement();
  if (!watchFlexyElement) return;

  const transcriptContainer = watchFlexyElement.querySelector(
    'ytd-transcript-segment-list-renderer #segments-container'
  );
  if (transcriptContainer && transcriptContainer.children.length > 0) {
    callback();
    return;
  }

  if (openTranscript()) {
    showNotification('正在打开文字记录...');
    waitForTranscript(callback);
  } else {
    alert('无法获取文字记录，请确保视频有字幕');
  }
}

function checkAndOpenTranscript(): void {
  if (!USER_CONFIG.autoOpenTranscript) return;
  if (!location.href.includes('/watch')) return;

  const watchFlexyElement = getWatchFlexyElement();
  if (!watchFlexyElement) return;

  const transcriptContainer = watchFlexyElement.querySelector(
    'ytd-transcript-segment-list-renderer #segments-container'
  );
  if (transcriptContainer && transcriptContainer.children.length > 0) {
    return;
  }

  openTranscript();
}

function waitForTranscript(callback: () => void, retries = 0): void {
  const maxRetries = 20;
  const interval = 500;

  const transcriptContainer = getWatchFlexyElement()?.querySelector(
    'ytd-transcript-segment-list-renderer #segments-container'
  );

  if (transcriptContainer && transcriptContainer.children.length > 0) {
    callback();
  } else if (retries < maxRetries) {
    setTimeout(() => waitForTranscript(callback, retries + 1), interval);
  } else {
    showNotification('加载失败');
    alert('加载失败，请手动打开文字记录面板后重试');
  }
}

function handleDownloadClick(): void {
  handleTranscriptAction(downloadTranscriptAsSRT);
}

function handleCopyClick(): void {
  handleTranscriptAction(selectAndCopyTranscript);
}

function handleTranslateClick(): void {
  handleTranscriptAction(triggerExtensionTranslation);
}

function triggerExtensionTranslation(): void {
  showNotification('正在启动翻译...');
  window.dispatchEvent(new CustomEvent('YTSP_StartTranslation'));
}

interface ButtonConfig {
  id: string;
  text: string;
  clickHandler: () => void;
  tooltip: string;
}

function buttonLocation(buttons: ButtonConfig[], callback?: () => void): void {
  const masthead = document.querySelector('#end');

  if (masthead) {
    buttons.forEach(({ id, text, clickHandler, tooltip }) => {
      if (document.getElementById(id)) return;

      const buttonWrapper = document.createElement('div');
      buttonWrapper.classList.add('YTSP-button-wrapper');

      const button = document.createElement('button');
      button.id = id;
      button.textContent = text;
      button.classList.add('YTSP-button-style');
      button.addEventListener('click', clickHandler);

      const tooltipDiv = document.createElement('div');
      tooltipDiv.textContent = tooltip;
      tooltipDiv.classList.add('YTSP-button-tooltip');

      const arrowDiv = document.createElement('div');
      arrowDiv.classList.add('YTSP-button-tooltip-arrow');
      tooltipDiv.appendChild(arrowDiv);

      let tooltipTimeout: ReturnType<typeof setTimeout>;
      button.addEventListener('mouseenter', () => {
        tooltipTimeout = setTimeout(() => {
          tooltipDiv.style.visibility = 'visible';
          tooltipDiv.style.opacity = '1';
        }, 700);
      });

      button.addEventListener('mouseleave', () => {
        clearTimeout(tooltipTimeout);
        tooltipDiv.style.visibility = 'hidden';
        tooltipDiv.style.opacity = '0';
      });

      buttonWrapper.appendChild(button);
      buttonWrapper.appendChild(tooltipDiv);
      masthead.prepend(buttonWrapper);
    });
  } else {
    const observer = new MutationObserver((_mutations, obs) => {
      const mastheadEl = document.querySelector('#end');
      if (mastheadEl) {
        obs.disconnect();
        if (callback) callback();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
}

function createButtons(): void {
  const buttonsToCreate: ButtonConfig[] = [
    {
      id: 'transcript-download-button',
      text: USER_CONFIG.buttonIcons.download,
      clickHandler: handleDownloadClick,
      tooltip: '下载字幕',
    },
    {
      id: 'transcript-copy-button',
      text: USER_CONFIG.buttonIcons.copy,
      clickHandler: handleCopyClick,
      tooltip: '复制字幕',
    },
    {
      id: 'transcript-translate-button',
      text: USER_CONFIG.buttonIcons.translate,
      clickHandler: handleTranslateClick,
      tooltip: '开始翻译',
    },
  ];

  buttonLocation(buttonsToCreate, () => createButtons());
}

function init(): void {
  createButtons();

  setTimeout(checkAndOpenTranscript, 2000);

  const observer = new MutationObserver(() => {
    if (!document.getElementById('transcript-download-button') && document.querySelector('#end')) {
      createButtons();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });

  let lastUrl = location.href;
  new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      if (location.href.includes('/watch')) {
        setTimeout(checkAndOpenTranscript, 2500);
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
