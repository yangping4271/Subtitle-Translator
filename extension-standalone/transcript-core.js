(function () {
    'use strict';

    // Configuration
    const USER_CONFIG = {
        buttonIcons: {
            download: '↓',
            copy: '📋',
            translate: '🚀'
        },
        fileNamingFormat: 'title-channel',
        includeTimestamps: true,
        includeChapterHeaders: true,
        settingsGuide: false,
        autoOpenTranscript: true
    };

    // Helper to get the main video element container
    function getWatchFlexyElement() {
        return document.querySelector('ytd-watch-flexy');
    }

    // Notification
    function showNotification(message) {
        const overlay = document.createElement('div');
        overlay.classList.add('YTSP-overlay');

        const modal = document.createElement('div');
        modal.classList.add('YTSP-notification');
        modal.textContent = message;

        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        setTimeout(() => { overlay.remove(); }, 1000);
    }

    // Get Video Info
    function getVideoInfo() {
        const watchFlexyElement = getWatchFlexyElement();
        if (!watchFlexyElement) return { ytTitle: 'N/A', channelName: 'N/A', uploadDate: 'N/A', videoURL: window.location.href, videoId: '' };

        const ytTitle = watchFlexyElement.querySelector('div#title h1 > yt-formatted-string')?.textContent.trim() || 'N/A';
        const channelName = watchFlexyElement.querySelector('ytd-video-owner-renderer ytd-channel-name#channel-name yt-formatted-string#text a')?.textContent.trim() || 'N/A';
        const uploadDate = watchFlexyElement.querySelector('ytd-video-primary-info-renderer #info-strings yt-formatted-string')?.textContent.trim() || 'N/A';
        const videoURL = window.location.href;
        const urlParams = new URLSearchParams(window.location.search);
        const videoId = urlParams.get('v') || '';

        return { ytTitle, channelName, uploadDate, videoURL, videoId };
    }

    // Helper to parse time string to seconds
    function parseTimeSeconds(timeStr) {
        const parts = timeStr.split(':').map(Number);
        let seconds = 0;
        if (parts.length === 3) {
            seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
        } else if (parts.length === 2) {
            seconds = parts[0] * 60 + parts[1];
        }
        return seconds;
    }

    // Helper to format seconds to SRT timestamp
    function formatTimeSRT(seconds) {
        const date = new Date(0);
        date.setSeconds(seconds);
        const iso = date.toISOString().substr(11, 8);
        return `${iso},000`;
    }

    // Get Transcript Segments (Structured)
    function getTranscriptSegments() {
        const watchFlexyElement = getWatchFlexyElement();
        if (!watchFlexyElement) return [];

        const transcriptContainer = watchFlexyElement.querySelector('ytd-transcript-segment-list-renderer #segments-container');
        if (!transcriptContainer) return [];

        const segments = [];
        [...transcriptContainer.children].forEach(element => {
            if (element.tagName === 'YTD-TRANSCRIPT-SEGMENT-RENDERER') {
                const timeElement = element.querySelector('.segment-timestamp');
                const textElement = element.querySelector('.segment-text');
                if (timeElement && textElement) {
                    segments.push({
                        timeStr: timeElement.textContent.trim(),
                        text: textElement.textContent.replace(/\s+/g, ' ').trim()
                    });
                }
            }
        });
        return segments;
    }

    // Generate Text for Copy (No Timestamps)
    function getTranscriptTextOnly() {
        const watchFlexyElement = getWatchFlexyElement();
        if (!watchFlexyElement) return '';

        const transcriptContainer = watchFlexyElement.querySelector('ytd-transcript-segment-list-renderer #segments-container');
        if (!transcriptContainer) return '';

        const lines = [];
        [...transcriptContainer.children].forEach(element => {
            if (element.tagName === 'YTD-TRANSCRIPT-SECTION-HEADER-RENDERER') {
                if (USER_CONFIG.includeChapterHeaders) {
                    const chapterTitle = element.querySelector('h2 > span')?.textContent.trim();
                    if (chapterTitle) lines.push(`\nChapter: ${chapterTitle}`);
                }
            } else if (element.tagName === 'YTD-TRANSCRIPT-SEGMENT-RENDERER') {
                const textElement = element.querySelector('.segment-text');
                if (textElement) {
                    lines.push(textElement.textContent.replace(/\s+/g, ' ').trim());
                }
            }
        });

        return lines.join('\n');
    }

    // Generate SRT for Download
    function getTranscriptSRT() {
        const segments = getTranscriptSegments();
        if (segments.length === 0) return '';

        let srtOutput = '';
        segments.forEach((seg, index) => {
            const startSeconds = parseTimeSeconds(seg.timeStr);
            let endSeconds = startSeconds + 5; // Default duration

            // Try to get start of next segment as end of current
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

    function downloadTranscriptAsSRT() {
        const srtContent = getTranscriptSRT();
        if (!srtContent) {
            showNotification('字幕为空');
            return;
        }

        const { ytTitle, channelName, videoId } = getVideoInfo();
        const blob = new Blob([srtContent], { type: 'text/plain' });

        const sanitize = str => str.replace(/[<>:"/\\|?*]+/g, '');

        // Changed extension to .srt and added video ID
        let fileName = `${sanitize(ytTitle)} - ${sanitize(channelName)}_${videoId}.srt`;

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

    function selectAndCopyTranscript() {
        // Use the TextOnly version (No timestamps)
        const finalText = getTranscriptTextOnly();

        if (!finalText) {
            showNotification('Transcript is empty or not loaded.');
            return;
        }

        const { ytTitle, channelName, uploadDate, videoURL } = getVideoInfo();
        const fullContent = `Information about the YouTube Video:\nTitle: ${ytTitle}\nChannel: ${channelName}\nUpload Date: ${uploadDate}\nURL: ${videoURL}\n\n\nYouTube Transcript:\n${finalText.trimStart()}`;

        navigator.clipboard.writeText(fullContent).then(() => {
            showNotification('字幕已复制');
        }).catch(err => {
            console.error('Failed to copy: ', err);
            showNotification('复制失败');
        });
    }

    function openTranscript() {
        // Try to find the "Show transcript" button and click it
        const transcriptButton = document.querySelector('#button-container button[aria-label="Show transcript"]') ||
            document.querySelector('button[aria-label="Show transcript"]');

        if (transcriptButton) {
            transcriptButton.click();
            return true;
        }

        // If button not found, try to force open via Main World script
        const engagementPanelSelector = 'ytd-engagement-panel-section-list-renderer[target-id="engagement-panel-searchable-transcript"]';
        const engagementPanel = document.querySelector(engagementPanelSelector);

        if (engagementPanel) {
            // Dispatch event to be caught by inject.js in the Main World
            window.dispatchEvent(new CustomEvent('YTSP_OpenTranscript'));
            return true;
        }

        return false;
    }

    function handleTranscriptAction(callback) {
        const watchFlexyElement = getWatchFlexyElement();
        if (!watchFlexyElement) return;

        // 1. Check if transcript is already loaded
        const transcriptContainer = watchFlexyElement.querySelector('ytd-transcript-segment-list-renderer #segments-container');
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

    function checkAndOpenTranscript() {
        if (!USER_CONFIG.autoOpenTranscript) return;

        // Simple check if we are on a watch page
        if (!location.href.includes('/watch')) return;

        const watchFlexyElement = getWatchFlexyElement();
        if (!watchFlexyElement) return;

        // Check if transcript is already loaded/visible
        const transcriptContainer = watchFlexyElement.querySelector('ytd-transcript-segment-list-renderer #segments-container');
        if (transcriptContainer && transcriptContainer.children.length > 0) {
            return;
        }

        // Attempt to open
        openTranscript();
    }

    function waitForTranscript(callback, retries = 0) {
        const maxRetries = 20; // Wait up to ~10 seconds (20 * 500ms)
        const interval = 500;

        const transcriptContainer = getWatchFlexyElement()?.querySelector('ytd-transcript-segment-list-renderer #segments-container');

        if (transcriptContainer && transcriptContainer.children.length > 0) {
            callback();
        } else if (retries < maxRetries) {
            setTimeout(() => waitForTranscript(callback, retries + 1), interval);
        } else {
            showNotification('加载失败');
            alert('加载失败，请手动打开文字记录面板后重试');
        }
    }

    function handleDownloadClick() {
        handleTranscriptAction(downloadTranscriptAsSRT);
    }

    function handleCopyClick() {
        handleTranscriptAction(selectAndCopyTranscript);
    }

    function handleTranslateClick() {
        handleTranscriptAction(triggerExtensionTranslation);
    }

    function triggerExtensionTranslation() {
        showNotification('正在启动翻译...');

        // 触发扩展的弹出窗口打开翻译
        // 由于无法直接从页面脚本调用 chrome.runtime,
        // 我们通过自定义事件通知 content.js
        window.dispatchEvent(new CustomEvent('YTSP_StartTranslation'));
    }

    // pollForTranslation 已删除,翻译由扩展后台处理

    function buttonLocation(buttons, callback) {
        const masthead = document.querySelector('#end'); // Masthead end section

        if (masthead) {
            buttons.forEach(({ id, text, clickHandler, tooltip }) => {
                if (document.getElementById(id)) return; // Prevent duplicates

                // button wrapper
                const buttonWrapper = document.createElement('div');
                buttonWrapper.classList.add('YTSP-button-wrapper');

                // buttons
                const button = document.createElement('button');
                button.id = id;
                button.textContent = text;
                button.classList.add('YTSP-button-style');
                button.addEventListener('click', clickHandler);

                // tooltip div
                const tooltipDiv = document.createElement('div');
                tooltipDiv.textContent = tooltip;
                tooltipDiv.classList.add('YTSP-button-tooltip');

                // tooltip arrow
                const arrowDiv = document.createElement('div');
                arrowDiv.classList.add('YTSP-button-tooltip-arrow');
                tooltipDiv.appendChild(arrowDiv);

                // show and hide tooltip on hover
                let tooltipTimeout;
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

                // append button elements
                buttonWrapper.appendChild(button);
                buttonWrapper.appendChild(tooltipDiv);

                // Insert before the profile icon or other end elements
                // The original script uses `masthead.prepend(buttonWrapper)`.
                masthead.prepend(buttonWrapper);
            });
        } else {
            // Retry if masthead is not ready
            const observer = new MutationObserver((mutations, obs) => {
                const masthead = document.querySelector('#end');
                if (masthead) {
                    obs.disconnect();
                    if (callback) callback();
                }
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
    }

    function createButtons() {
        const buttonsToCreate = [
            {
                id: 'transcript-download-button',
                text: USER_CONFIG.buttonIcons.download,
                clickHandler: handleDownloadClick,
                tooltip: '下载字幕'
            },
            {
                id: 'transcript-copy-button',
                text: USER_CONFIG.buttonIcons.copy,
                clickHandler: handleCopyClick,
                tooltip: '复制字幕'
            },
            {
                id: 'transcript-translate-button',
                text: USER_CONFIG.buttonIcons.translate,
                clickHandler: handleTranslateClick,
                tooltip: '开始翻译'
            }
        ];

        buttonLocation(buttonsToCreate, () => createButtons());
    }

    // Initialize
    function init() {
        createButtons();

        // Initial check for auto-open
        setTimeout(checkAndOpenTranscript, 2000);

        // Observe for navigation or DOM changes that might remove our buttons
        const observer = new MutationObserver(() => {
            if (!document.getElementById('transcript-download-button') && document.querySelector('#end')) {
                createButtons();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // URL change observer for auto-open
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

})();
