// YouTube字幕数据获取管理器
class YouTubeSubtitleFetcher {
  constructor() {
    this.videoId = null;
    this.videoDurationSec = null;
    this.captionTracks = [];
    this.fullSubtitles = null;
    this.logger = window.debugLogger;
    
    // 第三方工具已精简整合到主方法中
  }

  // 获取当前视频ID
  getCurrentVideoId() {
    const urlParams = new URLSearchParams(window.location.search);
    const videoId = urlParams.get('v');
    
    if (videoId && videoId !== this.videoId) {
      this.videoId = videoId;
      this.logger.info('🎥 检测到新视频ID:', videoId);
      return true; // 视频发生变化
    }
    
    return false; // 视频未变化
  }

  // 获取字幕轨道信息
  async getCaptionTracks() {
    try {
      // 从YouTube播放器配置中获取字幕信息
      const ytInitialPlayerResponse = this.extractPlayerResponse();
      if (!ytInitialPlayerResponse) {
        throw new Error('无法获取播放器配置');
      }

      const captions = ytInitialPlayerResponse.captions;
      if (!captions || !captions.playerCaptionsTracklistRenderer) {
        throw new Error('视频没有可用的字幕');
      }

      this.captionTracks = captions.playerCaptionsTracklistRenderer.captionTracks || [];
      this.logger.info('🎯 找到字幕轨道:', this.captionTracks.length + '个', this.captionTracks);
      
      // 优先选择英文字幕
      const englishTrack = this.captionTracks.find(track => {
        if (!track) return false;
        
        // 检查语言代码
        if (track.languageCode === 'en' || track.languageCode === 'en-US') {
          return true;
        }
        
        // 检查名称（安全访问）
        const trackName = track.name?.simpleText || track.name?.runs?.[0]?.text || '';
        if (trackName.toLowerCase().includes('english')) {
          return true;
        }
        
        return false;
      });

      if (englishTrack) {
        this.logger.info('✅ 找到英文字幕轨道:', englishTrack);
        return englishTrack;
      }

      // 如果没有英文字幕，选择第一个可用的
      if (this.captionTracks.length > 0) {
        this.logger.warn('⚠️ 未找到英文字幕，使用第一个可用字幕');
        return this.captionTracks[0];
      }

      throw new Error('没有找到任何可用的字幕轨道');

    } catch (error) {
      this.logger.error('❌ 获取字幕轨道失败:', error.message);
      return null;
    }
  }

  // 从页面中提取播放器配置
  extractPlayerResponse() {
    try {
      this.logger.info('🔍 开始提取播放器配置...');
      
      // 方法1: 从页面script标签中提取 (主要方法)
      const scripts = document.querySelectorAll('script');
      for (const script of scripts) {
        const content = script.textContent;
        if (content && content.includes('ytInitialPlayerResponse')) {
          // 改进的JSON提取方法：处理嵌套的花括号
          const startIndex = content.indexOf('ytInitialPlayerResponse');
          if (startIndex !== -1) {
            const jsonStart = content.indexOf('{', startIndex);
            if (jsonStart !== -1) {
              // 手动解析JSON，处理嵌套括号
              let braceCount = 0;
              let jsonEnd = jsonStart;
              
              for (let i = jsonStart; i < content.length; i++) {
                if (content[i] === '{') braceCount++;
                if (content[i] === '}') braceCount--;
                if (braceCount === 0) {
                  jsonEnd = i;
                  break;
                }
              }
              
              if (jsonEnd > jsonStart) {
                const jsonString = content.substring(jsonStart, jsonEnd + 1);
                try {
                  const parsed = JSON.parse(jsonString);
                  this.logger.info('✅ 从script标签提取播放器配置成功');
                  this.analyzePlayerResponse(parsed); // 添加详细分析
                  return parsed;
                } catch (parseError) {
                  this.logger.warn('⚠️ script标签JSON解析失败:', parseError.message);
                }
              }
            }
          }
        }
      }

      // 方法2: 从window对象中获取
      if (window.ytInitialPlayerResponse) {
        this.logger.info('✅ 从window对象获取播放器配置成功');
        this.analyzePlayerResponse(window.ytInitialPlayerResponse);
        return window.ytInitialPlayerResponse;
      }

      // 方法3: 从ytcfg中获取
      if (window.ytcfg && window.ytcfg.data_ && window.ytcfg.data_.PLAYER_VARS) {
        const playerVars = window.ytcfg.data_.PLAYER_VARS;
        if (playerVars.player_response) {
          const parsed = JSON.parse(playerVars.player_response);
          this.logger.info('✅ 从ytcfg获取播放器配置成功');
          this.analyzePlayerResponse(parsed);
          return parsed;
        }
      }

      // 方法4: 从页面URL中搜索timedtext链接 (新增)
      this.extractTimedtextFromPageSource();

      throw new Error('无法从页面中提取播放器配置');
    } catch (error) {
      this.logger.error('❌ 提取播放器配置失败:', error.message);
      return null;
    }
  }

  // 新增：详细分析播放器响应数据
  analyzePlayerResponse(playerResponse) {
    try {
      this.logger.debug('📊 分析播放器响应数据:', {
        hasVideoDetails: !!playerResponse.videoDetails,
        videoId: playerResponse.videoDetails?.videoId,
        title: playerResponse.videoDetails?.title,
        hasCaptions: !!playerResponse.captions,
        captionTracks: playerResponse.captions?.playerCaptionsTracklistRenderer?.captionTracks?.length || 0
      });

      // 记录视频总时长用于完整性校验
      if (playerResponse.videoDetails?.lengthSeconds) {
        const len = parseInt(playerResponse.videoDetails.lengthSeconds, 10);
        if (!Number.isNaN(len) && len > 0) {
          this.videoDurationSec = len;
          this.logger.debug('⏱️ 视频总时长(秒):', this.videoDurationSec);
        }
      }

      // 分析字幕轨道详情
      if (playerResponse.captions?.playerCaptionsTracklistRenderer?.captionTracks) {
        const tracks = playerResponse.captions.playerCaptionsTracklistRenderer.captionTracks;
        tracks.forEach((track, index) => {
          this.logger.debug(`📝 字幕轨道 ${index + 1}:`, {
            languageCode: track.languageCode,
            name: track.name?.simpleText || track.name?.runs?.[0]?.text,
            baseUrl: track.baseUrl ? `${track.baseUrl.substring(0, 80)}...` : 'None',
            isTranslatable: track.isTranslatable,
            kind: track.kind
          });
        });
      }
    } catch (error) {
      this.logger.warn('⚠️ 播放器响应分析失败:', error.message);
    }
  }

  // 新增：从页面源码直接搜索timedtext链接
  extractTimedtextFromPageSource() {
    try {
      this.logger.info('🔍 尝试从页面源码提取timedtext链接...');
      
      const scripts = document.querySelectorAll('script');
      for (const script of scripts) {
        const content = script.textContent;
        if (content && content.includes('timedtext')) {
          // 搜索timedtext URLs
          const timedtextRegex = /"(https:\/\/www\.youtube\.com\/api\/timedtext[^"]+)"/g;
          let match;
          const urls = [];
          
          while ((match = timedtextRegex.exec(content)) !== null) {
            urls.push(match[1]);
          }
          
          if (urls.length > 0) {
            this.logger.info('🎯 找到timedtext URLs:', urls.length + '个');
            urls.forEach((url, index) => {
              this.logger.debug(`URL ${index + 1}:`, url.substring(0, 120) + '...');
            });
            
            // 尝试直接获取字幕
            return this.fetchSubtitlesFromUrls(urls);
          }
        }
      }
    } catch (error) {
      this.logger.warn('⚠️ timedtext提取失败:', error.message);
    }
    return null;
  }

  // 新增：从播放器脚本中提取字幕
  async extractFromPlayerScripts() {
    try {
      this.logger.info('🔍 从播放器脚本提取字幕...');
      
      // 查找所有相关的script标签
      const scripts = document.querySelectorAll('script[src*="player"], script[src*="base"]');
      
      for (const script of scripts) {
        if (script.src) {
          try {
            const response = await fetch(script.src);
            const content = await response.text();
            
            if (content.includes('timedtext') || content.includes('captions')) {
              // 提取timedtext URLs
              const urls = this.extractTimedTextUrls(content);
              if (urls.length > 0) {
                return await this.fetchSubtitlesFromUrls(urls);
              }
            }
          } catch (error) {
            this.logger.debug('⚠️ 脚本获取失败:', error.message);
            continue;
          }
        }
      }
      
      return null;
    } catch (error) {
      this.logger.warn('⚠️ 播放器脚本提取失败:', error.message);
      return null;
    }
  }

  // 新增：从YouTube内部API提取字幕
  async extractFromYouTubeAPI() {
    try {
      this.logger.info('🌐 尝试YouTube内部API...');
      
      // 构建内部API调用
      const apiUrl = 'https://www.youtube.com/youtubei/v1/player';
      const payload = {
        context: {
          client: {
            clientName: 'WEB',
            clientVersion: '2.20241201.00.00'
          }
        },
        videoId: this.videoId
      };

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'Mozilla/5.0 (compatible; YouTube-API-Client)',
        },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        const data = await response.json();
        
        if (data.captions?.playerCaptionsTracklistRenderer?.captionTracks) {
          const tracks = data.captions.playerCaptionsTracklistRenderer.captionTracks;
          const englishTrack = tracks.find(track => 
            track.languageCode === 'en' || 
            track.languageCode === 'en-US'
          ) || tracks[0];

          if (englishTrack && englishTrack.baseUrl) {
            this.logger.info('✅ YouTube API获取字幕轨道成功');
            return await this.fetchFullSubtitles(englishTrack);
          }
        }
      }

      return null;
    } catch (error) {
      this.logger.warn('⚠️ YouTube API提取失败:', error.message);
      return null;
    }
  }

  // 辅助方法：提取timedtext URLs
  extractTimedTextUrls(content) {
    const urls = [];
    const patterns = [
      /\"(https:\/\/www\.youtube\.com\/api\/timedtext[^\"]*)\"/g,
      /\'(https:\/\/www\.youtube\.com\/api\/timedtext[^\']*)\'/g,
      /(https:\/\/www\.youtube\.com\/api\/timedtext[^\s\,\)]*)/g
    ];

    for (const pattern of patterns) {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const url = match[1] || match[0];
        if (url && !urls.includes(url)) {
          urls.push(url.replace(/\\u0026/g, '&'));
        }
      }
    }

    return urls;
  }

  // 新增：尝试从找到的URLs获取字幕
  async fetchSubtitlesFromUrls(urls) {
    for (const url of urls) {
      try {
        this.logger.info('🌐 尝试获取字幕:', url.substring(0, 80) + '...');
        
        const response = await fetch(url);
        if (response.ok) {
          const text = await response.text();
          
          if (text && text.trim().length > 0) {
            this.logger.info('✅ 成功获取字幕数据:', text.length + ' 字符');
            
            // 尝试解析为JSON或XML
            try {
              const jsonData = JSON.parse(text);
              if (jsonData.events) {
                this.fullSubtitles = this.parseSubtitleEvents(jsonData.events);
                this.logger.info('✅ JSON字幕解析成功:', this.fullSubtitles.length + '条');
                return this.fullSubtitles;
              }
            } catch (jsonError) {
              this.logger.debug('⚠️ JSON解析失败，尝试XML解析');
              // 可以在这里添加XML解析逻辑
            }
          }
        }
      } catch (error) {
        this.logger.debug('⚠️ URL获取失败:', error.message);
        continue; // 继续尝试下一个URL
      }
    }
    return null;
  }

  // 下载完整字幕数据
  async fetchFullSubtitles(captionTrack) {
    try {
      if (!captionTrack || !captionTrack.baseUrl) {
        throw new Error('无效的字幕轨道');
      }

      // 构建字幕数据请求URL，指定格式
      const subtitleUrl = captionTrack.baseUrl + '&fmt=json3';
      this.logger.info('📡 开始下载字幕数据:', subtitleUrl);

      const response = await fetch(subtitleUrl);
      if (!response.ok) {
        throw new Error(`字幕下载失败: ${response.status}`);
      }

      // 增强的响应处理
      const responseText = await response.text();
      this.logger.debug('📄 响应长度:', responseText.length + '字符');
      
      if (!responseText || responseText.trim().length === 0) {
        throw new Error('响应内容为空');
      }

      let data;
      try {
        // 尝试JSON解析
        data = JSON.parse(responseText);
        this.logger.debug('✅ JSON解析成功');
      } catch (jsonError) {
        this.logger.warn('⚠️ JSON解析失败，尝试XML格式:', jsonError.message);
        
        // 尝试XML格式作为fallback
        try {
          data = await this.parseXMLSubtitles(responseText);
          this.logger.info('✅ XML格式解析成功');
        } catch (xmlError) {
          this.logger.error('❌ XML解析也失败:', xmlError.message);
          
          // 提供详细的诊断信息
          this.logger.debug('📊 响应诊断:', {
            length: responseText.length,
            firstChars: responseText.substring(0, 100),
            lastChars: responseText.substring(Math.max(0, responseText.length - 100)),
            contentType: response.headers.get('content-type'),
            isValidJSON: this.isValidJSON(responseText),
            hasXMLStructure: responseText.includes('<')
          });
          
          throw new Error(`无法解析响应格式: JSON错误=${jsonError.message}, XML错误=${xmlError.message}`);
        }
      }

      if (!data.events && !data.transcript) {
        throw new Error('字幕数据格式错误：缺少events或transcript字段');
      }

      // 解析字幕事件
      const events = data.events || data.transcript || [];
      this.fullSubtitles = this.parseSubtitleEvents(events);
      this.logger.info('✅ 字幕数据下载成功:', this.fullSubtitles.length + '条');
      
      return this.fullSubtitles;

    } catch (error) {
      this.logger.error('❌ 下载字幕数据失败:', error.message);
      return null;
    }
  }

  // 解析字幕事件数据
  parseSubtitleEvents(events) {
    const subtitles = [];
    
    for (const event of events) {
      if (!event.segs) continue; // 跳过没有文本的事件

      const startTime = event.tStartMs / 1000; // 转换为秒
      const duration = event.dDurationMs ? event.dDurationMs / 1000 : 0;
      const endTime = startTime + duration;

      // 合并segments中的文本
      let text = '';
      for (const seg of event.segs) {
        if (seg.utf8) {
          text += seg.utf8;
        }
      }

      // 清理文本
      text = text.replace(/\n/g, ' ').trim();
      
      if (text) {
        subtitles.push({
          startTime,
          endTime,
          duration,
          text
        });
      }
    }

    return subtitles;
  }

  // 验证JSON格式
  isValidJSON(text) {
    try {
      JSON.parse(text);
      return true;
    } catch {
      return false;
    }
  }

  // XML格式字幕解析
  async parseXMLSubtitles(xmlText) {
    try {
      const parser = new DOMParser();
      const xmlDoc = parser.parseFromString(xmlText, 'text/xml');
      
      // 检查解析错误
      const parseError = xmlDoc.querySelector('parsererror');
      if (parseError) {
        throw new Error('XML解析错误: ' + parseError.textContent);
      }

      const events = [];
      
      // 尝试解析不同的XML格式
      let textElements = xmlDoc.querySelectorAll('text');
      if (textElements.length === 0) {
        textElements = xmlDoc.querySelectorAll('p'); // TTML格式
      }
      
      if (textElements.length === 0) {
        throw new Error('XML中未找到文本元素');
      }

      for (const element of textElements) {
        const startAttr = element.getAttribute('start') || element.getAttribute('t');
        const durationAttr = element.getAttribute('dur') || element.getAttribute('d');
        
        if (startAttr) {
          const startTime = this.parseTimeToSeconds(startAttr);
          const duration = durationAttr ? this.parseTimeToSeconds(durationAttr) : 3;
          const text = element.textContent.trim();
          
          if (text) {
            events.push({
              tStartMs: startTime * 1000,
              dDurationMs: duration * 1000,
              segs: [{ utf8: text }]
            });
          }
        }
      }

      return { events };
    } catch (error) {
      throw new Error('XML解析失败: ' + error.message);
    }
  }

  // 时间格式转换（支持多种格式）
  parseTimeToSeconds(timeStr) {
    if (!timeStr) return 0;
    
    // 如果是纯数字（毫秒）
    if (/^\d+$/.test(timeStr)) {
      return parseFloat(timeStr) / 1000;
    }
    
    // 如果是秒数格式
    if (/^\d+(\.\d+)?s?$/.test(timeStr)) {
      return parseFloat(timeStr.replace('s', ''));
    }
    
    // 如果是时:分:秒格式
    if (timeStr.includes(':')) {
      const parts = timeStr.split(':');
      let seconds = 0;
      for (let i = 0; i < parts.length; i++) {
        seconds = seconds * 60 + parseFloat(parts[i]);
      }
      return seconds;
    }
    
    return parseFloat(timeStr) || 0;
  }

  // 根据时间获取字幕文本
  getSubtitleAtTime(currentTime) {
    if (!this.fullSubtitles) return null;

    for (const subtitle of this.fullSubtitles) {
      if (currentTime >= subtitle.startTime && currentTime <= subtitle.endTime) {
        return subtitle;
      }
    }

    return null;
  }

  // 获取指定时间范围的字幕
  getSubtitlesInRange(startTime, endTime) {
    if (!this.fullSubtitles) return [];

    return this.fullSubtitles.filter(subtitle => 
      subtitle.startTime >= startTime && subtitle.endTime <= endTime
    );
  }

  // 获取所有字幕文本用于批量翻译
  getAllSubtitleTexts() {
    if (!this.fullSubtitles) return [];
    
    return this.fullSubtitles.map(subtitle => ({
      index: this.fullSubtitles.indexOf(subtitle),
      text: subtitle.text,
      startTime: subtitle.startTime,
      endTime: subtitle.endTime
    }));
  }

  // 只保留最可靠的字幕获取方法
  async getSubtitlesWithEnhancedMethods() {
    this.logger.info('🚀 启动最可靠的字幕获取流程...');
    
    const methods = [
      {
        name: '经过验证的完整字幕获取',
        method: async () => {
          return await this.getVerifiedCompleteSubtitles();
        }
      },
      {
        name: 'yt-dlp风格字幕获取（回退方案）',
        method: async () => {
          return await this.getSubtitlesWithYTDLP();
        }
      }
    ];

    for (const methodInfo of methods) {
      try {
        this.logger.info(`🔄 尝试方法: ${methodInfo.name}`);
        const result = await methodInfo.method();
        
        if (result && result.length > 10) { // 至少要有10条字幕才算成功
          this.logger.info(`✅ ${methodInfo.name} 成功获取 ${result.length} 条字幕`);
          this.fullSubtitles = result;
          return result;
        } else if (result) {
          this.logger.warn(`⚠️ ${methodInfo.name} 只获取到 ${result.length} 条字幕，数量不足`);
        }
      } catch (error) {
        this.logger.error(`❌ ${methodInfo.name} 失败:`, error.message);
        continue;
      }
    }

    this.logger.error('❌ 字幕获取失败，回退到实时翻译模式');
    return null;
  }

  // 经过验证的完整字幕获取方法
  async getVerifiedCompleteSubtitles() {
    this.logger.info('🎯 开始经过验证的完整字幕获取...');
    
    try {
      // 第一步：获取字幕轨道信息
      const playerResponse = this.extractPlayerResponse();
      if (!playerResponse?.captions?.playerCaptionsTracklistRenderer?.captionTracks) {
        throw new Error('无法获取播放器字幕轨道信息');
      }

      const tracks = playerResponse.captions.playerCaptionsTracklistRenderer.captionTracks;
      this.logger.info('📋 找到字幕轨道:', tracks.length + '个');
      
      // 第二步：选择最好的英文字幕轨道
      const bestTrack = this.findBestEnglishTrack(tracks);
      if (!bestTrack?.baseUrl) {
        throw new Error('无法找到合适的英文字幕轨道');
      }

      this.logger.info('✅ 选定字幕轨道:', {
        language: bestTrack.languageCode,
        name: bestTrack.name?.simpleText || '自动生成',
        isAsr: bestTrack.kind === 'asr'
      });

      // 第三步：并行尝试多种URL与格式组合，选取最佳
      const candidateUrls = this.buildCandidateUrls(bestTrack.baseUrl);
      this.logger.info(`🔎 生成候选URL ${candidateUrls.length} 个，开始并行请求`);

      const fetchHeaders = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.youtube.com/'
      };

      const results = await Promise.allSettled(candidateUrls.map(async (info) => {
        try {
          const res = await fetch(info.url, { headers: fetchHeaders });
          if (!res.ok) throw new Error(String(res.status));
          const text = await res.text();
          if (!text || text.trim().length < 50) throw new Error('响应过短');

          let subs = [];
          switch (info.format) {
            case 'srt':
              subs = this.parseSRTFormat(text); break;
            case 'vtt':
              subs = this.parseVTTFormat(text); break;
            case 'json3':
              try {
                const data = JSON.parse(text);
                subs = data?.events ? this.parseSubtitleEvents(data.events) : [];
              } catch { subs = []; }
              break;
            case 'ttml':
            case 'srv3':
              try { subs = await this.parseXMLSubtitles(text).then(d => this.parseSubtitleEvents(d.events)); }
              catch { subs = []; }
              break;
            default:
              subs = [];
          }

          const normalized = this.ensureContinuityAndSort(subs);
          const quality = this.validateSubtitleQuality(normalized);
          return { ok: true, info, subs: normalized, quality };
        } catch (e) {
          return { ok: false, info, error: e?.message || String(e) };
        }
      }));

      // 选取质量最佳的字幕
      let best = null;
      for (const r of results) {
        if (r.status === 'fulfilled' && r.value.ok && r.value.quality.isValid) {
          if (!best) best = r.value; else {
            const better = (parseFloat((r.value.quality.coverageRatio||'0%')) > parseFloat((best.quality.coverageRatio||'0%')))
                        || (r.value.subs.length > best.subs.length);
            if (better) best = r.value;
          }
        }
      }

      if (best) {
        this.logger.info('✅ 选取最佳字幕来源:', { format: best.info.format, url: best.info.url.substring(0, 120) + '...', quality: best.quality });
        return best.subs;
      }

      throw new Error('所有候选URL均未获得可用字幕');

    } catch (error) {
      this.logger.error('❌ 经过验证的字幕获取失败:', error.message);
      throw error;
    }
  }

  // 寻找最佳的英文字幕轨道
  findBestEnglishTrack(tracks) {
    // 优先级排序
    const priorities = [
      // 1. 手动制作的英文字幕（最高优先级）
      track => track.languageCode === 'en' && !track.kind,
      track => track.languageCode === 'en-US' && !track.kind,
      
      // 2. 自动生成的英文字幕
      track => track.languageCode === 'en' && track.kind === 'asr',
      track => track.languageCode === 'en-US' && track.kind === 'asr',
      
      // 3. 其他英文变体
      track => track.languageCode?.startsWith('en'),
      
      // 4. 任何ASR字幕
      track => track.kind === 'asr'
    ];

    for (const priority of priorities) {
      const track = tracks.find(priority);
      if (track) {
        return track;
      }
    }

    // 5. 如果都没有，返回第一个
    return tracks[0];
  }

  // 验证字幕质量
  validateSubtitleQuality(subtitles) {
    if (!subtitles || subtitles.length === 0) {
      return { isValid: false, reason: '字幕为空' };
    }

    if (subtitles.length < 10) {
      return { isValid: false, reason: '字幕数量太少', count: subtitles.length };
    }

    // 检查时间有效性
    let validTimeCount = 0;
    let textValidCount = 0;
    let totalDuration = 0;

    for (const sub of subtitles) {
      if (typeof sub.startTime === 'number' && typeof sub.endTime === 'number' && 
          sub.startTime >= 0 && sub.endTime > sub.startTime) {
        validTimeCount++;
        totalDuration += (sub.endTime - sub.startTime);
      }

      if (typeof sub.text === 'string' && sub.text.trim().length > 0) {
        textValidCount++;
      }
    }

    const timeValidRatio = validTimeCount / subtitles.length;
    const textValidRatio = textValidCount / subtitles.length;
    const avgDuration = totalDuration / validTimeCount || 0;

    // 覆盖率估算：总字幕跨度时间/视频总时长（粗略，但能过滤不完整数据）
    let coverageRatio = null;
    if (this.videoDurationSec && this.videoDurationSec > 0) {
      coverageRatio = Math.min(1, totalDuration / this.videoDurationSec);
    }

    const isValid = (
      timeValidRatio >= 0.9 &&
      textValidRatio >= 0.9 &&
      avgDuration > 0.5 &&
      (coverageRatio === null || coverageRatio >= 0.6)
    );

    return {
      isValid,
      count: subtitles.length,
      timeValidRatio: (timeValidRatio * 100).toFixed(1) + '%',
      textValidRatio: (textValidRatio * 100).toFixed(1) + '%',
      avgDuration: avgDuration.toFixed(2) + 's',
      totalDuration: totalDuration.toFixed(1) + 's',
      videoDurationSec: this.videoDurationSec || undefined,
      coverageRatio: coverageRatio !== null ? (coverageRatio * 100).toFixed(1) + '%' : '未知'
    };
  }

  // 方法1: 可靠的VTT格式获取
  async getReliableVTTSubtitles() {
    try {
      this.logger.info('📡 获取VTT格式字幕...');
      
      // 从播放器配置获取字幕轨道
      const playerResponse = this.extractPlayerResponse();
      if (!playerResponse?.captions?.playerCaptionsTracklistRenderer?.captionTracks) {
        throw new Error('无法获取字幕轨道信息');
      }
      
      const tracks = playerResponse.captions.playerCaptionsTracklistRenderer.captionTracks;
      const englishTrack = this.selectPreferredTrack(tracks);
      
      if (!englishTrack?.baseUrl) {
        throw new Error('无法找到英文字幕轨道');
      }
      
      // 构建VTT格式URL
      const vttUrl = englishTrack.baseUrl.replace(/fmt=[^&]*/, 'fmt=vtt');
      this.logger.info('📡 VTT URL:', vttUrl.substring(0, 100) + '...');
      
      const response = await fetch(vttUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
      });
      
      if (!response.ok) {
        throw new Error(`VTT请求失败: ${response.status}`);
      }
      
      const vttText = await response.text();
      this.logger.debug('📄 VTT响应长度:', vttText.length + '字符');
      
      if (!vttText || vttText.trim().length < 100) {
        throw new Error('VTT响应内容太短或为空');
      }
      
      // 解析VTT格式
      const subtitles = this.parseVTTFormat(vttText);
      this.logger.info('✅ VTT解析结果:', subtitles.length + '条字幕');
      
      return subtitles;
      
    } catch (error) {
      this.logger.error('❌ VTT字幕获取失败:', error.message);
      throw error;
    }
  }
  
  // 方法2: 可靠的SRT格式获取  
  async getReliableSRTSubtitles() {
    try {
      this.logger.info('📡 获取SRT格式字幕...');
      
      // 从播放器配置获取字幕轨道
      const playerResponse = this.extractPlayerResponse();
      if (!playerResponse?.captions?.playerCaptionsTracklistRenderer?.captionTracks) {
        throw new Error('无法获取字幕轨道信息');
      }
      
      const tracks = playerResponse.captions.playerCaptionsTracklistRenderer.captionTracks;
      const englishTrack = this.selectPreferredTrack(tracks);
      
      if (!englishTrack?.baseUrl) {
        throw new Error('无法找到英文字幕轨道');
      }
      
      // 构建SRT格式URL
      const srtUrl = englishTrack.baseUrl.replace(/fmt=[^&]*/, 'fmt=srt');
      this.logger.info('📡 SRT URL:', srtUrl.substring(0, 100) + '...');
      
      const response = await fetch(srtUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
      });
      
      if (!response.ok) {
        throw new Error(`SRT请求失败: ${response.status}`);
      }
      
      const srtText = await response.text();
      this.logger.debug('📄 SRT响应长度:', srtText.length + '字符');
      
      if (!srtText || srtText.trim().length < 100) {
        throw new Error('SRT响应内容太短或为空');
      }
      
      // 解析SRT格式
      const subtitles = this.parseSRTFormat(srtText);
      this.logger.info('✅ SRT解析结果:', subtitles.length + '条字幕');
      
      return subtitles;
      
    } catch (error) {
      this.logger.error('❌ SRT字幕获取失败:', error.message);
      throw error;
    }
  }
  
  // 方法3: 直接timedtext API调用
  async getDirectTimedTextSubtitles() {
    try {
      this.logger.info('📡 直接调用timedtext API...');
      
      // 尝试多种不同的timedtext API调用方式
      const apiUrls = [
        `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=vtt&lang=en`,
        `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srt&lang=en`,
        `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=vtt&lang=en&kind=asr`,
        `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srt&lang=en&kind=asr`
      ];
      
      for (const apiUrl of apiUrls) {
        try {
          this.logger.info('🔍 尝试API URL:', apiUrl.substring(0, 80) + '...');
          
          const response = await fetch(apiUrl, {
            headers: {
              'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
              'Accept': 'text/plain, */*',
              'Referer': 'https://www.youtube.com/'
            }
          });
          
          if (response.ok) {
            const text = await response.text();
            this.logger.debug('📄 API响应长度:', text.length + '字符');
            
            if (text && text.trim().length > 100) {
              // 根据URL中的格式解析
              let subtitles;
              if (apiUrl.includes('fmt=vtt')) {
                subtitles = this.parseVTTFormat(text);
              } else if (apiUrl.includes('fmt=srt')) {
                subtitles = this.parseSRTFormat(text);
              }
              
              if (subtitles && subtitles.length > 5) {
                this.logger.info('✅ 直接API调用成功:', subtitles.length + '条字幕');
                return subtitles;
              }
            }
          }
          
        } catch (error) {
          this.logger.debug('⚠️ API URL失败:', error.message);
          continue;
        }
      }
      
      throw new Error('所有API URL都失败了');
      
    } catch (error) {
      this.logger.error('❌ 直接timedtext API调用失败:', error.message);
      throw error;
    }
  }

  // 方法1: 直接尝试XML格式字幕
  async tryDirectXMLSubtitles() {
    try {
      this.logger.info('🔍 尝试XML格式字幕获取...');
      
      // 从播放器配置获取字幕轨道
      const playerResponse = this.extractPlayerResponse();
      if (!playerResponse?.captions?.playerCaptionsTracklistRenderer?.captionTracks) {
        throw new Error('无法获取字幕轨道信息');
      }
      
      const tracks = playerResponse.captions.playerCaptionsTracklistRenderer.captionTracks;
      const englishTrack = this.selectPreferredTrack(tracks);
      
      if (!englishTrack?.baseUrl) {
        throw new Error('无法找到英文字幕轨道');
      }
      
      // 尝试不同的XML格式参数
      const xmlFormats = ['srv3', 'ttml', 'vtt', 'srt'];
      
      for (const format of xmlFormats) {
        try {
          const xmlUrl = englishTrack.baseUrl.replace(/fmt=[^&]*/, `fmt=${format}`);
          this.logger.info(`📡 尝试${format.toUpperCase()}格式:`, xmlUrl.substring(0, 120) + '...');
          
          const response = await fetch(xmlUrl);
          if (response.ok) {
            const xmlText = await response.text();
            this.logger.debug(`📄 ${format}响应长度:`, xmlText.length + '字符');
            
            if (xmlText && xmlText.trim().length > 0) {
              // 尝试解析XML/SRT内容
              const parsed = await this.parseTextSubtitles(xmlText, format);
              if (parsed && parsed.length > 0) {
                this.logger.info(`✅ ${format.toUpperCase()}格式解析成功:`, parsed.length + '条字幕');
                return parsed;
              }
            }
          }
        } catch (error) {
          this.logger.debug(`⚠️ ${format}格式失败:`, error.message);
          continue;
        }
      }
      
      return null;
    } catch (error) {
      this.logger.debug('XML字幕获取失败:', error.message);
      return null;
    }
  }
  
  // 方法2: 尝试SRV3格式字幕
  async trySRV3Subtitles() {
    try {
      this.logger.info('🔍 尝试SRV3格式字幕...');
      
      // 构建SRV3格式的字幕URL
      const srv3Urls = [
        `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srv3&lang=en`,
        `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srv3&lang=en&kind=asr`,
        `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srv3&lang=a.en`
      ];
      
      for (const url of srv3Urls) {
        try {
          this.logger.info('📡 尝试SRV3 URL:', url);
          
          const response = await fetch(url);
          if (response.ok) {
            const srv3Text = await response.text();
            this.logger.debug('📄 SRV3响应长度:', srv3Text.length + '字符');
            
            if (srv3Text && srv3Text.trim().length > 0) {
              const parsed = await this.parseSRV3Format(srv3Text);
              if (parsed && parsed.length > 0) {
                this.logger.info('✅ SRV3格式解析成功:', parsed.length + '条字幕');
                return parsed;
              }
            }
          }
        } catch (error) {
          this.logger.debug('⚠️ SRV3 URL失败:', error.message);
          continue;
        }
      }
      
      return null;
    } catch (error) {
      this.logger.debug('SRV3字幕获取失败:', error.message);
      return null;
    }
  }
  
  // 方法3: DOM直接提取现有字幕
  async tryDOMSubtitleExtraction() {
    try {
      this.logger.info('🔍 尝试DOM字幕提取...');
      
      // 等待字幕加载
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 检查是否有现有字幕显示
      const subtitleSelectors = [
        '.caption-window .captions-text',
        '.ytp-caption-segment',
        '.caption-window',
        '.html5-captions-text',
        '[class*="caption"]'
      ];
      
      let foundSubtitles = [];
      const videoElement = document.querySelector('video');
      
      if (!videoElement) {
        throw new Error('未找到视频元素');
      }
      
      // 监听视频播放过程中的字幕变化
      return new Promise((resolve, reject) => {
        let subtitleData = [];
        let lastSubtitle = '';
        let startTime = 0;
        
        const collectSubtitle = () => {
          const currentTime = videoElement.currentTime;
          let currentSubtitle = '';
          
          for (const selector of subtitleSelectors) {
            const element = document.querySelector(selector);
            if (element && element.textContent && element.textContent.trim()) {
              currentSubtitle = element.textContent.trim();
              break;
            }
          }
          
          if (currentSubtitle && currentSubtitle !== lastSubtitle) {
            if (lastSubtitle) {
              // 保存上一个字幕
              subtitleData.push({
                startTime: startTime,
                endTime: currentTime,
                text: lastSubtitle
              });
            }
            lastSubtitle = currentSubtitle;
            startTime = currentTime;
            this.logger.debug('📝 收集字幕:', currentSubtitle);
          }
        };
        
        // 开始收集字幕
        const interval = setInterval(collectSubtitle, 1000);
        
        // 10秒后停止收集并返回结果
        setTimeout(() => {
          clearInterval(interval);
          
          // 添加最后一个字幕
          if (lastSubtitle) {
            subtitleData.push({
              startTime: startTime,
              endTime: videoElement.currentTime,
              text: lastSubtitle
            });
          }
          
          if (subtitleData.length > 0) {
            this.logger.info('✅ DOM字幕提取成功:', subtitleData.length + '条');
            resolve(subtitleData);
          } else {
            reject(new Error('未能提取到字幕数据'));
          }
        }, 10000);
      });
      
    } catch (error) {
      this.logger.debug('DOM字幕提取失败:', error.message);
      return null;
    }
  }
  
  // 解析文本格式字幕（XML/SRT/VTT等）
  async parseTextSubtitles(text, format) {
    try {
      if (format === 'srt') {
        return this.parseSRTFormat(text);
      } else if (format === 'vtt') {
        return this.parseVTTFormat(text);
      } else if (format === 'ttml') {
        return this.parseXMLSubtitles(text);
      } else if (format === 'srv3') {
        // 优先走通用XML解析，SRV3是XML子集
        try {
          const parsed = await this.parseXMLSubtitles(text);
          return this.parseSubtitleEvents(parsed.events);
        } catch (e) {
          // 回退到旧的SRV3解析器
          return this.parseSRV3Format(text);
        }
      }
      
      // 默认尝试XML解析
      return this.parseXMLSubtitles(text);
    } catch (error) {
      throw new Error(`${format}格式解析失败: ${error.message}`);
    }
  }
  
  // 解析SRT格式 - 优化版，兼容转换格式
  parseSRTFormat(srtText) {
    const subtitles = [];
    
    // 支持从其他格式转换的SRT
    if (!srtText || typeof srtText !== 'string') {
      this.logger.warn('⚠️ SRT数据无效');
      return subtitles;
    }
    
    // 分割成字幕条目
    const entries = srtText.trim().split(/\n\s*\n/).filter(entry => entry.trim());
    
    this.logger.debug('🔍 SRT解析开始:', entries.length + '个条目');
    
    for (const entry of entries) {
      const lines = entry.trim().split('\n');
      if (lines.length < 3) continue;
      
      // 找时间行（可能在第2行或第3行）
      let timeLineIndex = -1;
      for (let i = 0; i < Math.min(3, lines.length); i++) {
        if (lines[i].includes('-->')) {
          timeLineIndex = i;
          break;
        }
      }
      
      if (timeLineIndex === -1) continue;
      
      const timeMatch = lines[timeLineIndex].match(/([\d:,\.]+)\s*-->\s*([\d:,\.]+)/);
      if (timeMatch) {
        const startTime = this.parseSRTTime(timeMatch[1]);
        const endTime = this.parseSRTTime(timeMatch[2]);
        const text = lines.slice(timeLineIndex + 1).join(' ').trim();
        
        if (text && startTime >= 0 && endTime > startTime) {
          subtitles.push({ 
            startTime, 
            endTime, 
            text: text.replace(/<[^>]*>/g, '').trim() // 移除HTML标签
          });
        }
      }
    }
    
    this.logger.info(`✅ SRT解析完成: ${subtitles.length}条字幕`);
    return subtitles;
  }

  // 解析SRT时间格式（兼容多种格式）
  parseSRTTime(timeStr) {
    if (!timeStr) return 0;
    
    // 处理多种时间格式: 00:01:23,456 或 00:01:23.456 或 1:23.456
    const normalizedTime = timeStr.replace(',', '.');
    const parts = normalizedTime.split(':');
    
    if (parts.length === 3) {
      const hours = parseInt(parts[0]) || 0;
      const minutes = parseInt(parts[1]) || 0;
      const secondsParts = parts[2].split('.');
      const seconds = parseInt(secondsParts[0]) || 0;
      const milliseconds = parseInt(secondsParts[1]?.padEnd(3, '0').slice(0, 3)) || 0;
      
      return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000;
    } else if (parts.length === 2) {
      // 处理 mm:ss.sss 格式
      const minutes = parseInt(parts[0]) || 0;
      const secondsParts = parts[1].split('.');
      const seconds = parseInt(secondsParts[0]) || 0;
      const milliseconds = parseInt(secondsParts[1]?.padEnd(3, '0').slice(0, 3)) || 0;
      
      return minutes * 60 + seconds + milliseconds / 1000;
    }
    
    return 0;
  }
  
  // 解析VTT格式
  parseVTTFormat(vttText) {
    const subtitles = [];
    const lines = vttText.split('\n');
    let i = 0;
    
    // 跳过VTT头部
    while (i < lines.length && !lines[i].includes('-->')) {
      i++;
    }
    
    while (i < lines.length) {
      const timeMatch = lines[i].match(/(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})/);
      if (timeMatch) {
        const startTime = this.parseTimeToSeconds(timeMatch[1]);
        const endTime = this.parseTimeToSeconds(timeMatch[2]);
        
        let text = '';
        i++;
        while (i < lines.length && lines[i].trim() && !lines[i].includes('-->')) {
          text += lines[i] + ' ';
          i++;
        }
        
        if (text.trim()) {
          subtitles.push({ startTime, endTime, text: text.trim() });
        }
      } else {
        i++;
      }
    }
    
    return subtitles;
  }
  
  // 解析SRV3格式（YouTube特殊格式）
  parseSRV3Format(srv3Text) {
    try {
      // SRV3格式通常是基于XML的
      const parser = new DOMParser();
      const xmlDoc = parser.parseFromString(srv3Text, 'text/xml');
      
      const subtitles = [];
      const textElements = xmlDoc.querySelectorAll('text');
      
      for (const element of textElements) {
        const start = element.getAttribute('start') || element.getAttribute('t');
        const dur = element.getAttribute('dur') || element.getAttribute('d');
        
        if (start) {
          const startTime = parseFloat(start);
          const duration = dur ? parseFloat(dur) : 3;
          const endTime = startTime + duration;
          const text = element.textContent.trim();
          
          if (text) {
            subtitles.push({ startTime, endTime, text });
          }
        }
      }
      
      return subtitles;
    } catch (error) {
      throw new Error('SRV3格式解析失败: ' + error.message);
    }
  }

  // 检查是否有完整字幕（增强版）
  hasFullSubtitlesEnhanced() {
    return this.fullSubtitles && this.fullSubtitles.length > 0;
  }

  // 清理字幕数据（切换视频时）
  clearSubtitles() {
    this.fullSubtitles = null;
    this.captionTracks = [];
    this.logger.info('🧹 已清理字幕数据');
  }

  // 获取字幕统计信息
  getSubtitleStats() {
    if (!this.fullSubtitles) {
      return { count: 0, duration: 0, avgLength: 0 };
    }

    const count = this.fullSubtitles.length;
    const totalDuration = this.fullSubtitles.reduce((sum, sub) => sum + sub.duration, 0);
    const totalLength = this.fullSubtitles.reduce((sum, sub) => sum + sub.text.length, 0);
    
    return {
      count,
      duration: Math.round(totalDuration),
      avgLength: Math.round(totalLength / count)
    };
  }

  // yt-dlp风格的字幕获取方法（精简版）
  async getSubtitlesWithYTDLP() {
    this.logger.info('📺 尝试yt-dlp风格的字幕获取...');
    
    try {
      // 直接尝试最常用的格式 - 按可靠性排序
      const ytdlpUrls = [
        {
          name: 'YouTube SRT格式（最可靠）',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srt&lang=en`,
          format: 'srt'
        },
        {
          name: 'YouTube Auto-Generated SRT',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srt&lang=en&kind=asr`,
          format: 'srt'
        },
        {
          name: 'YouTube TimedText JSON3',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=json3&lang=en`,
          format: 'json3'
        },
        {
          name: 'YouTube Auto-Generated JSON3',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=json3&lang=en&kind=asr`,
          format: 'json3'
        },
        {
          name: 'YouTube WebVTT',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=vtt&lang=en`,
          format: 'vtt'
        },
        {
          name: 'YouTube SRV3 (XML)',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srv3&lang=en`,
          format: 'srv3'
        },
        {
          name: 'YouTube English (auto a.en)',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=srt&lang=a.en`,
          format: 'srt'
        }
      ];
      
      const ytdlpResults = await Promise.allSettled(ytdlpUrls.map(async (urlInfo) => {
        try {
          this.logger.info(`🔍 尝试${urlInfo.name}`);
          const response = await fetch(urlInfo.url, {
            headers: {
              'User-Agent': 'yt-dlp/2023.12.30',
              'Accept': 'application/json, text/plain, */*'
            }
          });
          if (!response.ok) throw new Error(String(response.status));
          const data = await response.text();
          if (!data || !data.trim()) throw new Error('空响应');
          const parsed = await this.parseYTDLPResponse(data, urlInfo.format);
          const normalized = this.ensureContinuityAndSort(parsed || []);
          const quality = this.validateSubtitleQuality(normalized);
          return { ok: true, urlInfo, subs: normalized, quality };
        } catch (e) {
          return { ok: false, urlInfo, error: e?.message || String(e) };
        }
      }));

      let best = null;
      for (const r of ytdlpResults) {
        if (r.status === 'fulfilled' && r.value.ok && r.value.quality.isValid) {
          if (!best) best = r.value; else {
            const better = (parseFloat((r.value.quality.coverageRatio||'0%')) > parseFloat((best.quality.coverageRatio||'0%')))
                        || (r.value.subs.length > best.subs.length);
            if (better) best = r.value;
          }
        }
      }

      if (best) {
        this.logger.info('✅ yt-dlp风格最佳字幕:', { name: best.urlInfo.name, quality: best.quality });
        return best.subs;
      }
      
      // 备用方案: YouTube内部API
      return await this.tryYouTubeInternalAPI();
      
    } catch (error) {
      this.logger.error('❌ yt-dlp风格获取失败:', error.message);
      return null;
    }
  }

  // 解析yt-dlp响应（精简版）
  async parseYTDLPResponse(data, format) {
    try {
      if (format === 'srt') {
        // 解析SRT格式
        const subtitles = this.parseSRTFormat(data);
        if (subtitles && subtitles.length > 0) {
          this.logger.debug('✅ yt-dlp SRT解析成功:', subtitles.length + '条字幕');
          return subtitles;
        }
      } else if (format === 'json3') {
        let jsonData;
        try {
          jsonData = JSON.parse(data);
          this.logger.debug('✅ yt-dlp JSON解析成功');
        } catch (jsonError) {
          this.logger.warn('⚠️ yt-dlp JSON解析失败，尝试其他方法:', jsonError.message);
          
          // 尝试修复常见的JSON问题
          const cleanedData = data.trim();
          if (cleanedData.length === 0) {
            throw new Error('响应为空');
          }
          
          // 提供诊断信息
          this.logger.debug('📊 yt-dlp响应诊断:', {
            length: cleanedData.length,
            startsWithBrace: cleanedData.startsWith('{'),
            endsWithBrace: cleanedData.endsWith('}'),
            firstChars: cleanedData.substring(0, 50),
            lastChars: cleanedData.substring(Math.max(0, cleanedData.length - 50))
          });
          
          return null;
        }
        
        if (jsonData.events) {
          return this.parseSubtitleEvents(jsonData.events);
        }
      } else if (format === 'vtt') {
        return this.parseVTTFormat(data);
      } else if (format === 'srv3' || format === 'ttml') {
        try {
          const parsed = await this.parseXMLSubtitles(data);
          return this.parseSubtitleEvents(parsed.events);
        } catch {
          return [];
        }
      }
    } catch (error) {
      this.logger.debug('yt-dlp解析响应失败:', error.message);
    }
    
    return null;
  }

  // 生成候选URL（基于baseUrl增强组合）
  buildCandidateUrls(baseUrl) {
    const hasFmt = /[?&]fmt=/.test(baseUrl);
    const ensureFmt = (fmt) => hasFmt ? baseUrl.replace(/fmt=[^&]*/, `fmt=${fmt}`) : (baseUrl + (baseUrl.includes('?') ? '&' : '?') + `fmt=${fmt}`);
    const addParam = (url, key, val) => url + (url.includes('?') ? '&' : '?') + `${key}=${val}`;

    const fmts = [
      { fmt: 'srt', format: 'srt' },
      { fmt: 'vtt', format: 'vtt' },
      { fmt: 'json3', format: 'json3' },
      { fmt: 'srv3', format: 'srv3' },
      { fmt: 'ttml', format: 'ttml' }
    ];

    const urls = [];
    for (const f of fmts) {
      let u = ensureFmt(f.fmt);
      urls.push({ url: u, format: f.format });
      urls.push({ url: addParam(u, 'kind', 'asr'), format: f.format });
      urls.push({ url: addParam(u, 'lang', 'en'), format: f.format });
      urls.push({ url: addParam(u, 'lang', 'en-US'), format: f.format });
      urls.push({ url: addParam(u, 'lang', 'a.en'), format: f.format });
    }
    return urls;
  }

  // 规范化与排序，去重/修复小问题，提升完整性
  ensureContinuityAndSort(subtitles) {
    if (!Array.isArray(subtitles)) return [];
    const sorted = subtitles
      .filter(s => s && typeof s.startTime === 'number' && typeof s.endTime === 'number' && s.endTime > s.startTime && typeof s.text === 'string')
      .map(s => ({ startTime: s.startTime, endTime: s.endTime, text: (s.text || '').trim() }))
      .sort((a, b) => a.startTime - b.startTime || a.endTime - b.endTime);

    const merged = [];
    const EPS = 0.04; // 40ms 容差
    for (const s of sorted) {
      const last = merged[merged.length - 1];
      if (last && Math.abs(s.startTime - last.startTime) < EPS && Math.abs(s.endTime - last.endTime) < EPS && s.text === last.text) {
        // 完全重复，跳过
        continue;
      }
      // 修复极小重叠
      if (last && s.startTime < last.endTime && (last.endTime - s.startTime) < 0.2) {
        last.endTime = Math.max(last.endTime, s.endTime);
        last.text = last.text === s.text ? last.text : (last.text + ' ' + s.text).trim();
      } else {
        merged.push(s);
      }
    }
    return merged;
  }

  // YouTube内部API (yt-dlp风格)
  async tryYouTubeInternalAPI() {
    try {
      this.logger.info('🔧 尝试YouTube内部API...');
      
      // 获取视频信息
      const playerResponse = await this.getPlayerResponse();
      if (!playerResponse) {
        throw new Error('无法获取播放器响应');
      }
      
      // 提取字幕轨道信息
      const captionTracks = playerResponse.captions?.playerCaptionsTracklistRenderer?.captionTracks;
      if (!captionTracks || captionTracks.length === 0) {
        throw new Error('没有找到字幕轨道');
      }
      
      this.logger.info('📋 找到字幕轨道:', captionTracks.length + '个');
      
      // 按优先级选择字幕
      const preferredTrack = this.selectPreferredTrack(captionTracks);
      if (!preferredTrack || !preferredTrack.baseUrl) {
        throw new Error('没有找到合适的字幕轨道');
      }
      
      this.logger.info('✅ 选择字幕轨道:', preferredTrack.name?.simpleText || '未知');
      
      // 获取字幕数据
      const subtitleUrl = preferredTrack.baseUrl + '&fmt=json3';
      const response = await fetch(subtitleUrl);
      
      if (!response.ok) {
        throw new Error(`字幕请求失败: ${response.status}`);
      }
      
      // 使用改进的响应处理
      const responseText = await response.text();
      this.logger.debug('📄 内部API响应长度:', responseText.length + '字符');
      
      if (!responseText || responseText.trim().length === 0) {
        throw new Error('内部API响应为空');
      }
      
      let data;
      try {
        data = JSON.parse(responseText);
        this.logger.debug('✅ 内部API JSON解析成功');
      } catch (jsonError) {
        this.logger.debug('❌ 内部API JSON解析失败:', jsonError.message);
        throw new Error(`内部API JSON解析失败: ${jsonError.message}`);
      }
      
      if (data.events) {
        const parsed = this.parseSubtitleEvents(data.events);
        this.logger.info('✅ 内部API成功:', parsed.length + '条字幕');
        return parsed;
      }
      
      throw new Error('字幕数据格式错误：缺少events字段');
      
    } catch (error) {
      this.logger.debug('内部API失败:', error.message);
      return null;
    }
  }

  // 获取播放器响应 (模拟yt-dlp)
  async getPlayerResponse() {
    try {
      // 从当前页面提取
      const scripts = document.querySelectorAll('script');
      for (const script of scripts) {
        const content = script.textContent || script.innerHTML;
        if (content.includes('ytInitialPlayerResponse')) {
          const match = content.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;/);
          if (match) {
            return JSON.parse(match[1]);
          }
        }
      }
      
      // 如果页面提取失败，尝试API调用
      const response = await fetch(`https://www.youtube.com/youtubei/v1/player`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'yt-dlp/2023.12.30'
        },
        body: JSON.stringify({
          context: {
            client: {
              clientName: 'WEB',
              clientVersion: '2.20231201.00.00'
            }
          },
          videoId: this.videoId
        })
      });
      
      if (response.ok) {
        return await response.json();
      }
      
    } catch (error) {
      this.logger.debug('获取播放器响应失败:', error.message);
    }
    
    return null;
  }

  // 选择首选字幕轨道
  selectPreferredTrack(tracks) {
    // 优先级: 英语 > 自动生成 > 其他
    const priorities = [
      track => track.languageCode === 'en' && !track.kind,
      track => track.languageCode === 'en-US' && !track.kind,
      track => track.languageCode === 'en' && track.kind === 'asr',
      track => track.languageCode === 'en-US' && track.kind === 'asr',
      track => track.languageCode?.startsWith('en'),
      track => track.kind === 'asr',
      track => true  // fallback
    ];
    
    for (const priority of priorities) {
      const track = tracks.find(priority);
      if (track) {
        return track;
      }
    }
    
    return tracks[0];
  }
}