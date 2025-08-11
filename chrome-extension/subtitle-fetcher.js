// YouTube字幕数据获取管理器
class YouTubeSubtitleFetcher {
  constructor() {
    this.videoId = null;
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

      const data = await response.json();
      if (!data.events) {
        throw new Error('字幕数据格式错误');
      }

      // 解析字幕事件
      this.fullSubtitles = this.parseSubtitleEvents(data.events);
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

  // 精简的字幕获取方法 - 只保留3种最有效的方法
  async getSubtitlesWithEnhancedMethods() {
    this.logger.info('🚀 启动字幕获取流程（精简版）...');
    
    const methods = [
      {
        name: '标准播放器配置',
        method: async () => {
          const captionTrack = await this.getCaptionTracks();
          if (captionTrack) {
            return await this.fetchFullSubtitles(captionTrack);
          }
          return null;
        }
      },
      {
        name: '增强页面解析',
        method: async () => {
          // 多重页面解析策略
          const results = await Promise.allSettled([
            this.extractTimedtextFromPageSource(),
            this.extractFromPlayerScripts(),
            this.extractFromYouTubeAPI()
          ]);
          
          for (const result of results) {
            if (result.status === 'fulfilled' && result.value) {
              return result.value;
            }
          }
          return null;
        }
      },
      {
        name: 'yt-dlp风格获取',
        method: async () => {
          return await this.getSubtitlesWithYTDLP();
        }
      }
    ];

    for (const methodInfo of methods) {
      try {
        this.logger.info(`🔄 尝试方法: ${methodInfo.name}`);
        const result = await methodInfo.method();
        
        if (result && result.length > 0) {
          this.logger.info(`✅ ${methodInfo.name} 成功获取 ${result.length} 条字幕`);
          this.fullSubtitles = result;
          return result;
        }
      } catch (error) {
        this.logger.warn(`⚠️ ${methodInfo.name} 失败:`, error.message);
        continue;
      }
    }

    this.logger.error('❌ 所有字幕获取方法都失败了');
    return null;
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
      // 直接尝试最常用的格式
      const ytdlpUrls = [
        {
          name: 'YouTube TimedText JSON3',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=json3&lang=en`,
          format: 'json3'
        },
        {
          name: 'YouTube Auto-Generated',
          url: `https://www.youtube.com/api/timedtext?v=${this.videoId}&fmt=json3&lang=en&kind=asr`,
          format: 'json3'
        }
      ];
      
      for (const urlInfo of ytdlpUrls) {
        try {
          this.logger.info(`🔍 尝试${urlInfo.name}`);
          
          const response = await fetch(urlInfo.url, {
            headers: {
              'User-Agent': 'yt-dlp/2023.12.30',
              'Accept': 'application/json, text/plain, */*'
            }
          });
          
          if (response.ok) {
            const data = await response.text();
            
            if (data && data.trim()) {
              const parsed = await this.parseYTDLPResponse(data, urlInfo.format);
              if (parsed && parsed.length > 0) {
                this.logger.info(`✅ ${urlInfo.name}解析成功:`, parsed.length + '条字幕');
                return parsed;
              }
            }
          }
        } catch (error) {
          this.logger.debug(`⚠️ ${urlInfo.name}失败:`, error.message);
          continue;
        }
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
      if (format === 'json3') {
        const jsonData = JSON.parse(data);
        if (jsonData.events) {
          return this.parseSubtitleEvents(jsonData.events);
        }
      }
    } catch (error) {
      this.logger.debug('解析响应失败:', error.message);
    }
    
    return null;
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
      
      const data = await response.json();
      if (data.events) {
        const parsed = this.parseSubtitleEvents(data.events);
        this.logger.info('✅ 内部API成功:', parsed.length + '条字幕');
        return parsed;
      }
      
      throw new Error('字幕数据格式错误');
      
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