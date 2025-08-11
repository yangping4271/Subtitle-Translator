// 智能分批翻译处理器
class SmartTranslationProcessor {
  constructor(settings, logger) {
    this.settings = settings;
    this.logger = logger;
    this.translationCache = new Map();
    this.isProcessing = false;
    this.processingProgress = 0;
    this.onProgressUpdate = null;

    // 代理工具（从内容脚本注入）
    this.needsProxy = (url) => {
      try { 
        const u = new URL(url); 
        const host = u.hostname;
        // 只有远程API主机需要通过后台代理，本地服务直接访问
        return host === 'ai-proxy.chatwise.app' || host === 'openrouter.ai';
      } catch { return false; }
    };
    this.proxyFetch = async (url, options) => {
      return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({ type: 'PROXY_FETCH', url, options }, (resp) => {
          if (!resp) return reject(new Error('代理请求失败'));
          if (!resp.ok) return reject(new Error(`代理响应失败: ${resp.status}${resp.error ? ' ' + resp.error : ''}`));
          try { const data = JSON.parse(resp.bodyText); resolve({ json: async () => data, ok: resp.ok, status: resp.status }); }
          catch (e) { resolve({ text: async () => resp.bodyText, ok: resp.ok, status: resp.status }); }
        });
      });
    };
  }

  // 智能断句处理 (移植自Python spliter.py的逻辑)
  smartSplit(subtitleTexts) {
    this.logger.info('📝 开始智能断句处理...', {
      inputCount: subtitleTexts.length,
      avgLength: subtitleTexts.reduce((sum, s) => sum + s.text.length, 0) / subtitleTexts.length
    });
    
    // 检测字幕类型：段落字幕 vs 单词级字幕
    const subtitleType = this.detectSubtitleType(subtitleTexts);
    this.logger.info(`🔍 字幕类型检测: ${subtitleType.type}`, subtitleType.analysis);
    
    if (subtitleType.type === 'paragraph') {
      this.logger.info('📖 检测到段落字幕，跳过断句处理，直接用于翻译');
      return subtitleTexts.map((sub, index) => ({
        index: index,
        text: sub.text,
        startTime: sub.startTime,
        endTime: sub.endTime,
        originalIndices: [sub.index]
      }));
    }
    
    this.logger.info('🔤 检测到单词级字幕，执行智能断句...');
    
    // 合并相邻的短字幕，避免过度分割
    const mergedTexts = this.mergeShortSubtitles(subtitleTexts);
    this.logger.info('🔗 短字幕合并完成:', {
      before: subtitleTexts.length,
      after: mergedTexts.length
    });
    
    // 按语义边界分割长句
    const segments = this.splitBySemanticBoundaries(mergedTexts);
    
    this.logger.info(`✅ 断句处理完成: ${subtitleTexts.length} → ${segments.length} 段`, {
      reductionRate: ((subtitleTexts.length - segments.length) / subtitleTexts.length * 100).toFixed(1) + '%'
    });
    
    return segments;
  }
  
  // 检测字幕类型（段落 vs 单词级）
  detectSubtitleType(subtitleTexts) {
    if (!subtitleTexts || subtitleTexts.length === 0) {
      return { type: 'unknown', analysis: { reason: '无字幕数据' } };
    }
    
    const sample = subtitleTexts.slice(0, Math.min(20, subtitleTexts.length));
    let indicators = {
      avgWordsPerSubtitle: 0,
      avgDuration: 0,
      shortSubtitles: 0,  // ≤3个词
      longSubtitles: 0,   // ≥8个词
      hasCompleteSentences: 0,
      hasPunctuation: 0
    };
    
    for (const sub of sample) {
      const words = sub.text.trim().split(/\s+/).filter(w => w.length > 0);
      const wordCount = words.length;
      const duration = sub.endTime - sub.startTime;
      
      indicators.avgWordsPerSubtitle += wordCount;
      indicators.avgDuration += duration;
      
      if (wordCount <= 3) indicators.shortSubtitles++;
      if (wordCount >= 8) indicators.longSubtitles++;
      
      // 检查是否有完整句子（以句号、问号、感叹号结尾）
      if (/[.!?]$/.test(sub.text.trim())) {
        indicators.hasCompleteSentences++;
      }
      
      // 检查是否有标点符号
      if (/[,.!?;:]/.test(sub.text)) {
        indicators.hasPunctuation++;
      }
    }
    
    indicators.avgWordsPerSubtitle /= sample.length;
    indicators.avgDuration /= sample.length;
    
    // 判断规则
    const isWordLevel = (
      indicators.avgWordsPerSubtitle <= 4 &&      // 平均每条字幕≤4个词
      indicators.shortSubtitles >= sample.length * 0.7 && // 70%以上是短字幕
      indicators.hasCompleteSentences <= sample.length * 0.3 // 30%以下有完整句子
    );
    
    const isParagraph = (
      indicators.avgWordsPerSubtitle >= 6 &&      // 平均每条字幕≥6个词  
      indicators.longSubtitles >= sample.length * 0.4 && // 40%以上是长字幕
      indicators.hasCompleteSentences >= sample.length * 0.5 // 50%以上有完整句子
    );
    
    let type, confidence;
    if (isParagraph && !isWordLevel) {
      type = 'paragraph';
      confidence = 'high';
    } else if (isWordLevel && !isParagraph) {
      type = 'word-level';  
      confidence = 'high';
    } else {
      // 边界情况，使用更保守的判断
      type = indicators.avgWordsPerSubtitle >= 5 ? 'paragraph' : 'word-level';
      confidence = 'medium';
    }
    
    return {
      type,
      confidence,
      analysis: {
        sampleSize: sample.length,
        avgWordsPerSubtitle: indicators.avgWordsPerSubtitle.toFixed(1),
        avgDuration: indicators.avgDuration.toFixed(1) + 's',
        shortSubtitlesRatio: (indicators.shortSubtitles / sample.length * 100).toFixed(1) + '%',
        longSubtitlesRatio: (indicators.longSubtitles / sample.length * 100).toFixed(1) + '%',
        completeSentencesRatio: (indicators.hasCompleteSentences / sample.length * 100).toFixed(1) + '%',
        punctuationRatio: (indicators.hasPunctuation / sample.length * 100).toFixed(1) + '%'
      }
    };
  }

  // 合并短字幕
  mergeShortSubtitles(subtitleTexts) {
    const merged = [];
    let currentSegment = null;
    
    for (const subtitle of subtitleTexts) {
      const words = subtitle.text.split(' ').length;
      
      // 如果当前字幕太短（少于5个词），尝试与相邻字幕合并
      if (words < 5 && currentSegment && 
          (currentSegment.text.split(' ').length + words) < 20) {
        // 合并到当前段落
        currentSegment.text += ' ' + subtitle.text;
        currentSegment.endTime = subtitle.endTime;
        currentSegment.indices.push(subtitle.index);
      } else {
        // 保存当前段落，开始新段落
        if (currentSegment) {
          merged.push(currentSegment);
        }
        currentSegment = {
          text: subtitle.text,
          startTime: subtitle.startTime,
          endTime: subtitle.endTime,
          indices: [subtitle.index]
        };
      }
    }
    
    if (currentSegment) {
      merged.push(currentSegment);
    }
    
    return merged;
  }

  // 按语义边界分割
  splitBySemanticBoundaries(segments) {
    const result = [];
    
    for (const segment of segments) {
      const words = segment.text.split(' ');
      
      // 如果段落过长（超过25个词），按标点符号分割
      if (words.length > 25) {
        const splitSegments = this.splitLongSegment(segment);
        result.push(...splitSegments);
      } else {
        result.push(segment);
      }
    }
    
    return result;
  }

  // 分割长段落
  splitLongSegment(segment) {
    const text = segment.text;
    const sentences = this.splitBySentenceBoundaries(text);
    const result = [];
    
    let currentText = '';
    let wordCount = 0;
    
    for (const sentence of sentences) {
      const sentenceWords = sentence.split(' ').length;
      
      if (wordCount + sentenceWords > 20 && currentText) {
        // 保存当前累积的文本
        result.push({
          text: currentText.trim(),
          startTime: segment.startTime,
          endTime: segment.endTime,
          indices: segment.indices
        });
        currentText = sentence;
        wordCount = sentenceWords;
      } else {
        currentText += (currentText ? ' ' : '') + sentence;
        wordCount += sentenceWords;
      }
    }
    
    if (currentText) {
      result.push({
        text: currentText.trim(),
        startTime: segment.startTime,
        endTime: segment.endTime,
        indices: segment.indices
      });
    }
    
    return result.length > 0 ? result : [segment];
  }

  // 按句子边界分割
  splitBySentenceBoundaries(text) {
    // 改进的句子分割正则表达式
    const sentenceRegex = /([.!?]+\s+)(?=[A-Z])|([.!?]+$)/g;
    const sentences = text.split(sentenceRegex).filter(s => s && s.trim());
    
    if (sentences.length <= 1) {
      // 如果没有明显的句子边界，按逗号分割
      return text.split(/,\s+/).filter(s => s.trim());
    }
    
    return sentences.map(s => s.trim()).filter(s => s);
  }

  // 内容总结分析 (移植自Python summarizer.py的逻辑)
  async generateContentSummary(segments) {
    this.logger.info('🔍 开始内容总结分析...', {
      totalSegments: segments.length,
      totalWords: segments.reduce((sum, s) => sum + s.text.split(' ').length, 0)
    });
    
    try {
      // 提取代表性文本片段（前10%、中间10%、后10%）
      const totalSegments = segments.length;
      const sampleSize = Math.max(3, Math.floor(totalSegments * 0.1));
      
      const startSegments = segments.slice(0, sampleSize);
      const middleStart = Math.floor((totalSegments - sampleSize) / 2);
      const middleSegments = segments.slice(middleStart, middleStart + sampleSize);
      const endSegments = segments.slice(-sampleSize);
      
      const sampleText = [
        ...startSegments.map(s => s.text),
        ...middleSegments.map(s => s.text),
        ...endSegments.map(s => s.text)
      ].join(' ');

      this.logger.info('📊 样本文本提取完成:', {
        sampleSize: sampleSize * 3,
        sampleLength: sampleText.length,
        coverage: ((sampleSize * 3 / totalSegments) * 100).toFixed(1) + '%'
      });

      // 调用API进行内容分析
      const summary = await this.callSummaryAPI(sampleText);
      if (summary && summary.trim().length > 0) {
        this.logger.info('✅ 内容总结分析完成', { summaryLength: summary.length });
        return summary;
      }
      this.logger.info('ℹ️ 内容总结为空，跳过使用总结');
      return '';
      
    } catch (error) {
      this.logger.error('❌ 内容总结失败:', error.message);
      // 不再返回默认总结，按照“无总结”处理
      return '';
    }
  }

  // 调用总结API
  async callSummaryAPI(sampleText) {
    const prompt = `请分析这段YouTube视频字幕内容，提供以下信息：

1. 视频主题和类型
2. 主要技术术语和专有名词
3. 语言风格和难度级别
4. 翻译建议和注意事项

字幕内容：${sampleText.substring(0, 2000)}`;

    const url = this.settings.apiUrl + '/chat/completions';
    const fetchOptions = {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.settings.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.settings.model || 'gpt-4o-mini',
        messages: [
          { role: 'user', content: prompt }
        ],
        temperature: 0.3,
        max_tokens: 800
      })
    };
    const response = this.needsProxy(url) ? await this.proxyFetch(url, fetchOptions) : await fetch(url, fetchOptions);

    if (!response.ok) {
      throw new Error(`API调用失败: ${response.status}`);
    }

    const data = await response.json();
    const content = data?.choices?.[0]?.message?.content ?? '';
    return typeof content === 'string' ? content.trim() : '';
  }

  // 获取默认总结
  getDefaultSummary() {
    return `视频类型：教育/技术内容
翻译建议：保持术语准确性，使用专业但易懂的中文表达`;
  }

  // 分批翻译处理 (移植自Python optimizer.py的逻辑)
  async processBatchTranslation(segments, summary) {
    this.isProcessing = true;
    this.processingProgress = 0;
    
    try {
      this.logger.info('🚀 开始分批翻译处理...', {
        totalSegments: segments.length,
        batchSize: 50,
        estimatedBatches: Math.ceil(segments.length / 50)
      });

      const batchSize = 50; // 每批处理50个段落
      const batches = this.createBatches(segments, batchSize);
      const results = new Map();

      this.logger.info('📦 批次划分完成:', {
        totalBatches: batches.length,
        avgBatchSize: (segments.length / batches.length).toFixed(1)
      });

      // 并行处理多个批次
      const maxConcurrent = 3; // 最多同时处理3个批次
      const promises = [];

      for (let i = 0; i < batches.length; i += maxConcurrent) {
        const batchGroup = batches.slice(i, i + maxConcurrent);
        
        this.logger.info(`🔄 开始处理批次组 ${Math.floor(i / maxConcurrent) + 1}/${Math.ceil(batches.length / maxConcurrent)}`, {
          batchNumbers: batchGroup.map((_, idx) => i + idx + 1)
        });
        
        const batchPromises = batchGroup.map(async (batch, batchIndex) => {
          const actualBatchIndex = i + batchIndex;
          return this.translateBatch(batch, actualBatchIndex, summary, results);
        });

        // 等待当前批次组完成
        await Promise.all(batchPromises);
        
        // 更新进度
        this.processingProgress = Math.min(95, ((i + maxConcurrent) / batches.length) * 100);
        this.updateProgress(this.processingProgress);
        
        this.logger.info(`✅ 批次组 ${Math.floor(i / maxConcurrent) + 1} 完成`, {
          progress: this.processingProgress.toFixed(1) + '%'
        });
      }

      // 整理结果
      const translatedSegments = this.assembleResults(segments, results);
      
      this.processingProgress = 100;
      this.updateProgress(100);
      
      this.logger.info('✅ 分批翻译处理完成', {
        totalTranslated: translatedSegments.length,
        successRate: (results.size / segments.length * 100).toFixed(1) + '%'
      });
      return translatedSegments;
      
    } catch (error) {
      this.logger.error('❌ 分批翻译处理失败:', error.message);
      throw error;
    } finally {
      this.isProcessing = false;
    }
  }

  // 创建批次
  createBatches(segments, batchSize) {
    const batches = [];
    for (let i = 0; i < segments.length; i += batchSize) {
      batches.push(segments.slice(i, i + batchSize));
    }
    return batches;
  }

  // 翻译单个批次
  async translateBatch(batch, batchIndex, summary, results) {
    try {
      this.logger.info(`📦 翻译批次 ${batchIndex + 1}:`, batch.length + '个段落');
      
      // 构建批量翻译请求
      const batchText = batch.map((segment, index) => 
        `[${index + 1}] ${segment.text}`
      ).join('\n\n');

      const prompt = this.createBatchTranslationPrompt(summary, batchText);
      
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
            { role: 'user', content: `请翻译以下字幕段落：\n\n${batchText}` }
          ],
          temperature: 0.7,
          max_tokens: 2000
        })
      };
      const response = this.needsProxy(url) ? await this.proxyFetch(url, fetchOptions) : await fetch(url, fetchOptions);

      if (!response.ok) {
        throw new Error(`批次${batchIndex + 1}翻译失败: ${response.status}`);
      }

      const data = await response.json();
      const translatedText = data.choices[0].message.content.trim();

      // 解析批量翻译结果
      const translations = this.parseBatchTranslations(translatedText);
      
      // 存储结果
      batch.forEach((segment, index) => {
        const segmentId = segment.indices ? segment.indices[0] : batchIndex * 50 + index;
        results.set(segmentId, {
          original: segment.text,
          translation: translations[index] || '[翻译失败]',
          startTime: segment.startTime,
          endTime: segment.endTime
        });
      });

      this.logger.info(`✅ 批次 ${batchIndex + 1} 翻译完成`);
      
    } catch (error) {
      this.logger.error(`❌ 批次 ${batchIndex + 1} 翻译失败:`, error.message);
      
      // 为失败的批次创建默认结果
      batch.forEach((segment, index) => {
        const segmentId = segment.indices ? segment.indices[0] : batchIndex * 50 + index;
        results.set(segmentId, {
          original: segment.text,
          translation: '[翻译失败]',
          startTime: segment.startTime,
          endTime: segment.endTime
        });
      });
    }
  }

  // 翻译单个批次（返回列表，便于增量翻译）
  async translateBatchReturnList(batch, batchIndex, summary) {
    try {
      this.logger.info(`📦(lazy) 翻译批次 ${batchIndex + 1}:`, batch.length + '个段落');

      // 验证设置
      if (!this.settings.apiKey || !this.settings.apiUrl) {
        throw new Error(`API配置不完整: apiKey=${!!this.settings.apiKey}, apiUrl=${!!this.settings.apiUrl}`);
      }

      const batchText = batch.map((segment, index) => `[${index + 1}] ${segment.text}`).join('\n\n');
      const prompt = this.createBatchTranslationPrompt(summary, batchText);

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
            { role: 'user', content: `请翻译以下字幕段落：\n\n${batchText}` }
          ],
          temperature: 0.7,
          max_tokens: 2000
        })
      };
      
      this.logger.info(`🌐 API调用详情:`, {
        url,
        model: this.settings.model,
        useProxy: this.needsProxy(url),
        apiKeyLength: this.settings.apiKey?.length,
        batchSize: batch.length
      });
      
      const response = this.needsProxy(url) ? await this.proxyFetch(url, fetchOptions) : await fetch(url, fetchOptions);
      
      this.logger.info(`📡 API响应:`, {
        status: response.status,
        ok: response.ok
      });
      
      if (!response.ok) {
        const errorText = await response.text().catch(() => '无法读取错误信息');
        throw new Error(`批次${batchIndex + 1}翻译失败: ${response.status} - ${errorText}`);
      }
      const data = await response.json();
      const translatedText = data.choices[0].message.content.trim();
      const translations = this.parseBatchTranslations(translatedText);
      this.logger.info(`✅(lazy) 批次 ${batchIndex + 1} 翻译完成`);
      return translations;
    } catch (error) {
      this.logger.error(`❌(lazy) 批次 ${batchIndex + 1} 翻译失败:`, error.message);
      // 返回等长的失败占位
      return batch.map(() => '[翻译失败]');
    }
  }

  // 创建批量翻译提示词
  createBatchTranslationPrompt(summary, batchText) {
    return `你是一个专业的YouTube字幕翻译专家。基于以下视频内容分析，请将英文字幕批量翻译成${this.settings.targetLang}。

视频内容分析：
${summary}

翻译要求：
1. 保持原意准确，使用自然流畅的${this.settings.targetLang}表达
2. 保持技术术语和专有名词的准确性
3. 适合字幕显示的简洁表达
4. 保持编号格式，每个段落独占一行
5. 直接返回翻译结果，不要添加解释

示例格式：
[1] 翻译结果1
[2] 翻译结果2
[3] 翻译结果3`;
  }

  // 解析批量翻译结果
  parseBatchTranslations(translatedText) {
    const lines = translatedText.split('\n').filter(line => line.trim());
    const translations = [];
    
    for (const line of lines) {
      const match = line.match(/^\[\d+\]\s*(.+)$/);
      if (match) {
        translations.push(match[1].trim());
      } else if (line.trim() && !line.match(/^\[\d+\]/)) {
        // 没有编号的行也算作翻译结果
        translations.push(line.trim());
      }
    }
    
    return translations;
  }

  // 整理翻译结果
  assembleResults(originalSegments, results) {
    const translatedSegments = [];
    
    for (let i = 0; i < originalSegments.length; i++) {
      const original = originalSegments[i];
      const result = results.get(i);
      
      if (result) {
        translatedSegments.push({
          index: i,
          text: original.text,
          translation: result.translation,
          startTime: original.startTime,
          endTime: original.endTime
        });
      } else {
        // 如果某个段落没有翻译结果，创建默认结果
        translatedSegments.push({
          index: i,
          text: original.text,
          translation: '[翻译缺失]',
          startTime: original.startTime,
          endTime: original.endTime
        });
      }
    }
    
    return translatedSegments;
  }

  // 更新进度
  updateProgress(progress) {
    if (this.onProgressUpdate) {
      this.onProgressUpdate(progress);
    }
  }

  // 设置进度回调
  setProgressCallback(callback) {
    this.onProgressUpdate = callback;
  }

  // 获取处理状态
  getProcessingStatus() {
    return {
      isProcessing: this.isProcessing,
      progress: this.processingProgress
    };
  }

  // 停止处理
  stop() {
    this.isProcessing = false;
    this.logger.info('⏹️ 翻译处理已停止');
  }
}