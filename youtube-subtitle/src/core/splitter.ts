/**
 * 断句模块 - 移植自 Python split_by_llm.py
 * 包含 5 层防护机制和智能分割策略
 * 支持批量并行处理（与 Python 版本保持一致）
 */

import { setupLogger } from '../utils/logger.js';
import { buildSplitPrompt } from './prompts.js';
import { SubtitleData } from './subtitle-data.js';
import { findBestMatch, preprocessText } from '../utils/similarity.js';
import type { TranslatorConfig, SplitStats, SubtitleEntry } from '../types/index.js';

const logger = setupLogger('splitter');

// 时间间隔阈值（毫秒）- 与 Python 版本一致
const MAX_GAP = 1500; // 1.5秒

/**
 * 按时间间隔分组片段（与 Python merge_by_time_gaps 一致）
 * 避免合并时间跨度过大的片段
 */
function groupSegmentsByTimeGaps(segments: SubtitleEntry[], maxGap: number = MAX_GAP): SubtitleEntry[][] {
  if (segments.length === 0) return [];

  const groups: SubtitleEntry[][] = [];
  let currentGroup: SubtitleEntry[] = [segments[0]];

  for (let i = 1; i < segments.length; i++) {
    const gap = segments[i].startTime - segments[i - 1].endTime;
    if (gap > maxGap) {
      // 间隔过大，开始新组
      groups.push(currentGroup);
      currentGroup = [segments[i]];
    } else {
      currentGroup.push(segments[i]);
    }
  }

  // 添加最后一组
  if (currentGroup.length > 0) {
    groups.push(currentGroup);
  }

  return groups;
}

/**
 * 基于句子相似度匹配来合并字幕片段
 * 与 Python 版本 merge_segments_based_on_sentences 完全一致
 *
 * @param segments 原始字幕片段列表
 * @param sentences LLM 返回的断句结果
 * @returns 合并后的字幕片段（带有正确的时间戳）
 */
function mergeSegmentsBasedOnSentences(
  segments: SubtitleEntry[],
  sentences: string[]
): SubtitleEntry[] {
  const newSegments: SubtitleEntry[] = [];
  let currentIndex = 0;
  let unmatchedCount = 0;
  const maxUnmatched = 5;

  logger.info(`🔗 开始时间戳对齐: ${sentences.length} 个句子 -> ${segments.length} 个原始片段`);

  for (let sentenceIdx = 0; sentenceIdx < sentences.length; sentenceIdx++) {
    const sentence = sentences[sentenceIdx];

    // 使用相似度匹配查找最佳对应位置
    const match = findBestMatch(
      sentence,
      segments,
      currentIndex,
      30, // maxShift
      0.5  // threshold
    );

    if (match) {
      const { position, windowSize, similarity } = match;

      // 获取匹配的片段
      const matchedSegments = segments.slice(position, position + windowSize);

      // 按时间间隔分组（与 Python 版本一致）
      const groups = groupSegmentsByTimeGaps(matchedSegments, MAX_GAP);

      // 为每组创建合并的字幕
      for (const group of groups) {
        const mergedStartTime = group[0].startTime;
        const mergedEndTime = group[group.length - 1].endTime;

        newSegments.push({
          index: newSegments.length + 1,
          startTime: mergedStartTime,
          endTime: mergedEndTime,
          text: sentence, // 使用 LLM 返回的原始句子
        });
      }

      // 更新当前索引
      currentIndex = position + windowSize;

      // 重置未匹配计数
      unmatchedCount = 0;

      // 输出匹配信息（仅在调试模式）
      if (similarity < 0.8) {
        logger.debug(
          `⚠️ 句子 ${sentenceIdx + 1} 相似度较低: ${(similarity * 100).toFixed(1)}%`
        );
      }
    } else {
      // 没有找到匹配
      unmatchedCount++;
      logger.warn(`❌ 句子 ${sentenceIdx + 1} 未找到匹配: "${sentence.substring(0, 50)}..."`);

      if (unmatchedCount > maxUnmatched) {
        throw new Error(
          `时间戳对齐失败：连续 ${unmatchedCount} 个句子未匹配（超过阈值 ${maxUnmatched}）`
        );
      }

      // 使用估算时间（降级处理）
      // 使用固定 5 秒默认持续时间（与下载 SRT 字幕算法一致）
      const estimatedDuration = 5000; // 固定5秒（5000毫秒，与 Python 版本保持一致）
      const lastEndTime = newSegments.length > 0
        ? newSegments[newSegments.length - 1].endTime
        : segments[0]?.startTime || 0;

      newSegments.push({
        index: newSegments.length + 1,
        startTime: lastEndTime,
        endTime: lastEndTime + estimatedDuration,
        text: sentence,
      });
    }
  }

  logger.info(
    `✅ 时间戳对齐完成: ${newSegments.length} 个字幕片段 (未匹配: ${unmatchedCount})`
  );

  return newSegments;
}

/**
 * 统计文本中的单词数
 */
export function countWords(text: string): number {
  // 移除非英文字符后统计单词数
  const englishText = text.replace(/[\u4e00-\u9fff]/g, ' ');
  const words = englishText.trim().split(/\s+/).filter(w => w.length > 0);
  return words.length;
}

/**
 * 按明确的句子结束标记拆分句子
 */
export function splitByEndMarks(sentence: string): string[] {
  const endMarks = ['. ', '! ', '? '];
  const positions: number[] = [];

  // 查找句子结束标记的位置
  for (const mark of endMarks) {
    let start = 0;
    while (true) {
      const pos = sentence.indexOf(mark, start);
      if (pos === -1) break;

      // 确保不是小数点
      if (mark === '. ' && pos > 0 && /\d/.test(sentence[pos - 1])) {
        start = pos + 1;
        continue;
      }
      positions.push(pos + 1); // 标点后的位置
      start = pos + 1;
    }
  }

  // 如果没有找到结束标记，返回原句子
  if (positions.length === 0) {
    return [sentence];
  }

  // 执行分割
  positions.sort((a, b) => a - b);
  const segments: string[] = [];
  let start = 0;

  for (const pos of positions) {
    const segment = sentence.slice(start, pos).trim();
    // 确保每段至少有3个单词才分割
    if (segment && countWords(segment) >= 3) {
      segments.push(segment);
      start = pos;
    }
  }

  // 处理最后一段
  const lastSegment = sentence.slice(start).trim();
  if (lastSegment) {
    if (segments.length > 0 && countWords(lastSegment) < 2) {
      // 最后一段太短，合并到前一段
      segments[segments.length - 1] += ' ' + lastSegment;
    } else {
      segments.push(lastSegment);
    }
  }

  // 记录分割结果
  if (segments.length > 1) {
    logger.info(`✂️ 标点分割: ${segments.length}段`);
  }

  return segments.length > 1 ? segments : [sentence];
}

/**
 * 智能分割：基于语义边界的拆分
 *
 * 策略：
 * 1. 优先基于标点符号（句号、分号、逗号等）
 * 2. 其次基于连接词（并列连词、从属连词、关系代词）
 * 3. 如果找不到合适的语义边界，返回原句
 */
export function aggressiveSplit(text: string, maxWords: number): string[] {
  const words = text.split(/\s+/);
  const wordCount = words.length;

  // 如果已经满足要求，直接返回
  if (wordCount <= maxWords) {
    return [text];
  }

  logger.info(`🔧 尝试智能分割: ${wordCount}字 -> 目标≤${maxWords}字`);

  // 策略1: 规则匹配分割（6层优先级）
  const splitCandidates: Array<[number, number, string]> = [];

  // 优先级1: 句子结束标记
  for (let i = 2; i < wordCount - 2; i++) {
    const word = words[i].replace(/[,;:]$/, '');
    if (word.endsWith('.') || word.endsWith('!') || word.endsWith('?')) {
      splitCandidates.push([i + 1, 10, `句号'${word.slice(-1)}'`]);
    }
  }

  // 优先级2: 分号/冒号
  for (let i = 2; i < wordCount - 2; i++) {
    const word = words[i];
    if (word.endsWith(';') || word.endsWith(':')) {
      splitCandidates.push([i + 1, 9, `分隔'${word.slice(-1)}'`]);
    }
  }

  // 优先级3: 逗号
  for (let i = 2; i < wordCount - 2; i++) {
    const word = words[i];
    if (word.endsWith(',')) {
      splitCandidates.push([i + 1, 8, '逗号']);
    }
  }

  // 优先级4: 并列连词
  const coordinatingConj = ['and', 'but', 'or', 'so', 'yet', 'nor'];
  for (let i = 3; i < wordCount - 2; i++) {
    const word = words[i].toLowerCase().replace(/[,.!?]/g, '');
    if (coordinatingConj.includes(word)) {
      splitCandidates.push([i, 7, `并列连词'${word}'`]);
    }
  }

  // 优先级5: 从属连词
  const subordinatingConj = ['because', 'although', 'though', 'unless', 'since',
    'while', 'whereas', 'if', 'when', 'before', 'after'];
  for (let i = 3; i < wordCount - 2; i++) {
    const word = words[i].toLowerCase().replace(/[,.!?]/g, '');
    if (subordinatingConj.includes(word)) {
      splitCandidates.push([i, 6, `从属连词'${word}'`]);
    }
  }

  // 优先级6: 关系代词
  const relativePronouns = ['that', 'which', 'who', 'whom', 'whose', 'where', 'when', 'whether'];
  for (let i = 3; i < wordCount - 2; i++) {
    const word = words[i].toLowerCase().replace(/[,.!?]/g, '');
    if (relativePronouns.includes(word)) {
      splitCandidates.push([i, 5, `关系词'${word}'`]);
    }
  }

  // 如果找到候选点，选择最优的
  if (splitCandidates.length > 0) {
    // 按优先级排序，同优先级选择最接近中点的
    const midPoint = Math.floor(wordCount / 2);
    splitCandidates.sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1]; // 优先级高的在前
      return Math.abs(a[0] - midPoint) - Math.abs(b[0] - midPoint); // 更接近中点的在前
    });

    const [bestPos, priority, reason] = splitCandidates[0];

    // 执行分割
    const firstPart = words.slice(0, bestPos).join(' ').trim();
    const secondPart = words.slice(bestPos).join(' ').trim();

    logger.info(`✅ [策略1] 规则匹配分割在${reason}处 (优先级${priority}):`);
    logger.info(`   片段1(${countWords(firstPart)}字): ${firstPart.slice(0, 50)}...`);
    logger.info(`   片段2(${countWords(secondPart)}字): ${secondPart.slice(0, 50)}...`);

    // 递归处理仍然超长的片段
    const result: string[] = [];
    const warningThreshold = Math.floor(maxWords * 1.5);

    for (const part of [firstPart, secondPart]) {
      if (countWords(part) > warningThreshold) {
        result.push(...aggressiveSplit(part, maxWords));
      } else {
        result.push(part);
      }
    }

    return result;
  }

  // 找不到语义边界，返回原句
  logger.warn('⚠️ 未找到语义边界，返回原句');
  return [text];
}

/**
 * 降级分割（兜底方案）：在理想切分点附近寻找语义边界
 */
export function fallbackSplit(text: string, maxWords: number, warningThreshold?: number): string[] {
  if (warningThreshold === undefined) {
    warningThreshold = Math.floor(maxWords * 1.5);
  }

  const words = text.split(/\s+/);
  const wordCount = words.length;

  // 计算需要分成几段
  const numSegments = Math.ceil(wordCount / maxWords);

  if (numSegments === 1) {
    return [text];
  }

  logger.info(`🔨 降级分割: ${wordCount}字 -> ${numSegments}段 (每段≤${maxWords}字)`);

  // 计算理想分割点
  const segmentSize = wordCount / numSegments;
  const idealPoints: number[] = [];
  for (let i = 1; i < numSegments; i++) {
    idealPoints.push(Math.floor(segmentSize * i));
  }

  // 在每个理想点附近寻找最佳分割位置
  const actualSplits: number[] = [];
  const searchRange = 5;

  for (const idealPos of idealPoints) {
    let bestPos = idealPos;
    let bestScore = 0;

    const start = Math.max(1, idealPos - searchRange);
    const end = Math.min(wordCount - 1, idealPos + searchRange);

    for (let i = start; i <= end; i++) {
      let score = 0;
      const word = words[i - 1].replace(/[,;:]$/, '');

      // 评分：标点优于连接词优于普通位置
      if (word.endsWith('.') || word.endsWith('!') || word.endsWith('?')) {
        score = 10;
      } else if (word.endsWith(',') || word.endsWith(';') || word.endsWith(':')) {
        score = 8;
      } else if (i < wordCount && ['and', 'but', 'or', 'so', 'because', 'when', 'while']
        .includes(words[i].toLowerCase())) {
        score = 6;
      } else {
        score = 1;
      }

      // 同等分数下，优先选择更接近理想点的
      if (score > bestScore || (score === bestScore && Math.abs(i - idealPos) < Math.abs(bestPos - idealPos))) {
        bestScore = score;
        bestPos = i;
      }
    }

    actualSplits.push(bestPos);
  }

  // 执行分割
  const result: string[] = [];
  let startIdx = 0;

  for (const splitPos of actualSplits) {
    const segment = words.slice(startIdx, splitPos).join(' ').trim();
    if (segment) {
      result.push(segment);
    }
    startIdx = splitPos;
  }

  // 添加最后一段
  const lastSegment = words.slice(startIdx).join(' ').trim();
  if (lastSegment) {
    result.push(lastSegment);
  }

  // 输出分割结果
  logger.info(`✅ 降级分割完成: ${result.length}段`);
  for (let i = 0; i < result.length; i++) {
    const segWords = countWords(result[i]);
    logger.info(`   片段${i + 1}(${segWords}字): ${result[i].slice(0, 50)}...`);
    if (segWords > maxWords) {
      logger.warn(`   ⚠️ 片段${i + 1}仍超标，需再次分割`);
    }
  }

  // 验证：如果仍有超标片段，递归处理
  const finalResult: string[] = [];
  for (const segment of result) {
    if (countWords(segment) > warningThreshold) {
      // 简单二分
      const segWords = segment.split(/\s+/);
      const mid = Math.floor(segWords.length / 2);
      finalResult.push(segWords.slice(0, mid).join(' '));
      finalResult.push(segWords.slice(mid).join(' '));
    } else {
      finalResult.push(segment);
    }
  }

  return finalResult;
}

/**
 * OpenAI API 客户端接口
 */
interface OpenAIClient {
  callChat(systemPrompt: string, userPrompt: string, options?: {
    temperature?: number;
    timeout?: number;
  }): Promise<string>;
}

/**
 * 使用 LLM 进行断句
 * @param text 要拆分的文本
 * @param client OpenAI 客户端
 * @param config 配置
 * @param batchIndex 批次索引（用于日志）
 */
export async function splitByLLM(
  text: string,
  client: OpenAIClient,
  config: TranslatorConfig,
  batchIndex?: number
): Promise<string[]> {
  const { maxWordCountEnglish, toleranceMultiplier, warningMultiplier, maxMultiplier } = config;

  logger.info(`📝 处理文本: 共${countWords(text)}个单词`);

  // 构建 Prompt
  const systemPrompt = buildSplitPrompt({ maxWordCountEnglish });
  const userPrompt = `Please use multiple <br> tags to separate the following sentence. Make sure to preserve all spaces and punctuation exactly as they appear in the original text:\n${text}`;

  // 调用 API
  const response = await client.callChat(systemPrompt, userPrompt, {
    temperature: 0.2,
    timeout: 80000,
  });

  if (!response) {
    throw new Error('API 返回为空');
  }

  logger.info(`API 返回结果: \n\n${response}\n`);

  // 清理响应
  let result = response;
  // 移除 <think></think> 标签
  result = result.replace(/<think>[\s\S]*?<\/think>/g, '');
  // 移除换行符
  result = result.replace(/\n+/g, '');

  // 按 <br> 分割
  let sentences = result.split('<br>').map(s => s.trim()).filter(s => s.length > 0);

  // 计算动态阈值
  const toleranceThreshold = Math.floor(maxWordCountEnglish * toleranceMultiplier);
  const warningThreshold = Math.floor(maxWordCountEnglish * warningMultiplier);
  const maxThreshold = Math.floor(maxWordCountEnglish * maxMultiplier);

  // 5 层防护机制
  const newSentences: string[] = [];
  const stats: SplitStats = {
    normal: 0,
    tolerated: 0,
    optimized: 0,
    forced: 0,
    rejected: 0,
  };

  for (const sentence of sentences) {
    // 首先按结束标记拆分句子
    const segments = splitByEndMarks(sentence);

    for (const segment of segments) {
      const wordCount = countWords(segment);

      // 层级1：正常范围 (≤ target)
      if (wordCount <= maxWordCountEnglish) {
        newSentences.push(segment);
        stats.normal++;
      }
      // 层级2：轻度容忍层 (target < x ≤ tolerance)
      else if (wordCount <= toleranceThreshold) {
        newSentences.push(segment);
        stats.tolerated++;
        logger.info(`✓ 轻度超标(${wordCount}/${maxWordCountEnglish}字): ${segment.slice(0, 40)}...`);
      }
      // 层级3：强制优化层 (tolerance < x ≤ warning)
      else if (wordCount <= warningThreshold) {
        logger.info(`🔧 尝试优化(${wordCount}/${maxWordCountEnglish}字): ${segment.slice(0, 40)}...`);
        const splitResults = aggressiveSplit(segment, maxWordCountEnglish);

        if (splitResults.length > 1) {
          stats.optimized++;
          logger.info(`✅ 优化成功: 分为${splitResults.length}段`);
          newSentences.push(...splitResults);
        } else {
          stats.tolerated++;
          logger.warn(`⚠️ 优化失败，接受原句(${wordCount}字)`);
          newSentences.push(segment);
        }
      }
      // 层级4：智能拆分层 (warning < x ≤ max)
      else if (wordCount <= maxThreshold) {
        logger.warn(`⚠️ 超出警告阈值(${wordCount}/${maxWordCountEnglish}字): ${segment.slice(0, 40)}...`);
        logger.info(`🔧 尝试智能分割...`);
        const splitResults = aggressiveSplit(segment, maxWordCountEnglish);

        if (splitResults.length > 1) {
          stats.optimized++;
          logger.info(`✅ 智能分割成功: 分为${splitResults.length}段`);
          newSentences.push(...splitResults);
        } else {
          logger.warn(`⚠️ 智能分割失败，使用降级分割`);
          const fallbackResults = fallbackSplit(segment, maxWordCountEnglish, warningThreshold);
          stats.forced++;
          newSentences.push(...fallbackResults);
        }
      }
      // 层级5：严重超标层 (> max)
      else {
        logger.error(`❌ 严重超标(${wordCount}/${maxWordCountEnglish}字): ${segment.slice(0, 40)}...`);
        logger.info(`🔧 尝试智能分割...`);
        const splitResults = aggressiveSplit(segment, maxWordCountEnglish);

        if (splitResults.length > 1) {
          stats.optimized++;
          logger.info(`✅ 智能分割成功: 分为${splitResults.length}段`);
          newSentences.push(...splitResults);
        } else {
          logger.warn(`⚠️ 智能分割失败，使用降级分割进行多次拆分`);
          const fallbackResults = fallbackSplit(segment, maxWordCountEnglish, warningThreshold);
          stats.rejected++;
          newSentences.push(...fallbackResults);
        }
      }
    }
  }

  sentences = newSentences;

  // 记录统计信息
  logger.info(`📊 断句质量统计:`);
  logger.info(`   ✅ 正常: ${stats.normal}句 (≤${maxWordCountEnglish}字)`);
  if (stats.tolerated > 0) {
    logger.info(`   ✓ 轻度超标: ${stats.tolerated}句 (${maxWordCountEnglish}-${toleranceThreshold}字)`);
  }
  if (stats.optimized > 0) {
    logger.info(`   🔧 优化拆分: ${stats.optimized}句 (${toleranceThreshold}-${warningThreshold}字)`);
  }
  if (stats.forced > 0) {
    logger.warn(`   🔨 强制拆分: ${stats.forced}句 (${warningThreshold}-${maxThreshold}字)`);
  }
  if (stats.rejected > 0) {
    logger.error(`   ❌ 严重超标: ${stats.rejected}句 (>${maxThreshold}字)`);
  }

  const batchPrefix = batchIndex ? `[批次${batchIndex}]` : '';
  logger.info(`✅ ${batchPrefix} 断句完成: ${sentences.length} 个句子`);

  return sentences;
}

/**
 * 按句子边界分批字幕（移植自 Python spliter.py:split_by_sentences）
 *
 * 1. 先按句子结束标记（.!?等）切分
 * 2. 按单词数阈值分组，但在句子边界处切分
 * 3. 如果单个句子超过阈值，在分句标点（,;等）处尝试切分
 *
 * @param subtitleData 字幕数据
 * @param wordThreshold 每批的单词数阈值（默认 500）
 * @returns 分批后的字幕数据数组
 */
export function splitByWordCount(
  subtitleData: SubtitleData,
  wordThreshold = 500
): SubtitleData[] {
  const segments = subtitleData.getSegments();

  // 句子结束标记
  const sentenceEndMarkers = ['.', '!', '?', '。', '！', '？', '…'];
  // 分句标点
  const splitMarkers = [',', '，', ';', '；', '、'];

  // 按句子切分
  const sentenceSegments: SubtitleEntry[][] = [];
  let currentSentenceSegments: SubtitleEntry[] = [];

  for (const seg of segments) {
    currentSentenceSegments.push(seg);
    const text = seg.text.trim();

    // 检查是否是句子结尾
    if (sentenceEndMarkers.some(marker => text.endsWith(marker))) {
      if (currentSentenceSegments.length > 0) {
        sentenceSegments.push(currentSentenceSegments);
        currentSentenceSegments = [];
      }
    }
  }

  // 处理最后一组未完成的句子
  if (currentSentenceSegments.length > 0) {
    sentenceSegments.push(currentSentenceSegments);
  }

  // 拆分过长的句子
  const splitLongSentence = (sentenceSegs: SubtitleEntry[]): SubtitleEntry[][] => {
    const result: SubtitleEntry[][] = [];
    let tempSegs: SubtitleEntry[] = [];
    let tempWordCount = 0;

    for (const seg of sentenceSegs) {
      const segText = seg.text.trim();
      const segWordCount = countWords(segText);

      // 如果当前段落加上之前的已经超过阈值，并且当前段落以分句标点结尾
      if (tempWordCount + segWordCount > wordThreshold &&
          splitMarkers.some(marker => segText.endsWith(marker))) {
        if (tempSegs.length > 0) {
          result.push(tempSegs);
          tempSegs = [];
          tempWordCount = 0;
        }
      }

      tempSegs.push(seg);
      tempWordCount += segWordCount;

      // 如果累积的单词数已经接近阈值，强制分段
      if (tempWordCount >= wordThreshold * 1.2) {
        if (tempSegs.length > 0) {
          result.push(tempSegs);
          tempSegs = [];
          tempWordCount = 0;
        }
      }
    }

    // 处理剩余的段落
    if (tempSegs.length > 0) {
      result.push(tempSegs);
    }

    return result;
  };

  // 按单词数阈值分组
  const batches: SubtitleData[] = [];
  let currentSegments: SubtitleEntry[] = [];
  let currentWordCount = 0;

  for (const sentence of sentenceSegments) {
    // 计算当前句子的单词数
    const sentenceText = sentence.map(seg => seg.text).join(' ');
    const sentenceWordCount = countWords(sentenceText);

    // 如果当前句子超过阈值，尝试拆分
    if (sentenceWordCount >= wordThreshold) {
      // 先保存当前批次
      if (currentSegments.length > 0) {
        batches.push(new SubtitleData(currentSegments));
        currentSegments = [];
        currentWordCount = 0;
      }

      // 拆分长句子
      const splitParts = splitLongSentence(sentence);
      for (const part of splitParts) {
        batches.push(new SubtitleData(part));
      }
      continue;
    }

    // 如果添加当前句子后超过阈值，先保存当前批次
    if (currentWordCount + sentenceWordCount > wordThreshold && currentSegments.length > 0) {
      batches.push(new SubtitleData(currentSegments));
      currentSegments = [];
      currentWordCount = 0;
    }

    currentSegments.push(...sentence);
    currentWordCount += sentenceWordCount;
  }

  // 处理最后一批
  if (currentSegments.length > 0) {
    batches.push(new SubtitleData(currentSegments));
  }

  return batches;
}

/**
 * 批量并行断句处理
 * 参考 Python 版本: spliter.py:merge_segments
 *
 * @param subtitleData 字幕数据
 * @param client OpenAI 客户端
 * @param config 配置
 * @param numThreads 并发线程数（默认 3）
 * @returns 处理后的字幕数据
 */
export async function mergeSegmentsBatch(
  subtitleData: SubtitleData,
  originalData: SubtitleData,
  client: OpenAIClient,
  config: TranslatorConfig,
  numThreads = 3
): Promise<SubtitleData> {
  const logger = setupLogger('断句合并');

  // 按单词数分批（用于生成文本和匹配）
  const wordThreshold = 500;  // 与 Python 版本一致
  const batches = splitByWordCount(subtitleData, wordThreshold);
  const totalBatches = batches.length;

  // 记录批次信息（与 Python 版本一致的日志格式）
  logger.info(`📋 批次规划: 每组${wordThreshold}字，共 ${totalBatches} 个批次`);

  // 显示批次分布
  const batchInfo: string[] = [];
  for (let i = 0; i < batches.length; i++) {
    const batchText = batches[i].toText();
    const wordCount = countWords(batchText);
    batchInfo.push(`批次${i + 1}: ${wordCount}字`);
  }
  logger.info(`批次详情: ${batchInfo.join(', ')}`);
  logger.info('🚀 开始并行断句处理...');

  // 并行处理每个批次
  const allSegments: SubtitleEntry[] = [];

  // 创建处理任务
  const tasks = batches.map(async (batch, index) => {
    const batchIndex = index + 1;
    const batchText = batch.toText();
    const wordCount = countWords(batchText);

    logger.info(`📝 [批次${batchIndex}] 处理 ${wordCount} 个单词`);

    // 调用 LLM 处理
    const sentences = await splitByLLM(batchText, client, config, batchIndex);

    // 🔍 调试：打印原始数据信息
    const batchSegments = batch.getSegments();
    logger.info(`🔍 批次${batchIndex} 时间戳范围: ${batchSegments[0]?.startTime}s - ${batchSegments[batchSegments.length - 1]?.endTime}s`);
    logger.info(`🔍 批次${batchIndex} 片段数量: ${batchSegments.length}`);

    // 使用相似度匹配重新分配时间戳（与 Python 版本一致）
    const resultSegments = mergeSegmentsBasedOnSentences(batchSegments, sentences);

    return resultSegments;
  });

  // 控制并发数
  const results: SubtitleEntry[][] = [];
  for (let i = 0; i < tasks.length; i += numThreads) {
    const chunk = tasks.slice(i, i + numThreads);
    const chunkResults = await Promise.all(chunk);
    results.push(...chunkResults);
  }

  // 合并所有结果
  for (const segments of results) {
    allSegments.push(...segments);
  }

  // 按时间排序
  allSegments.sort((a, b) => a.startTime - b.startTime);

  // 合并过短的分段（与 Python merge_short_segment 一致）
  mergeShortSegment(allSegments, config);

  // 重新编号
  allSegments.forEach((seg, idx) => {
    seg.index = idx + 1;
  });

  return new SubtitleData(allSegments);
}

/**
 * 合并过短的分段（与 Python merge_short_segment 一致）
 */
function mergeShortSegment(segments: SubtitleEntry[], config: TranslatorConfig): void {
  if (segments.length === 0) return;

  const maxWordCount = config.maxWordCountEnglish;
  let i = 0;

  while (i < segments.length - 1) {
    const currentSeg = segments[i];
    const nextSeg = segments[i + 1];

    // 判断是否需要合并:
    // 1. 时间间隔小于300ms
    // 2. 当前段落或下一段落词数小于5
    // 3. 合并后总词数不超过限制
    // 4. 当前段落不以句子结束标点结尾
    const timeGap = Math.abs(nextSeg.startTime - currentSeg.endTime);
    const currentWords = countWords(currentSeg.text);
    const nextWords = countWords(nextSeg.text);
    const totalWords = currentWords + nextWords;

    const endsWithPunctuation = /[.!?]$/.test(currentSeg.text);

    if (timeGap < 300 && (currentWords < 5 || nextWords <= 5) &&
        totalWords <= maxWordCount && !endsWithPunctuation) {
      // 执行合并操作
      logger.info(`合并优化: ${currentSeg.text} --- ${nextSeg.text}`);
      currentSeg.text += ' ' + nextSeg.text;
      currentSeg.endTime = nextSeg.endTime;

      // 移除下一个段落
      segments.splice(i + 1, 1);
      // 不增加i，继续检查合并后的段落
    } else {
      i++;
    }
  }
}
