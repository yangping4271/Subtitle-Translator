/**
 * 字符串相似度计算工具
 * 完全模拟 Python difflib.SequenceMatcher 的 Ratcliff/Obershelp 算法
 */

/**
 * 表示一个匹配块
 */
interface MatchingBlock {
  a: number; // 在text1中的起始位置
  b: number; // 在text2中的起始位置
  size: number; // 匹配长度
}

/**
 * 在给定范围内查找最长匹配
 * 完全模拟 Python difflib.SequenceMatcher.find_longest_match
 */
function findLongestMatch(
  text1: string,
  text2: string,
  alo: number,
  ahi: number,
  blo: number,
  bhi: number
): MatchingBlock {
  let besti = alo;
  let bestj = blo;
  let bestsize = 0;

  // 构建b2j映射（text2中每个字符出现的位置列表）
  const b2j: Map<string, number[]> = new Map();
  for (let j = blo; j < bhi; j++) {
    const ch = text2[j];
    if (!b2j.has(ch)) {
      b2j.set(ch, []);
    }
    b2j.get(ch)!.push(j);
  }

  // 对于text1中的每个字符，查找最长匹配
  for (let i = alo; i < ahi; i++) {
    const ch = text1[i];
    if (!b2j.has(ch)) continue;

    // 检查所有可能的匹配位置
    for (const j of b2j.get(ch)!) {
      // 计算从这个位置开始的匹配长度
      let k = 0;
      while (
        i + k < ahi &&
        j + k < bhi &&
        text1[i + k] === text2[j + k]
      ) {
        k++;
      }

      if (k > bestsize) {
        besti = i;
        bestj = j;
        bestsize = k;
      }
    }
  }

  return { a: besti, b: bestj, size: bestsize };
}

/**
 * 获取所有匹配块
 * 完全模拟 Python difflib.SequenceMatcher.get_matching_blocks
 */
function getMatchingBlocks(text1: string, text2: string): MatchingBlock[] {
  const matches: MatchingBlock[] = [];
  const queue: Array<[number, number, number, number]> = [
    [0, text1.length, 0, text2.length],
  ];

  while (queue.length > 0) {
    const [alo, ahi, blo, bhi] = queue.shift()!;
    const match = findLongestMatch(text1, text2, alo, ahi, blo, bhi);

    if (match.size > 0) {
      matches.push(match);

      // 递归处理匹配块之前和之后的部分
      if (alo < match.a && blo < match.b) {
        queue.push([alo, match.a, blo, match.b]);
      }
      if (match.a + match.size < ahi && match.b + match.size < bhi) {
        queue.push([match.a + match.size, ahi, match.b + match.size, bhi]);
      }
    }
  }

  // 按位置排序
  matches.sort((a, b) => a.a - b.a);
  return matches;
}

/**
 * 计算两个字符串的相似度（0-1之间的浮点数）
 * 完全模拟 Python 的 difflib.SequenceMatcher.ratio()
 * 使用 Ratcliff/Obershelp 模式识别算法
 *
 * @param text1 第一个字符串
 * @param text2 第二个字符串
 * @returns 相似度（0表示完全不同，1表示完全相同）
 */
export function calculateSimilarity(text1: string, text2: string): number {
  // 处理空字符串
  if (!text1 && !text2) {
    return 1.0;
  }
  if (!text1 || !text2) {
    return 0.0;
  }

  // 获取所有匹配块
  const matches = getMatchingBlocks(text1, text2);

  // 计算总匹配字符数
  const totalMatches = matches.reduce((sum, m) => sum + m.size, 0);

  // 相似度 = 2 * 匹配字符数 / 总字符数
  // 这与 Python difflib.SequenceMatcher.ratio() 完全一致
  return (2.0 * totalMatches) / (text1.length + text2.length);
}

/**
 * 标准化文本（移除多余空格）
 */
export function preprocessText(text: string): string {
  return text.split(/\s+/).filter(w => w.length > 0).join(' ');
}

/**
 * 查找最佳匹配位置
 *
 * @param sentence 要匹配的句子
 * @param segments 原始字幕段列表
 * @param startIndex 开始搜索的索引
 * @param maxShift 最大偏移量
 * @param threshold 相似度阈值
 * @returns 匹配结果 { position, windowSize, similarity } 或 null
 */
export function findBestMatch(
  sentence: string,
  segments: { text: string }[],
  startIndex: number,
  maxShift: number = 30,
  threshold: number = 0.5
): { position: number; windowSize: number; similarity: number } | null {
  const sentenceProc = preprocessText(sentence);
  const sentenceWordCount = sentenceProc.split(/\s+/).length;

  let bestRatio = 0.0;
  let bestPos: number | null = null;
  let bestWindowSize = 0;

  // 计算窗口大小范围
  const maxWindowSize = Math.min(sentenceWordCount * 2, segments.length - startIndex);
  const minWindowSize = Math.max(1, Math.floor(sentenceWordCount / 2));

  // 按接近目标词数的顺序尝试窗口大小
  const windowSizes = Array.from(
    { length: maxWindowSize - minWindowSize + 1 },
    (_, i) => minWindowSize + i
  ).sort((a, b) => Math.abs(a - sentenceWordCount) - Math.abs(b - sentenceWordCount));

  for (const windowSize of windowSizes) {
    const maxStart = Math.min(startIndex + maxShift + 1, segments.length - windowSize + 1);

    for (let start = startIndex; start < maxStart; start++) {
      // 合并窗口内的文本
      const substr = segments
        .slice(start, start + windowSize)
        .map(seg => seg.text)
        .join('');
      const substrProc = preprocessText(substr);

      // 计算相似度
      const ratio = calculateSimilarity(sentenceProc, substrProc);

      if (ratio > bestRatio) {
        bestRatio = ratio;
        bestPos = start;
        bestWindowSize = windowSize;
      }

      // 完全匹配，提前退出
      if (ratio === 1.0) {
        break;
      }
    }

    // 完全匹配，提前退出
    if (bestRatio === 1.0) {
      break;
    }
  }

  // 检查是否达到阈值
  if (bestRatio >= threshold && bestPos !== null) {
    return {
      position: bestPos,
      windowSize: bestWindowSize,
      similarity: bestRatio,
    };
  }

  return null;
}
