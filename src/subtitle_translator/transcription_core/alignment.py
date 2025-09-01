from dataclasses import dataclass


@dataclass
class AlignedToken:
    id: int
    text: str
    start: float
    duration: float
    end: float = 0.0  # temporary

    def __post_init__(self) -> None:
        self.end = self.start + self.duration


@dataclass
class AlignedSentence:
    text: str
    tokens: list[AlignedToken]
    start: float = 0.0  # temporary
    end: float = 0.0  # temporary
    duration: float = 0.0  # temporary

    def __post_init__(self) -> None:
        self.tokens = list(sorted(self.tokens, key=lambda x: x.start))
        self.start = self.tokens[0].start
        self.end = self.tokens[-1].end
        self.duration = self.end - self.start


@dataclass
class AlignedWord:
    text: str
    start: float
    end: float
    duration: float


@dataclass
class AlignedResult:
    text: str
    sentences: list[AlignedSentence]

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def tokens(self) -> list[AlignedToken]:
        return [token for sentence in self.sentences for token in sentence.tokens]

    @property
    def words(self) -> list[AlignedWord]:
        words = []
        current_word_tokens = []

        for token in self.tokens:
            # Check for a new word, which typically starts with a space
            if token.text.startswith(" "):
                if current_word_tokens:
                    text = "".join(t.text for t in current_word_tokens).strip()
                    if text:
                        start = current_word_tokens[0].start
                        end = current_word_tokens[-1].end
                        duration = end - start
                        words.append(AlignedWord(text, start, end, duration))
                current_word_tokens = [token]
            else:
                current_word_tokens.append(token)

        # Append the last word
        if current_word_tokens:
            text = "".join(t.text for t in current_word_tokens).strip()
            if text:
                start = current_word_tokens[0].start
                end = current_word_tokens[-1].end
                duration = end - start
                words.append(AlignedWord(text, start, end, duration))

        return words


def tokens_to_sentences(tokens: list[AlignedToken]) -> list[AlignedSentence]:
    sentences = []
    current_tokens = []

    for idx, token in enumerate(tokens):
        current_tokens.append(token)

        # hacky, will fix
        if (
            "!" in token.text
            or "?" in token.text
            or "。" in token.text
            or "？" in token.text
            or "！" in token.text
            or (
                "." in token.text
                and (idx == len(tokens) - 1 or " " in tokens[idx + 1].text)
            )
        ):  # type: ignore
            sentence_text = "".join(t.text for t in current_tokens)
            sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
            sentences.append(sentence)

            current_tokens = []

    if current_tokens:
        sentence_text = "".join(t.text for t in current_tokens)
        sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
        sentences.append(sentence)

    return sentences


def sentences_to_result(sentences: list[AlignedSentence]) -> AlignedResult:
    return AlignedResult("".join(sentence.text for sentence in sentences), sentences)


def merge_longest_contiguous(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
):
    if not a or not b:
        return b if not a else a

    a_end_time = a[-1].end
    b_start_time = b[0].start

    if a_end_time <= b_start_time:
        return a + b

    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]

    enough_pairs = max(len(overlap_a) // 3, 10)  # 降低要求：1/3或至少10对

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    best_contiguous = []
    for i in range(len(overlap_a)):
        for j in range(len(overlap_b)):
            if (
                overlap_a[i].id == overlap_b[j].id
                and abs(overlap_a[i].start - overlap_b[j].start) < min(overlap_duration * 0.8, 10.0)
            ):
                current = []
                k, l = i, j
                while (
                    k < len(overlap_a)
                    and l < len(overlap_b)
                    and overlap_a[k].id == overlap_b[l].id
                    and abs(overlap_a[k].start - overlap_b[l].start)
                    < min(overlap_duration * 0.8, 10.0)
                ):
                    current.append((k, l))
                    k += 1
                    l += 1

                if len(current) > len(best_contiguous):
                    best_contiguous = current

    if len(best_contiguous) >= enough_pairs:
        a_start_idx = len(a) - len(overlap_a)
        lcs_indices_a = [a_start_idx + pair[0] for pair in best_contiguous]
        lcs_indices_b = [pair[1] for pair in best_contiguous]

        result = []
        result.extend(a[: lcs_indices_a[0]])

        for i in range(len(best_contiguous)):
            idx_a = lcs_indices_a[i]
            idx_b = lcs_indices_b[i]

            result.append(a[idx_a])

            if i < len(best_contiguous) - 1:
                next_idx_a = lcs_indices_a[i + 1]
                next_idx_b = lcs_indices_b[i + 1]

                gap_tokens_a = a[idx_a + 1 : next_idx_a]
                gap_tokens_b = b[idx_b + 1 : next_idx_b]

                if len(gap_tokens_b) > len(gap_tokens_a):
                    result.extend(gap_tokens_b)
                else:
                    result.extend(gap_tokens_a)

        result.extend(b[lcs_indices_b[-1] + 1 :])
        return result
    else:
        raise RuntimeError(f"No pairs exceeding {enough_pairs}")


def merge_longest_common_subsequence(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
):
    if not a or not b:
        return b if not a else a

    a_end_time = a[-1].end
    b_start_time = b[0].start

    if a_end_time <= b_start_time:
        return a + b

    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    dp = [[0 for _ in range(len(overlap_b) + 1)] for _ in range(len(overlap_a) + 1)]

    for i in range(1, len(overlap_a) + 1):
        for j in range(1, len(overlap_b) + 1):
            if (
                overlap_a[i - 1].id == overlap_b[j - 1].id
                and abs(overlap_a[i - 1].start - overlap_b[j - 1].start)
                < min(overlap_duration * 0.8, 10.0)
            ):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_pairs = []
    i, j = len(overlap_a), len(overlap_b)

    while i > 0 and j > 0:
        if (
            overlap_a[i - 1].id == overlap_b[j - 1].id
            and abs(overlap_a[i - 1].start - overlap_b[j - 1].start)
            < min(overlap_duration * 0.8, 10.0)
        ):
            lcs_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_pairs.reverse()

    if not lcs_pairs:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    a_start_idx = len(a) - len(overlap_a)
    lcs_indices_a = [a_start_idx + pair[0] for pair in lcs_pairs]
    lcs_indices_b = [pair[1] for pair in lcs_pairs]

    result = []

    result.extend(a[: lcs_indices_a[0]])

    for i in range(len(lcs_pairs)):
        idx_a = lcs_indices_a[i]
        idx_b = lcs_indices_b[i]

        result.append(a[idx_a])

        if i < len(lcs_pairs) - 1:
            next_idx_a = lcs_indices_a[i + 1]
            next_idx_b = lcs_indices_b[i + 1]

            gap_tokens_a = a[idx_a + 1 : next_idx_a]
            gap_tokens_b = b[idx_b + 1 : next_idx_b]

            if len(gap_tokens_b) > len(gap_tokens_a):
                result.extend(gap_tokens_b)
            else:
                result.extend(gap_tokens_a)

    result.extend(b[lcs_indices_b[-1] + 1 :])

    return result
