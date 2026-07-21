# Subtitle Translation

This context describes how source subtitles become readable, time-aligned translated subtitles while preserving terminology and playback timing.

## Language

**Source subtitle**:
The original English subtitle content and timing supplied for translation.
_Avoid_: ASR data, input text

**Word segment**:
A source-subtitle fragment whose timing covers one word or character and can be combined with adjacent fragments.
_Avoid_: Token, chunk

**Sentence segment**:
A readable subtitle sentence with a continuous time range, produced by grouping and aligning word segments.
_Avoid_: Split result, merged segment

**Subtitle segmentation**:
The process of turning source subtitles into sentence segments that satisfy readability and timing constraints.
_Avoid_: Smart split, LLM split

**Timeline alignment**:
The association between sentence text and the source time range in which that sentence is spoken.
_Avoid_: Timestamp merge, sentence matching

**Translation batch**:
An ordered group of sentence segments translated together while retaining each segment's identity.
_Avoid_: Chunk, request batch

**Translation result**:
The outcome for one sentence segment, including its source text, corrected text, translated text, and discard state.
_Avoid_: Response item, result dict
