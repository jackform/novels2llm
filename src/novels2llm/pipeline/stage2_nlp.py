"""Stage 2: NLP preprocessing - jieba segmentation + HanLP NER + dialogue extraction."""

import re
from typing import Optional, Iterator
from dataclasses import dataclass, field

from ..config import config


@dataclass
class NLPToken:
    """A token with POS tag."""

    word: str
    pos: str
    start: int
    end: int


@dataclass
class NEREntity:
    """A named entity recognized by HanLP or jieba."""

    text: str
    label: str  # PERSON, LOCATION, ORG, TIME
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class DialogueSpan:
    """A dialogue segment."""

    text: str
    start: int
    end: int
    speaker: Optional[str] = None  # May be inferred from context
    context_before: str = ""  # Narration before the dialogue


@dataclass
class NLPResult:
    """Output of Stage 2 NLP processing."""

    novel_id: str
    chapter: int
    tokens: list[NLPToken] = field(default_factory=list)
    entities: list[NEREntity] = field(default_factory=list)
    dialogues: list[DialogueSpan] = field(default_factory=list)
    full_text: str = ""

    def to_hint_dict(self) -> dict:
        """Convert to a dict suitable for inclusion in LLM prompts."""
        entity_summary = {}
        for e in self.entities:
            key = f"{e.label}:{e.text}"
            if key not in entity_summary:
                entity_summary[key] = e.text
        return {
            "entities": list(entity_summary.values()),
            "dialogue_count": len(self.dialogues),
            "dialogue_speakers": list(set(
                d.speaker for d in self.dialogues if d.speaker
            )),
        }


class JiebaProcessor:
    """jieba word segmentation + POS tagging."""

    def __init__(self):
        import jieba
        import jieba.posseg as pseg
        self.jieba = jieba
        self.pseg = pseg

    def process(self, text: str) -> list[NLPToken]:
        """Segment text and return tokens with POS tags."""
        tokens = []
        pos = 0
        for word, flag in self.pseg.cut(text):
            # Find approximate position
            try:
                start = text.index(word, pos)
            except ValueError:
                start = pos
            end = start + len(word)
            pos = end
            tokens.append(NLPToken(word=word, pos=flag, start=start, end=end))
        return tokens


class HanLPProcessor:
    """HanLP NER annotation."""

    def __init__(self):
        import hanlp
        self.ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)

    def process(self, text: str) -> list[NEREntity]:
        """Run NER on text and return entities."""
        results = self.ner(text)
        entities = []
        for entity_text, label, start, end in results:
            # Map HanLP labels to common labels
            label_map = {
                'PERSON': 'PERSON',
                'NR': 'PERSON',
                'LOCATION': 'LOCATION',
                'NS': 'LOCATION',
                'ORGANIZATION': 'ORG',
            }
            mapped_label = label_map.get(label, label)
            entities.append(NEREntity(
                text=entity_text,
                label=mapped_label,
                start=start,
                end=end,
            ))
        return entities


class FallbackNERProcessor:
    """Fallback NER using jieba POS tags when HanLP is unavailable."""

    def __init__(self):
        import jieba.posseg as pseg
        self.pseg = pseg

    def process(self, text: str) -> list[NEREntity]:
        """Extract entities from POS tags."""
        entities = []
        pos = 0
        for word, flag in self.pseg.cut(text):
            try:
                start = text.index(word, pos)
            except ValueError:
                start = pos
            end = start + len(word)
            pos = end

            if flag == 'nr':
                entities.append(NEREntity(
                    text=word, label='PERSON', start=start, end=end,
                ))
            elif flag == 'ns':
                entities.append(NEREntity(
                    text=word, label='LOCATION', start=start, end=end,
                ))
            elif flag == 'nt':
                entities.append(NEREntity(
                    text=word, label='ORG', start=start, end=end,
                ))
            elif flag == 't':
                entities.append(NEREntity(
                    text=word, label='TIME', start=start, end=end,
                ))
        return entities


def _extract_dialogues(text: str) -> list[DialogueSpan]:
    """Extract dialogue segments using Chinese curly quotes."""
    dialogues = []

    # Match content inside Chinese curly quotes ""
    pattern = re.compile(r'\u201c([^\u201d]*)\u201d')

    for m in pattern.finditer(text):
        content = m.group(1)
        start = m.start()
        end = m.end()

        # Get context before the dialogue (up to 50 chars)
        ctx_start = max(0, start - 80)
        ctx_before = text[ctx_start:start].strip()

        # Try to infer speaker from context patterns
        speaker = _infer_speaker(ctx_before)

        dialogues.append(DialogueSpan(
            text=content,
            start=start,
            end=end,
            speaker=speaker,
            context_before=ctx_before,
        ))

    return dialogues


def _infer_speaker(context: str) -> Optional[str]:
    """Try to extract speaker name from context before dialogue.

    Looks for patterns like "XXX说", "XXX道", "XXX问" right before the quote.
    Only matches if the name is found within the last 10 chars of context.
    """
    # Non-name characters that should never be treated as names
    STOP_WORDS = {
        '可以', '可能', '应该', '必须', '一定', '也许', '只能', '还', '就', '都', '又',
        '再', '已经', '终于', '突然', '然后', '接着', '立刻', '马上', '赶紧', '连忙',
        '这', '那', '哪', '怎么', '什么', '谁', '哪', '怎样', '如何',
        '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们',
        '中', '上', '下', '前', '后', '里', '外', '来', '去', '想', '说',
        '不', '很', '都', '也', '还', '要', '会', '能', '是', '有',
        '大', '小', '多', '少', '一', '二', '三',
        '一个', '一位', '一口', '大声', '小声', '低声', '轻', '悄悄',
        '口中', '嘴里', '心里', '心中', '口', '嘴',
    }

    # Only search the last 15 characters of context for speaker attribution
    # This avoids picking up names from earlier sentences
    if len(context) > 20:
        context = context[-20:]

    # Pattern: name + speech verb, where name appears at or near end of context
    patterns = [
        # "XXX说/道/问/回答/叫/喊/告诉/说道/问道/问道"
        r'([\u4e00-\u9fff]{2,4})\s*(?:说道|问道|回答|告诉|说|道|问|叫|喊)',
        # "XXX心想/暗想"
        r'([\u4e00-\u9fff]{2,4})\s*(?:心想|暗想)',
    ]

    # Find the last match (closest to the dialogue)
    last_match = None
    last_pos = -1
    for pattern in patterns:
        for m in re.finditer(pattern, context):
            if m.start() > last_pos:
                name = m.group(1)
                if name not in STOP_WORDS and not any(c in '的口中来去' for c in name):
                    last_match = name
                    last_pos = m.start()

    return last_match


def process_chapter(
    text: str,
    novel_id: str,
    chapter: int,
    use_hanlp: bool = True,
    use_jieba: bool = True,
) -> NLPResult:
    """Run NLP processing on a chapter.

    Returns NLPResult with tokens, entities, and dialogue spans.
    """
    result = NLPResult(
        novel_id=novel_id,
        chapter=chapter,
        full_text=text,
    )

    # Step 1: jieba segmentation + POS tagging
    if use_jieba and config.NLP_ENABLE_JIEBA:
        try:
            jieba_proc = JiebaProcessor()
            result.tokens = jieba_proc.process(text)
        except Exception:
            pass

    # Step 2: NER (try HanLP first, fall back to jieba-based)
    hanlp_success = False
    if use_hanlp and config.NLP_ENABLE_HANLP:
        try:
            hanlp_proc = HanLPProcessor()
            result.entities = hanlp_proc.process(text)
            hanlp_success = True
        except Exception:
            pass

    if not hanlp_success:
        try:
            fallback = FallbackNERProcessor()
            result.entities = fallback.process(text)
        except Exception:
            pass

    # Step 3: Dialogue extraction
    result.dialogues = _extract_dialogues(text)

    return result
