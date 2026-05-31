"""Debug dialogue and relationship extraction."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from anthropic.types import TextBlock
from src.novels2llm.pipeline.stage1_preprocess import preprocess_novel
from src.novels2llm.chunking.smart_chunker import chunk_novel
from src.novels2llm.config import config

f = Path('data/jia_ting_luan_lun/qing-chun-yun-shi.md')
result = preprocess_novel(f)
chunks = chunk_novel(result.raw_text, result.metadata.novel_id)
chunk = chunks[0]
client = Anthropic(api_key=config.ANTHROPIC_API_KEY, base_url=config.ANTHROPIC_BASE_URL)

# Test dialogue prompt
dlg_template = (config.get_prompt_path("dialogue_extraction")).read_text(encoding='utf-8')
dlg_prompt = dlg_template.format(
    text=chunk.text[:8000],
    dialogue_count=51, speaker_hints="林鸿儒, 张淑惠"
)
print("=== DIALOGUE PROMPT (last 300 chars) ===")
print(dlg_prompt[-300:])
print()

message = client.messages.create(
    model=config.ANTHROPIC_MODEL,
    max_tokens=4096,
    messages=[{"role": "user", "content": dlg_prompt}],
)
raw = ""
for block in message.content:
    if isinstance(block, TextBlock):
        raw = block.text
        break
print(f"=== DIALOGUE RAW RESPONSE ({len(raw)} chars) ===")
print(raw[:2000])
print("...")
print(raw[-500:] if len(raw) > 500 else "")

# Test relationship prompt too
print("\n\n=== RELATIONSHIP RAW RESPONSE ===")
rel_template = (config.get_prompt_path("relationship_extraction")).read_text(encoding='utf-8')
known = "- 林鸿儒 (别名: 小儒, 儿子)\n- 张淑惠 (别名: 妈咪, 母亲)\n- 陈雪芬 (别名: 陈老师)"
rel_prompt = rel_template.format(text=chunk.text[:8000], known_characters=known)

message2 = client.messages.create(
    model=config.ANTHROPIC_MODEL,
    max_tokens=4096,
    messages=[{"role": "user", "content": rel_prompt}],
)
raw2 = ""
for block in message2.content:
    if isinstance(block, TextBlock):
        raw2 = block.text
        break
print(f"({len(raw2)} chars)")
print(raw2[:2000])
