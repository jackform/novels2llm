"""Quick test: extract characters from first chunk only, with debug output."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.novels2llm.pipeline.stage1_preprocess import preprocess_novel
from src.novels2llm.chunking.smart_chunker import chunk_novel
from src.novels2llm.config import config
from anthropic import Anthropic

f = Path('data/jia_ting_luan_lun/qing-chun-yun-shi.md')
result = preprocess_novel(f)
chunks = chunk_novel(result.raw_text, result.metadata.novel_id)
chunk = chunks[0]
print(f'Processing chunk 0: {chunk.char_count} chars')
print(f'Text starts with: {chunk.text[:300]}...\n')

# Load prompt
prompt_path = config.get_prompt_path("character_extraction")
template = prompt_path.read_text(encoding='utf-8')
prompt = template.format(text=chunk.text[:8000], entity_hints="无")
print(f'Prompt length: {len(prompt)} chars\n')

# Call Claude directly
client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
print('Calling Claude API...')
from anthropic.types import TextBlock

message = client.messages.create(
    model=config.ANTHROPIC_MODEL,
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}],
)
# DeepSeek may return ThinkingBlock; extract TextBlock
raw_response = ""
for block in message.content:
    if isinstance(block, TextBlock):
        raw_response = block.text
        break
if not raw_response and hasattr(message.content[0], 'text'):
    raw_response = message.content[0].text
if not raw_response:
    raw_response = str(message.content)
print(f'=== RAW RESPONSE ({len(raw_response)} chars) ===')
print(raw_response[:3000])
print('...')
print(raw_response[-500:])
