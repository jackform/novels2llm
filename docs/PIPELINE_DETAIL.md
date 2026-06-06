# novels2llm 管线详细分析

> 本文档逐阶段、逐函数分析管线的输入、输出、工具调用和内部逻辑。
> 每章开头有横排流程图，后续以表格形式列出每个函数的 I/O。

---

## 目录

1. [整体架构](#整体架构)
2. [两条管线路径](#两条管线路径)
3. [Stage 1: 预处理](#stage-1-预处理)
4. [Stage 2: NLP 预标注](#stage-2-nlp-预标注)
5. [Smart Chunking 分块](#smart-chunking-分块)
6. [Stage 3: LLM 提取](#stage-3-llm-提取)
7. [Stage 4: 场景 + 叙事单元提取](#stage-4-场景--叙事单元提取)
8. [Stage 5: 别名消歧 + 实体合并](#stage-5-别名消歧--实体合并)
9. [Stage 6: 关系图谱](#stage-6-关系图谱)
10. [Stage 7: 导出](#stage-7-导出)
11. [跨块连接全流程](#跨块连接全流程)

---

## 整体架构

```
.md 文件
  │
  ├─ Stage 1: 预处理 (本地)
  │    YAML解析 → 去重 → 分章
  │    输出: PreprocessedNovel {metadata, chapters[], raw_text}
  │
  ├─ Stage 2: NLP 预标注 (本地, 仅测试管线)
  │    jieba分词 → 对话检测
  │    输出: NLPResult {tokens[], dialogues[]} → hints 注入 LLM Prompt
  │
  ├─ Smart Chunking (本地)
  │    章边界 → 句边界分块
  │    输出: Chunk[] (每块~3000字测试, 200字重叠)
  │
  ├─ Stage 3: LLM 提取 (API, 3 个 Extractor)
  │    ┌─ CharacterExtractor     → 角色 JSON
  │    ├─ WorldExtractor         → 世界观 JSON (仅首块)
  │    └─ RelationshipExtractor  → 关系 JSON
  │
  ├─ Stage 4: 场景 + 叙事单元提取 (API)
  │    SceneExtractor 单次 LLM 调用完成:
  │    ┌─ 场景分割 (按地点/时间变化)
  │    └─ 叙事单元提取 (对话/动作/叙述/内心独白, 含说话人归属)
  │    输出: Scene[{narrative_units[], location, participants, summary}]
  │
  │    ├─ 4.1: 全局 scene_id 重编号
  │    ├─ 4.2: 跨块场景合并 (启发式 + LLM 判定)
  │    └─ 4.2b: 从 narrative_units 提取对话 (向后兼容)
  │
  ├─ Stage 5: 别名消歧 (本地)
  │    三阶段合并 → name_map
  │
  ├─ Stage 6: 关系图谱 (本地)
  │    NetworkX DiGraph + 冲突解决
  │
  └─ Stage 7: 导出 (本地)
       JSON + SQLite (9张表) + Markdown 角色卡 (场景组织)
```

---

## 两条管线路径

项目存在两条并行的管线实现，功能集不同：

| 维度 | `cli.py cmd_run` | `save_extraction_results.py` |
|------|------------------|------------------------------|
| **Extractor 数量** | 3 (Character, World, Dialogue) | 4 (Character, World, Relationship, Scene) |
| **NLP 预标注** | 否 | 是 (jieba + regex 对话检测) |
| **场景提取** | 否 | 是 (SceneExtractor 替代 Event + Dialogue) |
| **对话来源** | DialogueExtractor 直接提取 | SceneExtractor → narrative_units 中筛选 |
| **跨块场景合并** | 否 | 是 (LLM 判定同场景) |
| **关系提取** | 否 (RelationshipExtractor 实例化但未调用) | 是 |
| **事件提取** | 否 (EventExtractor 实例化但未调用) | 否 (场景替代事件) |
| **缓存** | 是 (CACHE_DIR) | 否 (直接调用 extractor) |
| **场景角色卡** | 否 | 是 (对话按场景组织) |

`save_extraction_results.py` 是功能更完整的路径，代表了管线的演进方向。
本文档以下内容以测试管线 (`save_extraction_results.py`) 为主进行描述。

---

## Stage 1: 预处理

### 流程图

```
文件内容 (str, ~200KB)
  │
  ├─ _parse_yaml_frontmatter()
  │   ├── 工具: PyYAML (yaml.safe_load)
  │   ├── 输入: file_content: str
  │   └── 输出: (frontmatter: dict, body: str)
  │        例: ({title:"青春韵事", word_count:50536}, "### 第1章 正文\n...")
  │
  ├─ _extract_metadata()
  │   ├── 工具: 无 (纯字典取值)
  │   ├── 输入: frontmatter: dict, filename: str
  │   └── 输出: NovelMetadata {novel_id, title, author, source, word_count, chapter_count}
  │
  ├─ _remove_markdown_formatting()
  │   ├── 工具: re.sub (4条正则)
  │   ├── 输入: body: str (YAML后的原始正文, 含Markdown元数据展示块)
  │   └── 输出: str (清洗后纯正文)
  │        移除: # 标题行, **key**: value 行, ## 简介块, ## 章节列表标题
  │
  ├─ _deduplicate_text()
  │   ├── 工具: hashlib.md5
  │   ├── 输入: text: str
  │   └── 输出: (deduped_text: str, removed_count: int)
  │        策略: 对每行取前500字符做MD5 + 长度作为key, 重复则跳过
  │        短行(<20字, 如Markdown标题)不参与去重
  │
  └─ _split_into_chapters()
      ├── 工具: detect_chapter_boundaries() + extract_chapter_number()
      ├── 输入: text: str
      └── 输出: list[Chapter {number, title, text, start_pos, end_pos}]

最终聚合: preprocess_novel(filepath) → PreprocessedNovel
```

### 章节边界检测 (`chapter_splitter.py`)

| 函数 | 工具 | 输入 | 输出 | 逻辑 |
|------|------|------|------|------|
| `detect_chapter_boundaries(text)` | `re.finditer` × 4 个 pattern | `text: str` | `list[tuple[int,str,int]]` (start, title, end) | 4个正则轮流匹配 `### 第1章`、`第1章`、`第一章`、`### 第01章`。取完整行作为标题。去重（10字符内重复视为同一个）。如果第一个边界 >100字符，插入"前言"边界。如果没有匹配，返回单个"全文"边界 |
| `extract_chapter_number(title)` | `re.search` + `chinese_to_int()` | `title: str` 例: "第3章" | `int \| None` | 先匹配 `第(\d+)章` 取数字，再匹配 `第(中文数字)章` 转整数 |
| `chinese_to_int(num)` | 无 (纯算法) | `num: str` 例: "十二" | `int \| None` | 查表 `{零:0, 一:1, ... 十:10, 百:100, 千:1000}`，从右向左累积。例: "十二" → 2×1 + 1×10 = 12 |

---

## Stage 2: NLP 预标注

> **注意**: 此阶段仅在测试管线 (`save_extraction_results.py`) 中使用，主管线 (`cli.py cmd_run`) 跳过 NLP。

### 流程图

```
Chunk 文本 (str, ~3000字)
  │
  ├─ JiebaProcessor.process()
  │   ├── 工具: jieba.posseg.cut()
  │   ├── 输入: text: str
  │   └── 输出: list[NLPToken {word, pos, start, end}]
  │        ~2200 tokens/3000字块
  │
  └─ _extract_dialogues()
      ├── 工具: re.finditer(re.compile(r'\u201c([^\u201d]*)\u201d'))
      ├── 输入: text: str
      └── 输出: list[DialogueSpan {text, start, end, speaker, context_before}]
           每匹配一个 "" 弯引号内的文本 → 提取前80字作为上下文 → 尝试推断说话人
```

### 说话人推断 (`_infer_speaker`)

| 步骤 | 工具 | 逻辑 |
|------|------|------|
| 截取上下文 | `context[-20:]` | 只取最后20字符, 避免匹配到远端的无关人名 |
| 正则匹配 | `re.finditer` × 2个 pattern | ① `([\u4e00-\u9fff]{2,4})\s*(?:说道\|问道\|回答\|告诉\|说\|道\|问\|叫\|喊)` ② `([\u4e00-\u9fff]{2,4})\s*(?:心想\|暗想)` |
| 过滤停用词 | `STOP_WORDS` 集合 | 排除63个非人名高频词: 我/你/他/这/那/可以/突然/口中/大声... |
| 取最后匹配 | `max(m.start())` | 多个匹配取离引号最近的（位置最大） |

### 聚合函数

| 函数 | 输入 | 输出 | 逻辑 |
|------|------|------|------|
| `process_chapter(text, novel_id, chapter, use_hanlp=False, use_jieba=True)` | 文本 + 参数 | `NLPResult` | 顺序执行 jieba → 对话检测。HanLP 默认禁用 |
| `NLPResult.to_hint_dict()` | (自身) | `dict {"entities": [...], "dialogue_count": N, "dialogue_speakers": [...]}` | 为 LLM Prompt 准备的摘要，实体名去重，说话人集合 |

---

## Smart Chunking 分块

### 流程图

```
全书正文 (str, ~63000字)
  │
  ├─ detect_chapter_boundaries(text)
  │   └── 返回: [(0, "前言", 1000), (1000, "第1章", 63000)]
  │
  └─ chunk_novel(text, novel_id, target=3000, max=4000, overlap=200)
      │
      ├── 对每个章节:
      │   ├── if len(chapter) ≤ 4000: 整个章节 = 1个 Chunk
      │   └── if len(chapter) > 4000: 调用 _split_long_text()
      │
      └── _split_long_text(text, target=3000, max=4000, overlap=200)
          ├── 工具: re.finditer(SENTENCE_BOUNDARY = r'[。！？!?\n]')
          ├── 策略:
          │   1. 目标位置 = pos + 3000
          │   2. 在 [pos+1500, pos+4000] 范围内找最近的句子边界
          │   3. 找不到就硬切在3000字处
          │   4. 下一块起始 = 切点 - 200 (重叠)
          └── 输出: list[str] (子块文本列表)
```

### 关键数据结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `Chunk.novel_id` | `str` | 小说标识 (文件stem) |
| `Chunk.chunk_index` | `int` | 全局块编号 0, 1, 2... |
| `Chunk.chapter` | `int` | 所属章号 (1-based) |
| `Chunk.chapter_title` | `str` | 章标题 |
| `Chunk.text` | `str` | 块文本 (~3000字测试, ~6000字生产) |
| `Chunk.is_chapter_start` | `bool` | 是否章首块 |
| `Chunk.is_chapter_end` | `bool` | 是否章尾块 |
| `Chunk.char_count` | `int` | 字符数 |

### 配置值

| 配置 | 生产默认 (`config.py`) | 测试覆盖 (`save_extraction_results.py`) |
|------|----------------------|------------------------------------------|
| `TARGET_CHUNK_SIZE_CHARS` | 6000 | **3000** |
| `MAX_CHUNK_SIZE_CHARS` | 8000 | **4000** |
| `CHUNK_OVERLAP_CHARS` | 200 | 200 (未覆盖) |

---

## Stage 3: LLM 提取

### 整体调用流程

```
对每个 Chunk (0..N):
  │
  ├─ 从 nlp_map[chunk.chapter] 取 NLPResult.to_hint_dict()
  │   └── entity_hints: "文本中检测到的命名实体: 林鸿儒, 张淑惠, 陈雪芬, ..."
  │
  ├─ CharacterExtractor.extract(text, nlp_hints, known_characters)
  │   └── 每块调用, 结果累加到 all_chars
  │
  ├─ WorldExtractor.extract(text, nlp_hints)   [仅 Chunk 0 调用]
  │
  └─ RelationshipExtractor.extract(text, known_characters)
      └── 每块调用, known_characters 随处理递进增长
```

> **对比主管线**: `cli.py cmd_run` 的 `Stage3Pipeline.process_chunk()` 调用 CharacterExtractor + WorldExtractor + DialogueExtractor。
> RelationshipExtractor 和 EventExtractor 在主管线中被实例化但**未调用**。

### BaseExtractor 基类 (`extractors/base.py`)

**所有 Extractor 的公共流程：**

| 方法 | 工具 | 输入 | 输出 | 逻辑 |
|------|------|------|------|------|
| `__init__(api_key, base_url)` | `Anthropic(base_url=...)` | API key + base URL | 无 | 创建 Anthropic client。base_url 用于 DeepSeek 兼容 |
| `_load_prompt()` | `Path.read_text()` | (无, 从 `prompts/{PROMPT_FILE}.txt` 读) | `str` 模板文本 | 读 prompt 模板文件 |
| `_build_prompt(text, **kwargs)` | `str.format()` | `text: str` + 模板变量 | `str` 完整 prompt | 将文本和变量填入模板。JSON示例用 `{{ }}` 转义 |
| `_call_claude(prompt, max_tokens)` | `client.messages.create(model, max_tokens, messages)` | `prompt: str` | `str` LLM 响应文本 | 过滤 ThinkingBlock → 取 TextBlock.text |
| `_parse_json_response(response)` | `re.search` + `json.loads` | LLM 原始响应文本 | `dict` 解析后的 JSON | 先找 ```json...``` 代码块 → 再找裸 {...} → 再尝试直接解析。失败抛 JSONDecodeError |

**DeepSeek 兼容处理 (ThinkingBlock)：**
```
message.content = [ThinkingBlock(thinking="..."), TextBlock(text="...")]
                                    ↑ 跳过              ↑ 取这个
```

### 3 个 Extractor 详细 I/O（测试管线）

#### 1. CharacterExtractor

| 项 | 内容 |
|------|------|
| **Prompt 文件** | `prompts/character_extraction.txt` |
| **输入** | `text: str` (截断到12000字), `nlp_hints: dict` (实体列表), `known_characters: list[dict]` (已知角色) |
| **Prompt 组装** | `template.format(text=..., entity_hints="检测到的命名实体: ...")` + 追加上轮已知角色列表 |
| **LLM 输出** | `{"characters": [{canonical_name, aliases[], gender, age_range, appearance, personality, role, family_role, notes}, ...]}` |
| **返回** | `list[dict]` — `data.get('characters', [])` |
| **提取字段** | canonical_name, aliases[], gender, age_range, appearance, personality, role, family_role, notes |

#### 2. WorldExtractor

| 项 | 内容 |
|------|------|
| **Prompt 文件** | `prompts/world_setting.txt` |
| **输入** | `text: str` (截断到12000字), `nlp_hints: dict` |
| **调用频率** | 仅 Chunk 0（前两章），世界观信息集中在小说的开局部分 |
| **LLM 输出** | `{era, genre, setting_summary, locations[{name,type,description}], items[{name,type,description}], special_rules[], key_themes[]}` |
| **返回** | `dict` — 整个 JSON 对象 |
| **提取字段** | era, genre, setting_summary, locations[], items[], special_rules[], key_themes[] |

#### 3. RelationshipExtractor

| 项 | 内容 |
|------|------|
| **Prompt 文件** | `prompts/relationship_extraction.txt` |
| **输入** | `text: str` (截断到12000字), `known_characters: list[dict]` |
| **Prompt 组装** | `template.format(text=..., known_characters="林鸿儒 (别名: 小儒)\n张淑惠 (别名: 妈咪)\n...")` |
| **LLM 输出** | `{"relationships": [{character_a, character_b, rel_type, direction, intimacy_level, evidence[], confidence}, ...]}` |
| **返回** | `list[dict]` |
| **提取字段** | character_a, character_b, rel_type (spouse/parent/child/sibling/lover/friend/enemy/classmate/colleague/other), direction (bidirectional/a_to_b/b_to_a), intimacy_level, evidence[], confidence |

---

## Stage 4: 场景 + 叙事单元提取

> 这是管线中最核心的新增阶段。SceneExtractor 将原先分离的事件提取和对话归属合并为**一次 LLM 调用**，同时输出场景分割和叙事单元。

### SceneExtractor 概览

| 项 | 内容 |
|------|------|
| **类** | `SceneExtractor` (`extractors/scene_extractor.py`) |
| **Prompt 文件** | `prompts/scene_narrative_extraction.txt` |
| **输入** | `text` (截断到12000字), `chapter`, `chapter_title`, `known_locations`, `known_characters`, `previous_scene_summary` |
| **LLM 输出** | `{"scenes": [{scene_id, chapter, location, sub_location_of, participants[], summary, time_marker, narrative_units[{unit_id, character, text, type, listener, sequence_index}]}, ...]}` |
| **返回** | `list[dict]` — 场景列表 (含嵌套 narrative_units) |
| **max_tokens** | 16384 (覆盖默认 4096) |

### Prompt 变量组装

```
template.format(
    text=chunk.text,
    chapter=chunk.chapter,
    chapter_title=chunk.chapter_title,
    global_location_hint="已知地点: 医院病房, 林家客厅, 张淑惠卧室, ...",
    local_location_hint="本文中出现: 林家客厅, 医院病房",
    character_hint="已知角色: 林鸿儒 (别名: 小儒, 鸿儒)\n张淑惠 (别名: 妈咪, 母亲)\n...",
    previous_scene_summary="上一块的最后一个场景: 医院病房, 林鸿儒照顾生病的母亲"
)
```

### 场景提取的核心逻辑

```
SceneExtractor.extract(text, chapter, chapter_title=..., known_locations=..., known_characters=..., previous_scene_summary=...)
  │
  ├─ 构建提示 (hints):
  │   ├── global_location_hint: 全部已知地点列表
  │   ├── local_location_hint: 哪些地点在本文中出现 (字符串模糊匹配)
  │   ├── character_hint: 前30个已知角色及其别名
  │   └── previous_scene_summary: 上一块最后一个场景的摘要 (跨块连续性)
  │
  ├─ LLM 调用 (一次完成三件事):
  │   ├── 场景分割: 按地点变化或时间跳跃切分
  │   ├── 叙事单元提取: 对每个场景, 按顺序提取
  │   │   ├── dialogue: 对话 (含 action/表情描述前缀, 如 "（轻声道）妈咪...")
  │   │   ├── action: 动作
  │   │   ├── narration: 叙述
  │   │   └── inner_thought: 内心独白
  │   └── 说话人归属: 每个 dialogue 标注 character + listener
  │
  └─ 后处理:
      ├── chapter 覆写: 用传入的 chunk.chapter 替换 LLM 输出
      └── chunk_index 标注: 每个 scene 标记来源块
```

### 叙事单元 JSON 结构

```json
{
  "unit_id": "ch1_sc2_u0",
  "character": "林鸿儒",
  "text": "（轻声道）妈咪，我放学了",
  "type": "dialogue",
  "listener": "张淑惠",
  "sequence_index": 0
}
```

类型说明:
- **dialogue**: 有说话人的对话, 文本以（动作描述）为前缀
- **action**: 角色动作, character 为执行者
- **narration**: 叙述性文本, character 为 "narrator"
- **inner_thought**: 内心独白, 无 listener

### Stage 4.1: 全局 scene_id 重编号

每块的 LLM 都从 `sc1` 开始编号，导致跨块 ID 冲突。

```
每块场景: [{scene_id: "sc1", ...}, {scene_id: "sc2", ...}]
                                          ↓
重编号规则: scene_id = f"ch{chapter}_sc{counter}"  (全局递增)
                                          ↓
重编号后:  chunk0: ch1_sc1, ch1_sc2
          chunk1: ch1_sc3, ch1_sc4
          chunk2: ch1_sc5, ch1_sc6

同时更新每个 scene 内 narrative_units 的 unit_id 前缀:
  "sc1_u0" → "ch1_sc3_u0"
```

### Stage 4.2: 跨块场景合并

一个场景可能被块边界切断，需要跨块合并。

```
算法: merge_scenes_across_chunks(all_scenes)

  1. 按 chunk_index 分组场景
  2. 对相邻 chunk 组 (chunk N-1 的 tail vs chunk N 的 head):

     启发式触发条件:
       ├── tail.location == head.location  (同一地点)
       └── tail.participants ∩ head.participants ≠ ∅  (共享参与者)

     如果触发 → LLM 判定:
       Prompt: "这两个场景是否应该合并为同一个场景？
               上一块末尾: {tail.summary} (地点: {tail.location})
               当前块开头: {head.summary} (地点: {head.location})"
       LLM 返回: {"should_merge": true/false, "reason": "..."}

     如果合并:
       ├── head.narrative_units 追加到 tail.narrative_units
       ├── sequence_index 重新计算 (接续 tail 最后索引)
       ├── participants 取并集
       └── summary 拼接: f"{tail.summary}; {head.summary}"

     如果不合并 (或启发式未触发):
       └── 所有场景按原样追加

  3. 返回合并后的扁平场景列表
```

### Stage 4.2b: 从 narrative_units 提取对话 (向后兼容)

```
对每个 scene 的 narrative_units:
  if unit.type in ("dialogue", "inner_thought"):
    创建 Dialogue 对象:
      speaker = unit.character
      listener = unit.listener
      content = unit.text
      chapter = scene.chapter
      scene_id = scene.scene_id

→ all_extracted_dialogues (扁平列表, 保持与 DialogueExtractor 输出兼容)
```

---

## Stage 5: 别名消歧 + 实体合并

### 流程图

```
各块原始角色列表 (list[dict], ~20条)
  │
  └─ resolve_aliases(characters, sim_threshold=0.85, alias_threshold=0.5)
      │
      ├── Phase 0: 过滤 None 和 非 dict 条目
      │
      ├── Phase 1: 按 canonical_name.lower() 分组
      │    "林鸿儒" → [entry0, entry1, ...]
      │
      └── _merge_groups(groups, ...)
          对每对 group (i, j):
            ├── ① 精确名匹配: name_i.lower() == name_j.lower()
            ├── ② 别名重叠: _alias_overlap(aliases_i+[name_i], aliases_j+[name_j]) >= 0.5
            │     计算: |交集| / min(|集合A|, |集合B|)
            └── ③ 编辑距离: _name_similarity(name_i, name_j) >= 0.85
                  SequenceMatcher.ratio() 或 子串包含检查(0.9)
            满足任一条件 → 合并:
              - aliases 合并 + 去重 + 排序
              - 非 null 字段互补 (appearance/personality/gender/age/role)
              - canonical_name 从别名中移除

merge_character_entries(entries, canonical_name=None)
  └── 取最常见的 canonical_name (或指定)
  └── 字段取最长值 (max(values, key=len))
```

### 关键函数 I/O

| 函数 | 核心算法/工具 | 输入 | 输出 |
|------|-------------|------|------|
| `_name_similarity(a, b)` | `SequenceMatcher(None, a, b).ratio()` + 子串检查 | 两个名字字符串 | `float` 0-1 |
| `_alias_overlap(aliases_a, aliases_b)` | 集合交集/Jaccard-like | 两个别名列表 | `float` 0-1 |
| `resolve_aliases(characters)` | 三阶段贪心合并 | `list[dict]` 原始角色 | `list[dict]` 消歧后角色 |
| `merge_character_entries(entries)` | 频率统计 + 最长字段 | `list[dict]` 同人条目 | `dict` 合并后条目 |
| `merge_characters_across_chunks(all_chars)` | resolve → sort by role | `list[dict]` 全部角色 | `list[dict]` 按 role 排序 |

---

## Stage 6: 关系图谱

### 流程图

```
消歧后关系列表 (list[dict], 每块2-4条)
  │
  ├── 人名映射:
  │   对每条关系的 character_a / character_b 应用 name_map
  │   "小儒" → "林鸿儒", "妈咪" → "张淑惠"
  │   跳过 self-relationship (a == b)
  │   按 (a, b, rel_type) 三元组去重
  │
  └─ RelationshipGraph
      ├── 工具: networkx.DiGraph
      │
      ├── add_relationship(a, b, rel_type, direction, evidence, confidence)
      │   ├── 添加节点 (如果不存在)
      │   ├── 根据 direction 添加边:
      │   │   "bidirectional" → a→b, b→a 两条边
      │   │   "a_to_b"        → a→b 一条边
      │   │   "b_to_a"        → b→a 一条边
      │   └── 记录 _rel_types[(a,b).sorted] 用于冲突解决
      │
      ├── resolve_conflicts()
      │   对同一对角色有多个 rel_type:
      │     选 confidence 最高的 → 其次选 type 最长的（更specific）
      │     替换冲突边
      │
      ├── get_relationships() → list[dict]
      │   遍历所有边, 按 sorted(a,b) 去重, 返回唯一关系列表
      │
      ├── get_connected_characters(name, depth) → list[str]
      │   BFS 查询与指定角色相连的角色 (exclude self)
      │
      └── to_dict() → {nodes: [...], edges: [{source, target, rel_type, ...}]}
```

### 数据结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `graph` | `nx.DiGraph` | 有向图, 节点=角色名, 边属性: rel_type, evidence[], confidence |
| `_rel_types` | `dict[(str,str), list[dict]]` | 同一对角色可能被报告多种关系类型 (如母子+lover)，记录所有以供冲突解决 |

---

## Stage 7: 导出

| 函数 | 工具 | 输入 | 输出 |
|------|------|------|------|
| `export_to_json(novel_world, output_dir)` | `model_dump_json(indent=2)` + `Path.write_text()` | `NovelWorld` Pydantic 模型 | `novel_id.json` |
| `export_to_sqlite(novel_world, db_path)` | `sqlite3.connect()` | `NovelWorld` + 数据库路径 | `novels.db` (追加模式) |
| `export_character_cards(novel_world, output_dir)` | `Path.write_text()` | `NovelWorld` | `character_cards/novel_id/{角色名}.md` |

### SQLite Schema (9 张表)

```sql
novels          (novel_id, title, author, source, word_count, chapter_count, era, genre, world_summary)
characters      (id PK, novel_id FK, canonical_name, aliases[JSON], relational_labels[JSON], gender, age_range, appearance, personality, role, first_chapter)
relationships   (id PK, novel_id FK, character_a, character_b, rel_type, direction, intimacy_level, a_calls_b[JSON], b_calls_a[JSON], evidence[JSON], confidence)
dialogues       (id PK, novel_id FK, speaker, listener, content, context, chapter, line_index)
timeline        (id PK, novel_id FK, event_id, description, chapter, chapter_order, global_order, participants[JSON], time_marker)
locations       (id PK, novel_id FK, name, type, description, parent_location)
items           (id PK, novel_id FK, name, type, description, owner)
scenes          (id PK, novel_id FK, scene_id, chapter, location, sub_location_of, participants[JSON], summary, time_marker, chunk_index)
narrative_units (id PK, novel_id FK, scene_id, unit_id, character, text, type, listener, sequence_index)
```

新增表:
- **scenes**: 场景级信息，按地点/时间分割，含参与者、摘要和时间标记
- **narrative_units**: 每个场景内的叙事单元 (对话/动作/叙述/内心独白)，按 sequence_index 排序

### Character Card 模板 (场景组织)

角色卡现在以场景为组织维度展示对话摘录：

```markdown
# {canonical_name}
- 小说: {novel_title}
- **基本信息**: 性别={gender}, 年龄={age_range}, 角色={role}
- **别名**: {aliases}

## 外貌
{appearance}

## 性格
{personality}

## 关系
- {character_b}: {rel_type} (称呼: {calls_from_a})

## 场景对话摘录
### Ch1 Scene 2: 林家客厅
> （轻声道）妈咪，我放学了

### Ch1 Scene 3: 医院病房
> （关切地）妈，你今天感觉怎么样？
```
- 对话按 scene_id: location 分组展示
- 最多显示 10 个场景对话单元
- 去重逻辑: 按场景键去重

### 原始数据导出

测试管线额外导出中间数据到 `data/output/raw/{novel_id}/`:
```
all_chars.json           — 每块原始角色
resolved_chars.json      — 消歧后角色
relationships_raw.json   — 原始关系
relationships_mapped.json— name_map 映射后关系
dialogues.json           — 提取的对话
events.json              — (空, 场景替代事件)
world.json               — 世界观
graph.json               — NetworkX 图
name_map.json            — 人名映射表
scene_events.json        — SceneEvent 转换 (向后兼容)
scenes.json              — 完整的 Scene 列表
merged_scenes_raw.json   — 跨块合并前原始场景列表
```

---

## 跨块连接全流程

```
Chunk 0 (3000字)               Chunk 1 (3000字)               Chunk 2 (3000字)
  │                               │                               │
  ├─ NLP: tokens0, dialogues0    ├─ NLP: tokens1, dialogues1    ├─ NLP: tokens2, dialogues2
  ├─ Char: 4条角色               ├─ Char: 5条角色               ├─ Char: 5条角色
  │   林鸿儒(aliases: 小儒)     │   鸿儒(aliases: 亲哥哥)      │   儒(aliases: 小弟弟)
  │   张淑惠(aliases: 母亲)     │   张淑惠(aliases: 妈)        │   母亲(aliases: 妈)
  ├─ World: 世界观 ✓             ├─ World: 跳过                 ├─ World: 跳过
  ├─ Rel: 2条关系                ├─ Rel: 3条关系                ├─ Rel: 2条关系
  │   小儒 → 母亲                │   鸿儒 → 张淑惠              │   儒 → 母亲
  └─ Scene: 3个场景              └─ Scene: 4个场景              └─ Scene: 3个场景
      sc1: 林家客厅, [儒,惠]         sc1: 医院病房, [儒,惠]         sc1: 学校教室, [儒,芬]
      sc2: 医院病房, [儒,惠]         sc2: 走廊, [儒,芬]            sc2: 林家客厅, [儒,惠]
      sc3: 林家, [儒]                sc3: 林家客厅, [儒]           sc3: 卧室, [儒,惠]
                                    sc4: 卧室, [儒,惠]
        │                               │                               │
        └────── Stage 4.1 ─────┬──────── Stage 4.1 ──────┬────── Stage 4.1 ────┘
                               ↓                          ↓
                      scene_id 重编号:            scene_id 重编号:
                      ch1_sc1..ch1_sc3           ch1_sc4..ch1_sc10
                               │                          │
                               └────── Stage 4.2 ────────┘
                                     跨块场景合并
                               ┌──────────────────────────┐
                               │ Chunk 0 tail: ch1_sc3    │
                               │   地点: 林家, [儒]       │
                               │ Chunk 1 head: ch1_sc4    │
                               │   地点: 医院病房, [儒,惠]│
                               │ → 地点不同, 不合并       │
                               │                          │
                               │ Chunk 1 tail: ch1_sc7    │
                               │   地点: 卧室, [儒,惠]    │
                               │ Chunk 2 head: ch1_sc8    │
                               │   地点: 学校教室, [儒,芬]│
                               │ → 地点不同, 不合并       │
                               └──────────────────────────┘
                                      │
                                      ↓
                               Stage 4.2b: 从 narrative_units
                               提取 dialogue 和 inner_thought
                               → 向后兼容的对话列表
                                      │
        ┌─────────────────────────────┘
        ↓
Stage 5 别名消歧                  Stage 5 人名映射
14条角色 → 8条唯一角色            关系中的原始名 → 规范名
┌──────────────────┐             ┌──────────────────────┐
│ 林鸿儒 ← 小儒,   │             │ "小儒 → 母亲"         │
│          鸿儒,   │    ───→     │   → 林鸿儒 → 张淑惠  │
│          儒      │             │ "鸿儒 → 张淑惠"       │
│ 张淑惠 ← 母亲,   │             │   → 林鸿儒 → 张淑惠  │
│          妈      │             │ "儒 → 母亲"           │
└──────────────────┘             │   → 林鸿儒 → 张淑惠  │
                                 └──────────────────────┘
                                         ↓
                                按 (a,b,type) 去重
                                林鸿儒─张淑惠 → 1条关系
                                         ↓
                                Stage 6 NetworkX 图
                                6 nodes, 8 edges
```

---

## 已知缺口

| # | 缺口 | 说明 |
|---|------|------|
| ① | **主管线未集成 SceneExtractor** | `Stage3Pipeline.process_chunk()` 仍用旧的 DialogueExtractor，场景提取仅在测试管线中可用 |
| ② | **EventExtractor 空置** | 主/测试管线均不再使用事件提取，但 extractor 类仍保留 |
| ③ | **场景 text_start/text_end 缺失** | 叙事单元按顺序排列即可，无原文精确位置索引 |
| ④ | **关系类型未校验** | 母子关系被误标为 sibling 仍可能通过，缺少合理性校验 |
| ⑤ | **时间线未填充** | 测试管线构建 NovelWorld 时 `events=[]`，时间线为空 |
| ⑥ | **缓存未启用于测试管线** | 测试管线直接调用 extractor，绕过了 Stage3Pipeline 的缓存层 |
