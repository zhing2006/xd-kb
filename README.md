# xd-kb

本地知识库引擎 CLI：基于 grafeo 的语义检索 + 图查询 + 文件 hash 跟踪。

完整规范见 `xd-spec/specs/Tools/KB/xd-kb-cli/spec.md`。

## 配置

复制 `.env.example` 为 `.env` 并填入 OpenAI 兼容 embedding API 的访问参数：

```env
XD_KB_API_BASE=https://your-endpoint/v1
XD_KB_API_KEY=sk-...
XD_KB_MODEL=voyage/voyage-4-large
```

### 可选环境变量

```env
# embedding 向量维度，缺省 1024。需与所选 model 一致。
XD_KB_DIM=1024

# 单 chunk 字符数上限，缺省 8192。结构性切分（按 heading 切）后若仍超限，
# 会按 "段落 (\n\n) → 句号 (。！？.!?) → 强制 char" 三级降级继续切段，
# 强制 char 切分按 UTF-8 char 边界，不破坏多字节字符。
XD_KB_MAX_CHUNK_CHARS=8192
```

`XD_KB_DIM` 常见取值：

| Model | dim |
|---|---|
| voyage-4-large | 1024 |
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |
| bge-m3 | 1024 |

`XD_KB_MAX_CHUNK_CHARS` 不同 model 的参考上限：

| Model | token 上限 | 建议 char 上限 |
|---|---|---|
| voyage-4-large | ~32K | 8192 – 16384 |
| bge-m3 | 8192 | ≤ 4096（中文 char×2 ≈ token） |
| text-embedding-3 系列 | 8191 | ≤ 3000（中文 OpenAI tokenizer 偏耗 token） |

默认 8192 是 voyage 优先的中庸值，换其他 model 时按上表自行调整；超限触发 4xx 时单文件回滚不污染数据，调小后重 ingest 即可。

### IndexConfig 防呆校验

每个 DB 在首次 `index` 时会写入 `IndexConfig {model, dim, max_chunk_chars}` 节点，后续 `index` / `query` 都会校验当前 env 与已存值一致；任一不一致 → 非零退出码报错（错误信息同时给出已存值与当前值），既不读也不写 KNN。

切换 model / dim / max_chunk_chars 的两种方式：

- 换一个新的 `--db` 路径（推荐：旧库可保留对照）
- 对原 DB 做 force-rebuild（清掉旧数据后重新 `index`）

### 迁移提示

当前尚未生成生产库，本次直接以默认值 `XD_KB_DIM=1024` / `XD_KB_MAX_CHUNK_CHARS=8192` 落地，无需 backfill。

## 子命令

所有子命令必须接收一个全局参数：`--db <path>`（grafeo DB 目录）。每个 DB 路径就是一个独立的 KB，不同 KB 之间通过不同 `--db` 路径做物理隔离。

全局可选参数（控制 embedding HTTP 请求的指数退避）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--backoff-base <secs>` | `2` | 第一次重试前的等待秒数（必须 > 0 且 finite） |
| `--backoff-exp <factor>` | `2` | 每次重试的退避乘子（必须 > 0 且 finite） |
| `--max-retries <n>` | `3` | embedding HTTP 最大重试次数 |

```bash
# 1. 把 markdown 目录建索引（首次或全量）
xd-kb --db ./kb.db index ./docs --batch-size 50

# 2. 增量更新（仅处理新增 / 修改 / 删除的文件）
xd-kb --db ./kb.db index ./docs --incremental

# 3. 语义查询（默认 cosine top-N）
xd-kb --db ./kb.db query "异步任务" --top 5

# 4. 语义查询 + MMR 多样性（lambda < 1.0 触发 MMR；越小越偏多样性）
xd-kb --db ./kb.db query "异步任务" --top 5 --lambda 0.5
xd-kb --db ./kb.db query "异步任务" --top 5 --lambda 0.3 --fetch-pool 40

# 5. 原生 GQL（语句原样转发，CLI 不注入任何谓词）
xd-kb --db ./kb.db gql "MATCH (c:Chunk) RETURN c.file_path LIMIT 10"

# 6. 检查文件 hash 状态（new / unchanged / modified / deleted）
#    --type 必填，9 个合法取值（kebab-case）：
#      vector / graph / llm / fulltext / summary / entity / vector-2 / custom / custom-2
xd-kb --db ./kb.db check ./docs/x.md --type vector
xd-kb --db ./kb.db check ./docs/x.md --type vector --mark
```

### Query：相关性 vs 多样性（`--lambda` / `--fetch-pool`）

`query` 子命令通过 `--lambda` 单标志在两条 KNN 路径之间分派：

| `--lambda` | 路径 | 行为 |
|---|---|---|
| `1.0`（默认） | `GrafeoDB::vector_search` | 纯 cosine top-N |
| `< 1.0` | `GrafeoDB::mmr_search` | MMR 重排：`1.0` 偏相关性、`0.0` 偏多样性 |

`--fetch-pool <N>` 控制 MMR 内部从 HNSW 取多少候选再做重排（即 `mmr_search` 的 `fetch_k`）：

- 默认 `top * 4`（与 grafeo 内置默认对齐），CLI 层面固化，不随 grafeo 上游默认值漂移。
- 必须 `>= --top`，否则 CLI 直接以非零退出码报错。
- `--lambda >= 1.0` 时该值被忽略（cosine 路径不消费候选池）。

`--lambda` 取值范围 `[0.0, 1.0]` 闭区间，越界（含 `--lambda -0.1` 这种空格分隔的负值）一律以非零退出码 + `--lambda must be in [0.0, 1.0]` 错误信息退出。

### 多维度 hash schema

`:FileHash` 节点为同一份 markdown 同时承载 9 个独立维度的 hash：`{path, vector_hash?, graph_hash?, llm_hash?, fulltext_hash?, summary_hash?, entity_hash?, vector_2_hash?, custom_hash?, custom_2_hash?}`，`path` 是唯一键。

- `index` 子命令只读写 `vector_hash`（增量比对、删除清理整节点 DETACH DELETE）。
- 其他 8 个维度（graph / llm / fulltext / summary / entity / vector-2 / custom / custom-2）的 hash 仅由 `check --type <dim> --mark` 维护：`--mark` 写入对应 `<dim>_hash` 属性，删除文件时若该 dim 是节点上最后一个非空 hash 则整节点删除，否则只 REMOVE 该属性。
- `check` 输出格式始终是单行 `[<status>] <file>`（不带 dim 后缀），`--mark` 真正清理了节点或属性时追加 ` -> [removed]`。
