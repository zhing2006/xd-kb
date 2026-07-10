# xd-kb

本地知识库引擎 CLI：基于 grafeo 的语义检索 + 图查询 + 文件 hash 跟踪。

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
| voyage-context-4 | 1024（支持 256/512/1024/2048） |
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

### 上下文化嵌入（voyage-context 系列）

模型名含 `voyage-context`（如 `voyage/voyage-context-4`）时自动切换为上下文化 chunk 嵌入模式，无需额外配置：

- **索引**：同一文档的全部 chunk 作为一组、一文档一请求送嵌（嵌套 `input` + `input_type: "document"`），每个 chunk 向量同时编码块内细节与全文上下文；`--batch-size` 在此模式下不生效。单文档累计字符超过安全预算（20K chars）时按 chunk 边界拆为多组、多请求（组间失去互相上下文）。预算依据：模型单文档上下文窗口 32K token 为硬限（contextualized 不支持截断，超限直接报错），中文按 ~1.5 token/char 估算，20K chars ≈ 30K token 留有余量。此模式下 `XD_KB_MAX_CHUNK_CHARS` 不得超过 20000（启动时校验报错），否则单个 chunk 可能永远塞不进一次请求。
- **防降级**：一文档一请求的响应必须是两层嵌套形态；若网关回退返回普通扁平嵌入（丢失上下文语义），xd-kb 直接报错拒绝入库，不会静默降级。
- **并发**：`index --concurrency <n>`（默认 4）控制同时在飞的嵌入请求数；DB 写入始终串行有序，失败隔离粒度不变。一文档一请求会显著放大请求数（请求数 ≈ 文件数），全量重建大库时并发是主要提速手段。普通模型路径同样生效。
- **查询**：自动携带 `input_type: "query"`。
- **失败隔离**：粒度为单文档——某文档请求失败只影响该文件的 `vector_hash`，其余文件正常入库。
- **切换模型必须重建**：contextualized 与普通模型的向量不在同一语义空间，切换后按下方 IndexConfig 防呆的指引换新 `--db` 全量重建。

普通模型（模型名不含 `voyage-context`）行为不变：扁平 `input`、跨文件按 `--batch-size` 聚合。

### IndexConfig 防呆校验

每个 DB 在首次 `index` 时会写入 `IndexConfig {model, dim, max_chunk_chars}` 节点，后续 `index` / `query` 都会校验当前 env 与已存值一致；任一不一致 → 非零退出码报错（错误信息同时给出已存值与当前值），既不读也不写 KNN。

切换 model / dim / max_chunk_chars 的两种方式：

- 换一个新的 `--db` 路径（推荐：旧库可保留对照）
- 对原 DB 做 force-rebuild（清掉旧数据后重新 `index`）

### IndexConfig 初始化说明

生产库按默认值 `XD_KB_DIM=1024` / `XD_KB_MAX_CHUNK_CHARS=8192` 建立，无需 backfill。

## 子命令

所有子命令必须接收一个全局参数：`--db <path>`。每个 DB 路径就是一个独立的 KB，不同 KB 之间通过不同 `--db` 路径做物理隔离。路径支持两种存储格式：

- **单文件（推荐）**：`.grafeo` 扩展名（如 `.kb/wiki.grafeo`）。新路径带该扩展名即自动以单文件格式创建；close 时 checkpoint 全量落盘并清除 sidecar WAL，文件尺寸稳定。
- **WAL 目录（旧格式）**：目录路径（如 `.kb/wiki.db`）。日志只增不减、无 compaction，目录随写入持续膨胀；检测到该格式时除 `migrate` 外的命令会在 stderr 提示用 `migrate` 转换（详见下方 migrate 章节）。

全局可选参数（控制 embedding HTTP 请求的指数退避）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--backoff-base <secs>` | `2` | 第一次重试前的等待秒数（必须 > 0 且 finite） |
| `--backoff-exp <factor>` | `2` | 每次重试的退避乘子（必须 > 0 且 finite） |
| `--max-retries <n>` | `3` | embedding HTTP 最大重试次数 |

```bash
# 1. 把 markdown 目录建索引（首次或全量）
xd-kb --db ./kb.grafeo index ./docs --batch-size 50

# 1a. 提高嵌入请求并发（默认 4；DB 写入始终串行有序）
xd-kb --db ./kb.grafeo index ./docs --concurrency 8

# 2. 增量更新（仅处理新增 / 修改 / 删除的文件）
xd-kb --db ./kb.grafeo index ./docs --incremental

# 3. 语义查询（默认 cosine top-N）
xd-kb --db ./kb.grafeo query "异步任务" --top 5

# 3a. 语义查询 + 完整 chunk 输出（LLM Agent / RAG 默认开启，免再发 Read 取完整内容）
xd-kb --db ./kb.grafeo query "异步任务" --top 5 --full

# 4. 语义查询 + MMR 多样性（lambda < 1.0 触发 MMR；越小越偏多样性）
xd-kb --db ./kb.grafeo query "异步任务" --top 5 --lambda 0.5
xd-kb --db ./kb.grafeo query "异步任务" --top 5 --lambda 0.3 --fetch-pool 40

# 5. 原生 GQL（语句原样转发，CLI 不注入任何谓词）
xd-kb --db ./kb.grafeo gql "MATCH (c:Chunk) RETURN c.file_path LIMIT 10"

# 6. 检查文件 hash 状态（new / unchanged / modified / deleted）
#    --type 必填，9 个合法取值（kebab-case）：
#      vector / graph / llm / fulltext / summary / entity / vector-2 / custom / custom-2
xd-kb --db ./kb.grafeo check ./docs/x.md --type vector
xd-kb --db ./kb.grafeo check ./docs/x.md --type vector --mark

# 7. check 配合 --root：让写入的 :FileHash.path 与 index 的 :Chunk.file_path 对齐
xd-kb --db ./kb.grafeo check --root Wiki/example-kb 故事/X.md --type graph --mark

# 8. 旧 WAL 目录库迁移到单文件格式（纯本地拷贝，不触发任何 embedding 请求）
xd-kb --db ./kb.db migrate
xd-kb --db ./kb.db migrate --target ./elsewhere/kb.grafeo
```

### migrate：迁移单文件存储格式

`migrate` 把旧 WAL 目录格式的 KB 完整复制为单文件 `.grafeo` 格式：

- **拷贝范围**：全部节点（标签 + 全部属性，embedding 向量原样搬运）与全部边——`:Chunk` / `:FileHash`（9 维 hash）/ `:IndexConfig` / 图数据一个不落；全程零网络请求、零重新 embedding。
- **目标路径**：缺省为 `--db` 路径换 `.grafeo` 扩展名（`.kb/wiki.db` → `.kb/wiki.grafeo`），可用 `--target` 覆盖；目标必须带小写 `.grafeo` 扩展名（grafeo 按扩展名大小写敏感地分派单文件格式），且不能位于源库目录内部。
- **失败安全**：先写同目录隐藏临时文件（文件名含进程 pid），节点数 / 边数 / Chunk 数 / FileHash 数四项校验全部一致后才以原子 no-clobber 方式发布（目标在拷贝期间被他人创建则失败而不覆盖）；任何一步失败旧库原样不动、临时产物（含 grafeo 的 `.wal` / `.spill` sidecar）自动清理；旧目录改名 `.bak` 是提交点，越过后仅剩的清理动作（如锁文件删除）失败只在 stderr 告警，不影响迁移成功。迁移全程持有 `<db>.migrate-lock` 锁文件（按规范化路径生成，符号链接别名共享同一把锁）：并发 migrate 互斥，其它 xd-kb 命令检测到锁也会拒绝打开该库。仅内容形似 grafeo WAL 目录（含 `wal/wal_<数字>.log` 段文件）的路径才会被迁移，误传普通目录直接报错不动。
- **旧库处置**：迁移成功后旧目录改名为 `<原名>.bak` 保留，确认新库正常后手动删除。
- **收益**：WAL 历史全部压缩掉（一次全量重建后 110MB 的目录迁移后显著缩小），后续写入 close 时原位 checkpoint，尺寸不再单调增长。
- **限制**：HNSW 向量索引不迁移也不预建——grafeo 0.5.x 的单文件 checkpoint 虽会写入索引拓扑，但 open 时的加载路径丢弃索引定义、拓扑被无声跳过（上游 bug），预建只会白占文件空间。query 仍按每进程重建（见下），行为与迁移前一致；上游修复后可在 index/migrate 收尾一次性建索引。

### `.xd-kb.yaml` 配置（index 子命令）

`index` 子命令在 `<markdown-root>/.xd-kb.yaml` 找配置，决定哪些 markdown 入库：

```yaml
include:
  - "**/*.md"
exclude:
  - "drafts/**"
  - "**/scratch.md"
```

- `include` / `exclude` 字段 MUST 都为相对索引根的 glob 列表；`exclude` 优先于 `include`。
- 配置缺少 `include` 时视为 `["**/*.md"]`；缺少 `exclude` 时视为空列表。
- 引擎内置一组兜底 `DEFAULT_EXCLUDES = [".kb/**", ".xd-kb.yaml", "WIKI.md", "log.md"]`，与用户 `exclude` 集合并；用户 `include` 不可抵消（防止引擎自身产物 / 配置文件 / 常见 KB 副产物被入库）。
- `.xd-kb.yaml` 不存在时引擎按 `include = ["**/*.md"]` + `DEFAULT_EXCLUDES` 兜底处理，「无配置即可用」。
- yaml 解析失败、glob 编译失败 → CLI 以非零退出码报错并指出 `.xd-kb.yaml` 路径。

哪些文件入库完全由 `.xd-kb.yaml` 声明；引擎不消费任何 KB 专属约定（不识别 path 模式、不读取 frontmatter 字段做投影、不维护文件类型枚举）。

### `check --root <dir>` 路径对齐

`check` 子命令默认按用户传入字符串原样读盘 + 原样写库（保持向后兼容）。需要让 `check` 的 `:FileHash.path` 与 `index` 写入的 `:Chunk.file_path` 完全对齐时，加 `--root <dir>`：

- 显式 `--root <dir>` 时：`<file>` 视为相对 `<dir>` 的路径；xd-kb 内部读盘路径为 `<dir>/<file>`，写入的 `:FileHash.path` 为 `<file>`（规范化后的相对路径）。
- 隐式探测（未传 `--root`、cwd 向上找到 `.xd-kb.yaml`）：探测到的目录为隐式 root；`<file>` 视为相对 **cwd** 的路径，读盘路径为 `<cwd>/<file>`，写入的 `:FileHash.path` 为 `<cwd>` 相对隐式 root 的路径再拼 `<file>`（规范化后）。
- 两种模式共享同一组规范化与安全检查：分隔符 `\` / `/` / 混合都接受、统一写为 `/`；`./` 段被丢弃；非逃逸 `..` 段被弹出；逃逸 root 的 `..`、绝对路径、读盘失败均以非零退出码报错。
- 都未命中（无 `--root` 且无任何祖先 `.xd-kb.yaml`）时退回向后兼容分支：按用户原样字符串读盘 + 原样写库。

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

向量索引（HNSW）不跨进程持久化——WAL 目录格式天生不落盘索引；单文件格式受 grafeo 0.5.x 上游 bug 限制（checkpoint 写入拓扑但 open 不加载），两种格式行为一致：`index` / `migrate` 只写节点数据、不构建索引；`query` 每个进程启动时从节点数据重建一次（约 1ms/chunk，数千 chunk 的库约数秒），同进程内多次检索共享这一次构建。

`--full` 控制每条命中的 `content` 字段渲染：缺省（紧凑）只取首行并截断到 120 字符，便于人眼浏览；`--full` 原样输出完整 chunk content（保留所有换行、每行 4 空格缩进、命中之间用一个空行分隔），其他列（序号、相似度分数、`<file>:<start>-<end>` 头）不变。LLM Agent / RAG 调用方应默认开启，避免再发 Read 取完整 chunk。

### 多维度 hash schema

`:FileHash` 节点为同一份 markdown 同时承载 9 个独立维度的 hash：`{path, vector_hash?, graph_hash?, llm_hash?, fulltext_hash?, summary_hash?, entity_hash?, vector_2_hash?, custom_hash?, custom_2_hash?}`，`path` 是唯一键。

- `index` 子命令只读写 `vector_hash`（增量比对、增量删除走 per-dim 协议：DETACH DELETE 该文件的所有 `:Chunk` + REMOVE `:FileHash.vector_hash`；当且仅当节点上无其它非空 `<dim>_hash` 时整节点 DETACH DELETE，否则节点保留、其它 dim 的 hash 完整无损）。
- 其他 8 个维度（graph / llm / fulltext / summary / entity / vector-2 / custom / custom-2）的 hash 仅由 `check --type <dim> --mark` 维护：`--mark` 写入对应 `<dim>_hash` 属性，删除文件时若该 dim 是节点上最后一个非空 hash 则整节点删除，否则只 REMOVE 该属性。
- `check` 输出格式始终是单行 `[<status>] <file>`（不带 dim 后缀），`--mark` 真正清理了节点或属性时追加 ` -> [removed]`。
