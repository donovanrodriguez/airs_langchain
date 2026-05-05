# Prisma AIRS + LangChain Integration

## Project goal

Wrap any LangChain `BaseChatModel` with Palo Alto Prisma AIRS scanning on both
inbound prompts and outbound responses, so the wrapped model can drop into
existing chains, agents, and LangGraph nodes without code changes.

The deliverable is `airs_guarded_chat_model.py`, defining
`AIRSGuardedChatModel` (the wrapper) and `PrismaAIRSBlocked` (the exception
raised on a block verdict).

## Reference docs

- AIRS Python SDK overview: https://pan.dev/prisma-airs/api/airuntimesecurity/pythonsdk/
- Sync scan API reference: https://pan.dev/prisma-airs/api/airuntimesecurity/scan-sync-request/
- Use cases (response shape, masking semantics): https://pan.dev/prisma-airs/api/airuntimesecurity/usecases/
- AIRS feature page: https://pan.dev/airs/
- SDK on PyPI: https://pypi.org/project/pan-aisecurity/

When iterating on this code, prefer these docs over training-data recall — the
SDK has shipped breaking changes between minor versions.

## SDK shape

```python
import aisecurity
from aisecurity.scan.inline.scanner import Scanner
from aisecurity.generated_openapi_client.models.ai_profile import AiProfile
from aisecurity.scan.models.content import Content

aisecurity.init()  # reads PANW_AI_SEC_API_KEY from env if not passed
scanner = Scanner()
profile = AiProfile(profile_name="...")  # or profile_id=...
response = scanner.sync_scan(
    ai_profile=profile,
    content=Content(prompt="..."),  # or response="..."
    metadata={"app_user": "...", "app_name": "...", "ai_model": "..."},
)
```

Pass `metadata` as a plain dict, not via `aisecurity.generated_openapi_client.models.metadata.Metadata`.
The dict form is shown in the official pan.dev examples and is portable across
SDK versions; the `Metadata` class import path has shifted before.

## Key design decisions

### Decorator pattern, not replacement

`AIRSGuardedChatModel` *is-a* `BaseChatModel` and holds an `inner_llm`. Anything
that accepts a chat model accepts the wrapper. Don't refactor this into a
runnable or a callback — the type identity is what makes it drop-in.

### Honor `action`, not `category`

AIRS responses have two relevant fields:

- `category` — the verdict (`"malicious"` or `"benign"`)
- `action` — the policy decision your AIRS profile made (`"block"` or `"allow"`)

Block on `action == "block"`. The two diverge because each detection type
(prompt injection, DLP, URL filtering, toxic content, etc.) has its own
configurable action. Trusting `action` lets the security team tune the profile
in Strata Cloud Manager without redeploying the app.

### `fail_closed` governs SDK errors only, not block verdicts

Two distinct failure modes:

1. **Security verdict** (AIRS says "block") → `PrismaAIRSBlocked` always
   propagates, regardless of `fail_closed`.
2. **Infrastructure error** (`AISecSDKException`: network, auth, rate limit,
   oversize payload) → `fail_closed=True` propagates and blocks the request,
   `fail_closed=False` swallows it and continues unscanned.

Default is `fail_closed=True`. The fail-open path (`fail_closed=False`)
logs the exception via the module's `logger` (`logging.getLogger(__name__)`)
with `direction`, `chunk_bytes`, `app_user`, and `app_name` in `extra` —
silent fail-open is the worst of both worlds, so the wrapper never takes
that path. The chunk text itself is deliberately *not* logged: it can
contain user PII or the very payload that failed to scan.

The same logger emits INFO-level audit events whenever AIRS masks
content: `"AIRS masked prompt before LLM call"` (in `_generate` when the
prompt scan returns substituted text) and `"AIRS masked response
content"` (in `_generate` when a response scan substitutes content).
Both include `app_user` and `app_name` in `extra`. These give callers
an audit trail without inspecting return values; in production, route
this logger to your SIEM/audit pipeline.

### Masking is "block + masked_data"

AIRS DLP Mask mode emits `action="block"` *plus* `prompt_masked_data.data` /
`response_masked_data.data` carrying a sanitized version. The masked text is
the intended substitute, not a block signal.

The wrapper's logic: if `action == "block"` *and* masked data is present,
substitute and continue. If `action == "block"` and no masked data, raise
`PrismaAIRSBlocked`. The substitution is silent (per user choice).

On the prompt side, masking collapses the entire conversation to one
`HumanMessage` — we lose role structure but guarantee the LLM never sees
unmasked data. On the response side, masking mutates `gen.message.content`
in place, but only for plain-string content (multimodal list-of-parts content
is skipped). Response-side `tool_calls` use `mask_as_block` instead — see
the Tool-call scanning section.

If per-message masking matters, the fix is to scan messages individually
rather than as a flattened blob — not currently implemented.

### Chunking large payloads

AIRS sync API caps request bodies at 2MB. The wrapper chunks at 1.5MB to
leave headroom (`AIRS_SYNC_CHUNK_BYTES`). `_chunk_text` splits on UTF-8 byte
length (handles CJK and emoji), prefers newline boundaries, falls back to a
codepoint-aware byte cut.

`_chunk_text` has a contract that `_scan_text` depends on: every byte of
the input ends up in exactly one chunk, and `"".join(chunks)` reproduces
the input. Boundary bytes like `\n` go to the start of the next chunk via
`remaining[split_at:]`, never consumed. If you ever change the splitter
to drop a boundary byte, `_scan_text`'s masked reassembly will silently
corrupt the unmasked spans.

Chunks are scanned independently; any chunk that blocks (without mask) blocks
the whole request, masked chunks substitute in place. Tradeoff: a prompt
injection split across a chunk boundary could theoretically slip through.
Newline preference reduces this risk; if it becomes a real concern, the
upgrade is overlapping windows (each chunk includes ~1KB of the previous
chunk's tail).

### Streaming is buffered, not true streaming

True streaming would bypass post-hoc response scanning. `_stream` buffers via
`_generate`, scans the full response, then yields as one `ChatGenerationChunk`.
The caller gets a working `.stream()` interface; the streaming UX is degraded.
This is the right tradeoff when AIRS scanning is the point.

`ChatGenerationChunk` requires `AIMessageChunk`, not `AIMessage` — the
conversion is in `_stream` and is easy to forget.

### Tool-call scanning

`bind_tools` is overridden on the wrapper. Naive re-wrapping
(`AIRSGuardedChatModel(inner_llm=inner.bind_tools(...))`) doesn't work:
`bind_tools` returns a `RunnableBinding`, which isn't a `BaseChatModel`
and has no `_generate`. Instead, the override delegates formatting to
the inner LLM (each provider uses its own tool schema), then rebinds
those formatted kwargs on `self` via `self.bind(**inner_bound.kwargs)`.
At invoke time the kwargs flow through the wrapper's `_generate` to
`inner_llm._generate`, so AIRS still scans both prompt and response.

Tool calls are scanned in two places, with different semantics:

- **Response side** (`_generate`): `msg.content` and `msg.tool_calls` are
  scanned as separate AIRS requests. `content` uses normal mask-substitute
  semantics. `tool_calls` is scanned with `mask_as_block=True` — masked
  text can't be put back into a structured dict, so any DLP mask verdict
  there becomes a hard block. Otherwise the original sensitive value would
  leak through the unmodified `tool_calls` field while only `content`
  carried the masked version.
- **Prompt side** (`_messages_to_text`): historical AIMessages contribute
  both `content` and stringified `tool_calls` to the flattened prompt.
  This catches tool-arg injection in histories assembled outside this
  wrapper (the wrapper's response-side scan only sees what *it* generates).

Tool-argument injection is a real attack vector. The two-scan response
design exists specifically because joining content + tool_calls into a
single scan made it impossible to substitute masked content back without
either polluting `content` with a stringified tool-call dump or leaving
the structured `tool_calls` field unmasked.

### Pydantic specifics

- `model_config = {"arbitrary_types_allowed": True}` is required because
  `Scanner` and `AiProfile` aren't Pydantic models.
- `_scanner`, `_ai_profile`, and `_response_ai_profile` use
  `PrivateAttr(default=None)` to keep Pydantic from treating them as
  fields. They're set in `__init__` after `super().__init__(**kwargs)`.
- Profile validation (`profile_name or profile_id`) happens in `__init__`
  rather than via Pydantic validators — fails fast with a clear error.

## Exception design

`PrismaAIRSBlocked` exposes these attributes for log aggregators and incident
triage:

- `direction` — `"prompt"` or `"response"`
- `action` — the AIRS policy decision
- `category` — the verdict
- `scan_id`, `report_id` — for pivoting into the AIRS console (also in the
  message string so they land in stack traces)
- `detected` — the `prompt_detected` / `response_detected` block (tells you
  *what* was flagged: injection, dlp, url_cats, etc.)
- `scan_response` — raw object, for anything the named attributes miss

Catch sites typically want:
```python
except PrismaAIRSBlocked as e:
    logger.warning("airs blocked", extra={
        "direction": e.direction,
        "report_id": e.report_id,
        "detected": e.detected,
    })
```

## Configuration

Required environment:
- `PANW_AI_SEC_API_KEY` (or `PANW_AI_SEC_API_TOKEN`) — set before
  `aisecurity.init()` runs

Constructor:
- `inner_llm: BaseChatModel` — required
- `profile_name` or `profile_id` — at least one required (raises `ValueError`).
  When both are set, `profile_id` wins.
- `response_profile_name` or `response_profile_id` — optional. Used only
  on `direction="response"` scans. When unset, response scans reuse the
  base profile. Set these when the response side needs a stricter (or
  looser) profile than the prompt side. Same id-wins-over-name rule.
- `app_user`, `app_name` — surfaced in AIRS console metadata for forensics
- `fail_closed: bool = True` — see above

## Usage

```python
from langchain_openai import ChatOpenAI

guarded = AIRSGuardedChatModel(
    inner_llm=ChatOpenAI(model="gpt-4o-mini"),
    profile_name="prod-profile",
    app_user="user-123",
)

# Drop into anything that takes a chat model
chain = prompt | guarded | parser
result = chain.invoke({"input": "..."})
```

## Open items / future work

- **Per-message masking.** Currently masking on the prompt side collapses the
  conversation to one `HumanMessage`. If role preservation matters, scan
  messages individually instead of flattening.
- **Overlapping chunk windows.** If prompt injection splitting across chunk
  boundaries becomes a concern, add ~1KB of tail-overlap between chunks.
- **Async path (`_agenerate`/`_astream`).** Currently sync-only. Inner async
  LangChain calls fall back to running `_generate` in a threadpool. If real
  async is needed, the SDK has `aisecurity.scan.asyncio.scanner.Scanner`
  with awaitable `scan(...)`.
- **Streaming UX.** True streaming would require AIRS to support streaming
  scans (it doesn't, as of the docs referenced). Won't change without
  upstream API support.

## Don't

- Don't import `Metadata` from `aisecurity.generated_openapi_client.models.metadata`
  — use a plain dict.
- Don't check `category == "malicious"` to decide blocking — check
  `action == "block"`.
- Don't catch `PrismaAIRSBlocked` inside `_scan_chunk` or `_scan_text`.
  Block verdicts are meant to propagate.
- Don't truncate large prompts silently. Chunk them, or raise.
- Don't add `_stream` that yields token-by-token from the inner LLM directly
  — that bypasses response scanning entirely.
- Don't merge response `content` and `tool_calls` into one scan and substitute
  the result back into `content` — that pollutes content with a stringified
  tool-call dump and leaves the structured `tool_calls` field unmasked. Scan
  them separately; mask-substitute content, mask-as-block tool_calls.
- Don't drop `tool_calls` from `_messages_to_text` — historical tool calls
  in messages assembled outside this wrapper would never get scanned
  otherwise.