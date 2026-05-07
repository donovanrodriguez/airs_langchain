# airs_langchain

A LangChain `BaseChatModel` wrapper that scans prompts and responses through the Prisma AIRS sync API. Drop it around any chat model and every `invoke` / `generate` / `stream` call becomes a guarded call: the prompt is scanned before it reaches the inner LLM, the response is scanned before it returns to the caller, and DLP-flagged content is either masked in place or blocked depending on profile configuration.

## Quick start

```bash
pip install pan-aisecurity langchain-core langchain-anthropic
export PANW_AI_SEC_API_KEY=your-airs-key
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from langchain_anthropic import ChatAnthropic
from airs_langchain import AIRSGuardedChatModel, PrismaAIRSBlocked

guarded = AIRSGuardedChatModel(
    inner_llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
    profile_name="my-airs-profile",
)

try:
    print(guarded.invoke("What is the capital of France?").content)
except PrismaAIRSBlocked as e:
    print(f"Blocked {e.direction}: {e.category} (report_id={e.report_id})")
```

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. `requirements.txt` lists:

- `pan-aisecurity`
- `langchain-core>=0.3`
- `langchain-anthropic>=0.3` (used by the demo; not required by the wrapper itself)
- `pydantic>=2.0`
- `python-dotenv>=1.0` (demo only)

## Configuration

The wrapper reads its credentials and profile from environment variables consumed by `aisecurity.init()`:

| Variable | Purpose |
|---|---|
| `PANW_AI_SEC_API_KEY` (or `PANW_AI_SEC_API_TOKEN`) | Prisma AIRS scan API key |
| Profile name or ID | Passed as a constructor arg, not an env var |

Provide either `profile_name` or `profile_id` when constructing `AIRSGuardedChatModel`. One of the two is required; passing neither raises `ValueError`.

## Usage

```python
from langchain_anthropic import ChatAnthropic
from airs_langchain import AIRSGuardedChatModel, PrismaAIRSBlocked

guarded = AIRSGuardedChatModel(
    inner_llm=ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0),
    profile_name="my-airs-profile",
    app_user="alice@example.com",
    app_name="my-app",
)

try:
    result = guarded.invoke("What is the capital of France?")
    print(result.content)
except PrismaAIRSBlocked as e:
    print(f"Blocked {e.direction}: {e.category}, report_id={e.report_id}")
```

The wrapper is itself a `BaseChatModel`, so it composes into LCEL chains like any other chat model:

```python
chain = prompt | guarded | parser
chain.invoke({"input": "..."})
```

### Tool calling

`bind_tools` is forwarded to the inner LLM so provider-specific tool schemas are preserved, then rebound on the wrapper so AIRS still scans both the prompt and the response, including stringified `tool_calls` on assistant messages.

```python
@tool
def lookup(id: str) -> str:
    ...

tooled = guarded.bind_tools([lookup])
result = tooled.invoke("Look up record 42")
```

Tool-call arguments are scanned with mask-as-block semantics: AIRS cannot mask values back into a structured `tool_calls` dict cleanly, so a mask verdict on tool args is escalated to a block rather than silently leaking the original value.

### Streaming

`_stream` buffers the full response, scans it, and yields a single `ChatGenerationChunk`. The streaming interface still works, but token-by-token streaming is not preserved because the AIRS sync API has no streaming-scan equivalent. True streaming would scan after delivery, defeating the point.

## Constructor arguments

| Argument | Type | Default | Notes |
|---|---|---|---|
| `inner_llm` | `BaseChatModel` | required | Any LangChain chat model |
| `profile_name` | `str` | `None` | One of `profile_name` or `profile_id` is required |
| `profile_id` | `str` | `None` | Wins over `profile_name` if both set |
| `response_profile_name` | `str` | `None` | Overrides the base profile for response-side scans only |
| `response_profile_id` | `str` | `None` | Wins over `response_profile_name`; falls back to base profile |
| `app_user` | `str` | `"unknown"` | Surfaces in AIRS console for audit and correlation |
| `app_name` | `str` | `"langchain-app"` | Surfaces in AIRS console |
| `fail_closed` | `bool` | `True` | Behavior on AIRS SDK errors. `True` re-raises, `False` logs and passes the chunk through unscanned |

## Behavior

### Verdict handling

- `action == "block"` with no `masked_data`: raises `PrismaAIRSBlocked`.
- `action == "block"` with `masked_data` (DLP "Block + Mask" mode): substitutes the masked text and continues.
- `action == "allow"` with `masked_data`: substitutes the masked text. Some profiles return masked content on allow verdicts.
- Allow with no mask: passes through unchanged.

### Chunking

The AIRS sync API caps request bodies at 2 MB. Payloads above the chunking threshold (1.5 MB, leaving headroom for the JSON envelope) are split on UTF-8 boundaries, scanned independently, and reassembled. Newline-preferred boundaries with codepoint-safe fallback ensure multi-byte text (CJK, emoji) is not corrupted across the cut. Any single chunk that blocks blocks the whole request; masked chunks are reassembled in order with the unmasked spans preserved byte-for-byte.

### `fail_closed` semantics

`fail_closed` governs only `AISecSDKException` (network errors, auth failures, rate limits, oversize requests). It does not affect AIRS *block verdicts*. A blocked prompt always raises `PrismaAIRSBlocked` regardless of `fail_closed`.

| `fail_closed` | AIRS SDK error | AIRS block verdict |
|---|---|---|
| `True` (default) | Re-raise `AISecSDKException`; deny the request | Raise `PrismaAIRSBlocked` |
| `False` | Log via `logger.exception`, pass the chunk through unscanned | Raise `PrismaAIRSBlocked` |

### Mask collapse on the prompt side

When AIRS masks a prompt, the wrapper collapses the entire conversation into a single `HumanMessage` whose content is the masked, role-flattened string. The original role structure (system / user / assistant / tool turns) is lost on that one masked call so that the LLM never sees unmasked content. Per-message mask granularity is not implemented; if you need it, scan messages individually outside the wrapper.

### Audit logging

The module logger (`airs_langchain` by default, or whatever `__name__` resolves to in your import) emits:

- `INFO` on every prompt-mask and response-mask event.
- `EXCEPTION` on every fail-open SDK error, including a traceback.

`PrismaAIRSBlocked` is not logged by the wrapper itself; the caller decides whether to log it. The exception carries `direction`, `action`, `category`, `scan_id`, `report_id`, and `detected` fields for structured logging and AIRS console pivots.

Route the logger to your SIEM in production. Block events should be alertable; fail-open events warrant investigation.

## `PrismaAIRSBlocked`

```python
class PrismaAIRSBlocked(Exception):
    direction: str          # "prompt" or "response"
    scan_response: Any      # raw AIRS response
    action: Optional[str]   # typically "block"
    category: Optional[str] # verdict category, e.g. "malicious", "benign"
    scan_id: Optional[str]
    report_id: Optional[str]
    detected: Any           # breakdown of detectors that fired
```

The string form embeds `action`, `category`, `scan_id`, and `report_id` so unstructured loggers still capture the IDs needed to pivot into the AIRS console.

## Compatibility

- Wraps any `langchain_core.language_models.chat_models.BaseChatModel` subclass: ChatOpenAI, ChatAnthropic, ChatBedrock, ChatVertexAI, ChatOllama, ChatGroq, custom subclasses. The wrapper has no provider-specific code.
- Pydantic v2 only.
- The `pan-aisecurity` SDK has shipped breaking changes between minor versions; pin it after a successful install (`pip freeze | grep pan-aisecurity`) if reproducibility matters.

## Caching

`_llm_type` is namespaced as `airs-guarded-<inner type>` so the LangChain global cache does not share entries between guarded and unguarded calls. Inputs and outputs differ after masking, so cache reuse across the boundary would be incorrect.

## Demo

`test_app.py` exercises four scenarios end-to-end: happy path, prompt-injection block, DLP-shaped content, and tool-call scanning. Each section is independent. Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in real values
python test_app.py
```

Required env vars for the demo: `PANW_AI_SEC_API_KEY`, `ANTHROPIC_API_KEY`, `AIRS_PROFILE_NAME`. The output annotates each section with what AIRS did, so a profile that does not have prompt-injection or DLP detectors enabled will surface a clear "AIRS did not flag this" line rather than failing silently.

## Limitations

- No async (`_agenerate` / `_astream`). Synchronous calls only.
- No true token streaming. `_stream` buffers, scans, and yields one chunk.
- Multimodal message content (lists of typed parts: text, image_url, etc.) stringifies via `repr` for scanning. Image bytes are not scanned; AIRS profiles are text-pattern based.
- Mask collapse on the prompt side flattens the conversation into a single `HumanMessage`. Models that depend on multi-turn role structure may behave differently on masked turns.
