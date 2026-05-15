# airs_langchain

A LangChain `BaseChatModel` wrapper that scans prompts and responses through the Prisma AIRS sync API. Drop it around any chat model and every `invoke` / `generate` / `stream` call becomes a guarded call: the prompt is scanned before it reaches the inner LLM, the response is scanned before it returns to the caller, and DLP-flagged content is either masked in place or blocked depending on profile configuration.

## Quick start

```bash
pip install pan-aisecurity langchain-core langchain-anthropic python-dotenv
export PANW_AI_SEC_API_KEY=your-airs-key
export ANTHROPIC_API_KEY=sk-ant-...
```

`python-dotenv` is optional — only required if you pass `load_dotenv=True` or `dotenv_path=...` to the wrapper. With pure environment variables it is not needed.

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

## Integrating into an existing LangChain codebase

`AIRSGuardedChatModel` *is* a `BaseChatModel` — anywhere your code accepts or constructs a `BaseChatModel`, you can wrap it without touching the consumer. The pattern is always the same: keep your existing chat model object, wrap it once at construction, hand the wrapper to whatever was using the original.

The wrapper has no provider-specific code, so the same wrapping works for `ChatOpenAI`, `ChatAnthropic`, `ChatBedrock`, `ChatVertexAI`, `ChatGroq`, `ChatOllama`, custom subclasses — anything inheriting from `langchain_core.language_models.chat_models.BaseChatModel`.

### 1. Replace your chat model at the construction site

Before:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

After:

```python
from langchain_openai import ChatOpenAI
from airs_langchain import AIRSGuardedChatModel

llm = AIRSGuardedChatModel(
    inner_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    profile_name="prod-airs-profile",
    app_user="alice@example.com",   # for AIRS console audit
    app_name="my-service",
    load_dotenv=True,               # or rely on process env / pass api_key directly
)
```

Everything downstream — chains, agents, retrievers, evaluators — keeps using `llm` exactly as before. Every `invoke` / `generate` / `stream` / `batch` call now scans prompt + response.

### 2. LCEL pipelines

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

answer = chain.invoke({"question": "Summarize the latest sales report."})
```

The wrapper sits in the middle of the chain like any other `Runnable`. No changes to prompt templates, parsers, or chain composition.

### 3. Tool calling and agents

Always call `bind_tools` on the wrapper, **not** on the inner LLM:

```python
@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    ...

tooled = llm.bind_tools([search_docs])      # correct — AIRS still scans
# tooled = AIRSGuardedChatModel(inner_llm=inner.bind_tools([...]), ...)  # WRONG
```

`bind_tools` returns a `RunnableBinding`, which is not a `BaseChatModel` and exposes no `_generate` for the wrapper to call. The wrapper's `bind_tools` override delegates provider-specific schema formatting to the inner LLM, then rebinds the formatted kwargs on `self`, so AIRS still scans both sides and historical tool-call args in long-running agents are scanned on every turn.

Drop the bound wrapper into any agent constructor that takes a chat model:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools=[search_docs])
agent.invoke({"messages": [("human", "Find me the Q3 deck.")]})
```

For LangGraph nodes, pass `llm` (or `tooled`) into your node function just as you would the unwrapped model. The wrapper is stateless w.r.t. graph state and shares no state across calls.

### 4. Streaming

```python
for chunk in llm.stream("Tell me a joke."):
    print(chunk.content, end="")
```

The interface works. The semantics are buffered: the wrapper internally calls `_generate`, scans the full response, then yields it as a single `ChatGenerationChunk`. True token-by-token streaming is incompatible with post-hoc response scanning. If your UI shows a typing animation, you can keep it — just expect the full payload to arrive at once.

### 5. Async

`_agenerate` / `_astream` are not implemented. LangChain falls back to running the sync `_generate` in a threadpool when async callers invoke the wrapper (`ainvoke`, `astream`, async chains, LangGraph async nodes). This works but blocks a worker thread for each call. If your service is async-first and high-QPS, this is the limitation to plan around.

### 6. Error handling

Two failure modes to catch:

```python
from aisecurity.exceptions import AISecSDKException
from airs_langchain import PrismaAIRSBlocked

try:
    result = llm.invoke(user_input)
except PrismaAIRSBlocked as e:
    # AIRS verdict — surface to the user, write a structured audit log
    audit_logger.warning(
        "airs blocked",
        extra={
            "direction": e.direction,     # "prompt" or "response"
            "action": e.action,           # "block"
            "category": e.category,       # "malicious" / "benign"
            "report_id": e.report_id,     # pivot into the AIRS console
            "detected": e.detected,       # which detectors fired
            "app_user": "alice@example.com",
        },
    )
    return user_facing_block_message
except AISecSDKException as e:
    # Only reachable when fail_closed=True (the default) and the SDK itself
    # errored (network / auth / rate limit / oversize). With fail_closed=False
    # the wrapper logs and continues unscanned.
    audit_logger.exception("airs sdk error")
    raise
```

`PrismaAIRSBlocked` always propagates regardless of `fail_closed` — block verdicts are policy decisions, not infrastructure errors. The wrapper's `logger` (`airs_langchain` by default) emits INFO-level audit events on every mask substitution; route it to your SIEM or audit pipeline.

### 7. Caching

`_llm_type` is namespaced as `airs-guarded-<inner type>`. If you use LangChain's global LLM cache (`langchain.cache`), guarded and unguarded calls keep separate cache entries, which is the correct behavior — masked inputs and outputs differ from unmasked ones.

### 8. Replacing only certain code paths

If you want guarded scanning for some routes but not others (e.g. internal tools opt-out), keep the inner model as a separate object and wrap conditionally at the boundary:

```python
def get_llm(audit_required: bool) -> BaseChatModel:
    base = ChatAnthropic(model="claude-haiku-4-5-20251001")
    if not audit_required:
        return base
    return AIRSGuardedChatModel(inner_llm=base, profile_name="prod-airs-profile")
```

The wrapper is cheap to construct; reuse the same instance across requests rather than reconstructing per call.

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

The wrapper resolves AIRS credentials and endpoint in this order (highest → lowest):

1. **Explicit constructor args** — `api_key=`, `api_token=`, `api_endpoint=`.
2. **`.env` file** — loaded only when you pass `load_dotenv=True` or `dotenv_path="..."`. `python-dotenv` loads with `override=False`, so existing process env vars still win.
3. **Process env vars** — `PANW_AI_SEC_API_KEY`, `PANW_AI_SEC_API_TOKEN`, `PANW_AI_SEC_API_ENDPOINT`.
4. **SDK default** — anything left unresolved falls through to `aisecurity.init()`'s own env-read.

Empty strings are treated as unset at every level, so a blank ctor arg or stray shell var never shadows a real value.

| Variable | Purpose |
|---|---|
| `PANW_AI_SEC_API_KEY` | Prisma AIRS scan API key (most common form) |
| `PANW_AI_SEC_API_TOKEN` | Alternative token-based auth |
| `PANW_AI_SEC_API_ENDPOINT` | Override the API host for non-default regions/tenants |
| Profile name or ID | Constructor arg, not an env var. Required. |

Provide either `profile_name` or `profile_id` when constructing `AIRSGuardedChatModel`. One of the two is required; passing neither raises `ValueError`.

### Credential-source variants

```python
# Local dev / notebooks — auto-discover a .env in CWD or parents
guarded = AIRSGuardedChatModel(
    inner_llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
    profile_name="prod-profile",
    load_dotenv=True,
)

# Explicit .env path (e.g. operator-supplied)
guarded = AIRSGuardedChatModel(
    inner_llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
    profile_name="prod-profile",
    dotenv_path="/etc/myapp/airs.env",
)

# Secrets manager / programmatic injection
guarded = AIRSGuardedChatModel(
    inner_llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
    profile_name="prod-profile",
    api_key=secrets_client.get("airs/api_key"),
)

# Plain process env (12-factor; nothing extra to pass)
guarded = AIRSGuardedChatModel(
    inner_llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
    profile_name="prod-profile",
)
```

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
| `api_key` | `str` | `None` | Explicit AIRS API key. Overrides env/`.env` when set |
| `api_token` | `str` | `None` | Explicit AIRS API token. Overrides env/`.env` when set |
| `api_endpoint` | `str` | `None` | Explicit AIRS endpoint host. Overrides env/`.env` when set |
| `load_dotenv` | `bool` | `False` | Opt-in `.env` auto-discovery (CWD/parents). Off by default so importing this module never reads the filesystem unexpectedly |
| `dotenv_path` | `str` | `None` | Explicit `.env` path. Setting this implies loading even when `load_dotenv=False`. Requires `python-dotenv` |

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

`test_app.py` exercises six scenarios end-to-end:

1. happy path
2. prompt-injection block
3. DLP-shaped content (mask or block depending on profile config)
4. tool-call scanning with sensitive args (`mask_as_block` on tool args)
5. pre-built history with `tool_calls` (forces the role-prefixed flatten path)
6. benign tool call (exercises the response-side `tool_calls` scan)

Each section is independent — a block in one does not stop later sections. Setup:

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
