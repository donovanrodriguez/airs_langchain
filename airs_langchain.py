import logging
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import aisecurity
from aisecurity.scan.inline.scanner import Scanner
from aisecurity.generated_openapi_client.models.ai_profile import AiProfile
from aisecurity.scan.models.content import Content
from aisecurity.exceptions import AISecSDKException

from pydantic import PrivateAttr

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun


logger = logging.getLogger(__name__)


# The AIRS sync scan API caps request bodies at 2MB. We chunk at 1.5MB to
# leave headroom for JSON envelope, metadata, and profile fields.
AIRS_SYNC_CHUNK_BYTES = 1_500_000


class PrismaAIRSBlocked(Exception):
    """Raised when AIRS decides to block a prompt or response."""

    def __init__(self, direction: str, scan_response: Any):
        self.direction = direction
        self.scan_response = scan_response
        self.action = getattr(scan_response, "action", None)
        self.category = getattr(scan_response, "category", None)
        self.scan_id = getattr(scan_response, "scan_id", None)
        self.report_id = getattr(scan_response, "report_id", None)
        self.detected = getattr(
            scan_response,
            "prompt_detected" if direction == "prompt" else "response_detected",
            None,
        )
        super().__init__(
            f"AIRS blocked {direction}: "
            f"action={self.action} category={self.category} "
            f"scan_id={self.scan_id} report_id={self.report_id}"
        )


@dataclass
class _ScanOutcome:
    """Result of scanning one piece of content.

    `masked_text` is set when AIRS returned masked data and the caller should
    substitute it. `None` means no substitution is needed. `scan_response` is
    the raw AIRS response, retained so callers that can't substitute (e.g.
    structured tool_calls) can raise with full context.
    """
    masked_text: Optional[str] = None
    scan_response: Any = None


class AIRSGuardedChatModel(BaseChatModel):
    """Wraps any LangChain chat model with Prisma AIRS scanning on input and output.

    Behavior:
    * Scans the flattened conversation (content + any tool_calls in history)
      as a `prompt` before calling the inner LLM.
    * Scans each generated message after: `content` with mask-substitution
      semantics, `tool_calls` separately with mask-as-block — masked text
      can't be substituted back into a structured dict, so a flagged tool
      arg blocks rather than silently leaking through the unmodified field.
    * If AIRS returns action=="block", raises PrismaAIRSBlocked, *unless* AIRS
      also returned masked data on a substitutable field — in which case the
      masked text is substituted and the request continues. (DLP "Mask
      sensitive data" mode emits action="block" plus masked_data; the
      substitution is the whole point of that mode.)
    * If a payload exceeds the sync API's 2MB cap, the text is chunked and
      each chunk scanned independently. Any chunk that blocks (without mask)
      blocks the whole request. Masked chunks are reassembled in order.
    * If the AIRS SDK itself errors, behavior depends on `fail_closed`.
    """

    model_config = {"arbitrary_types_allowed": True}

    inner_llm: BaseChatModel
    profile_name: Optional[str] = None
    profile_id: Optional[str] = None
    # Optional overrides used only on direction="response" scans. When unset,
    # response scans use the base profile. Useful when the response side
    # needs a stricter profile than the prompt side (or vice versa).
    response_profile_name: Optional[str] = None
    response_profile_id: Optional[str] = None
    app_user: str = "unknown"
    app_name: str = "langchain-app"
    fail_closed: bool = True  # if AIRS errors, block (True) or pass through (False)

    # Set in __init__; PrivateAttr keeps Pydantic from treating these as fields.
    _scanner: Optional[Scanner] = PrivateAttr(default=None)
    _ai_profile: Optional[AiProfile] = PrivateAttr(default=None)
    _response_ai_profile: Optional[AiProfile] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not (self.profile_name or self.profile_id):
            raise ValueError(
                "AIRSGuardedChatModel requires either profile_name or profile_id"
            )

        # Safe to call repeatedly; intended once at app startup.
        aisecurity.init()
        self._scanner = Scanner()
        self._ai_profile = (
            AiProfile(profile_id=self.profile_id)
            if self.profile_id
            else AiProfile(profile_name=self.profile_name)
        )
        # Response profile defaults to the base profile when no override is
        # provided. Same id-wins-over-name preference as the base.
        if self.response_profile_id:
            self._response_ai_profile = AiProfile(
                profile_id=self.response_profile_id
            )
        elif self.response_profile_name:
            self._response_ai_profile = AiProfile(
                profile_name=self.response_profile_name
            )
        else:
            self._response_ai_profile = self._ai_profile

    @property
    def _llm_type(self) -> str:
        return f"airs-guarded-{self.inner_llm._llm_type}"

    def bind_tools(self, tools, **kwargs):
        """Bind tools through the wrapper.

        The inner LLM does the provider-specific formatting (OpenAI uses
        one tool schema, Anthropic another). We extract the resulting
        bound kwargs and rebind them on `self` so AIRS scanning still
        wraps both prompt and response. `tools=` then flows through
        `_generate`'s **kwargs to `inner_llm._generate` at invoke time.

        Naively re-wrapping with the inner binding as `inner_llm` doesn't
        work: bind_tools returns a RunnableBinding, which isn't a
        BaseChatModel and doesn't expose `_generate`.
        """
        inner_bound = self.inner_llm.bind_tools(tools, **kwargs)
        return self.bind(**inner_bound.kwargs)

    # ---- text helpers ----

    @staticmethod
    def _messages_to_text(messages: List[BaseMessage]) -> str:
        """Flatten a LangChain message list into a single string for scanning.

        Scans the entire history each turn so injection attempts buried in
        earlier turns are caught. Tool calls in prior AIMessages are
        stringified alongside content — tool-arg injection is a real attack
        vector, and a history assembled outside this wrapper wouldn't have
        had them scanned at generation time. Multimodal content
        (list-of-dicts) will stringify via repr; special-case if needed.
        """
        parts: List[str] = []
        for m in messages:
            parts.append(f"{m.type}: {m.content}")
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                parts.append(f"{m.type} tool_calls: {tool_calls}")
        return "\n".join(parts)

    @staticmethod
    def _chunk_text(text: str, max_bytes: int = AIRS_SYNC_CHUNK_BYTES) -> List[str]:
        """Split text into chunks that fit under the AIRS sync payload limit.

        Splits on UTF-8 byte length so multi-byte text (CJK, emoji) is
        handled correctly. Prefers newline boundaries, falling back to a
        codepoint-aware byte cut. Returns the input unchanged in a
        single-element list if it already fits.
        """
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return [text]

        chunks: List[str] = []
        remaining = encoded
        while len(remaining) > max_bytes:
            window = remaining[:max_bytes]
            split_at = window.rfind(b"\n")
            if split_at < max_bytes // 2:
                # No usable newline; back off any partial UTF-8 codepoint.
                split_at = max_bytes
                while split_at > 0 and (remaining[split_at] & 0xC0) == 0x80:
                    split_at -= 1
                # Defensive: only reachable on invalid UTF-8 (a Python str
                # encode can't produce this). Cut at max_bytes anyway to
                # guarantee forward progress; decode(errors="replace")
                # handles any partial codepoint.
                if split_at == 0:
                    split_at = max_bytes
            chunks.append(remaining[:split_at].decode("utf-8", errors="replace"))
            remaining = remaining[split_at:]
        if remaining:
            chunks.append(remaining.decode("utf-8", errors="replace"))
        # Contract: every byte of the input ends up in exactly one chunk.
        # Boundary bytes (e.g. the '\n' at split_at) live at the start of the
        # next chunk via remaining[split_at:], never consumed. _scan_text
        # depends on this so "".join(chunks) reproduces the original input;
        # if you change this splitter to drop a boundary byte, _scan_text's
        # masked reassembly will silently corrupt the unmasked spans.
        return chunks

    def _build_metadata(self) -> dict:
        """Build per-scan metadata for the AIRS console.

        The SDK accepts a plain dict (per pan.dev examples), avoiding a
        dependency on internal generated-model imports.
        """
        return {
            "app_user": self.app_user,
            "app_name": self.app_name,
            "ai_model": getattr(self.inner_llm, "_llm_type", "unknown"),
        }

    # ---- scan plumbing ----

    @staticmethod
    def _extract_masked(scan_response: Any, direction: str) -> Optional[str]:
        """Pull masked text out of a scan response, if present.

        AIRS returns masked content as `prompt_masked_data.data` or
        `response_masked_data.data`. When DLP is set to Block + Mask, you
        get action="block" *and* masked_data; the masked version is the
        intended substitute, not a block signal.
        """
        field = "prompt_masked_data" if direction == "prompt" else "response_masked_data"
        masked_obj = getattr(scan_response, field, None)
        if masked_obj is None:
            return None
        # Some SDK versions return a plain dict instead of the model object.
        if isinstance(masked_obj, dict):
            return masked_obj.get("data")
        return getattr(masked_obj, "data", None)

    def _scan_chunk(self, text: str, direction: str) -> _ScanOutcome:
        """Scan one chunk and decide what to do with the result.

        Returns _ScanOutcome with the masked replacement if AIRS returned
        one. Raises PrismaAIRSBlocked if AIRS blocks without offering masked
        data. Re-raises AISecSDKException only when fail_closed; otherwise
        logs the error and returns an empty outcome (request continues
        unscanned for that chunk).

        Picks `_response_ai_profile` for response scans and `_ai_profile`
        for prompt scans — they're the same object unless a response_profile
        override was set on the constructor.
        """
        profile = (
            self._response_ai_profile
            if direction == "response"
            else self._ai_profile
        )
        content = (
            Content(prompt=text) if direction == "prompt" else Content(response=text)
        )
        try:
            scan_response = self._scanner.sync_scan(
                ai_profile=profile,
                content=content,
                metadata=self._build_metadata(),
            )
        except AISecSDKException:
            # Infrastructure error (network, auth, rate limit, oversize).
            if self.fail_closed:
                raise
            # Fail-open: log so this isn't silent. Don't log `text` itself —
            # it can contain user PII or the very payload we failed to scan.
            logger.exception(
                "AIRS SDK error; passing chunk through unscanned "
                "(fail_closed=False)",
                extra={
                    "direction": direction,
                    "chunk_bytes": len(text.encode("utf-8")),
                    "app_user": self.app_user,
                    "app_name": self.app_name,
                },
            )
            return _ScanOutcome()

        action = getattr(scan_response, "action", None)
        masked = self._extract_masked(scan_response, direction)

        if action == "block":
            if masked is not None:
                # Block + masked_data is DLP Mask mode; substitute, don't raise.
                return _ScanOutcome(masked_text=masked, scan_response=scan_response)
            raise PrismaAIRSBlocked(direction, scan_response)

        # Non-blocking action; some profiles still return masked data on allow.
        return _ScanOutcome(masked_text=masked, scan_response=scan_response)

    def _scan_text(
        self, text: str, direction: str, mask_as_block: bool = False
    ) -> str:
        """Scan text (chunking as needed) and return the possibly-masked result.

        A blocking chunk propagates PrismaAIRSBlocked. Masked chunks are
        substituted; unmasked chunks pass through. Returns the input
        unchanged if nothing was masked.

        If `mask_as_block` is True, a mask verdict is treated as a block.
        Use this when the caller can't substitute masked text (e.g. when
        scanning a stringified structured field like tool_calls) — silent
        substitution there would leak the original value through the
        unmodified structured field.
        """
        chunks = self._chunk_text(text)
        out: List[str] = []
        any_masked = False
        for chunk in chunks:
            outcome = self._scan_chunk(chunk, direction)
            if outcome.masked_text is not None:
                if mask_as_block:
                    raise PrismaAIRSBlocked(direction, outcome.scan_response)
                out.append(outcome.masked_text)
                any_masked = True
            else:
                out.append(chunk)
        # Returning the original `text` when nothing was masked preserves any
        # invalid UTF-8 byte sequences that decode(errors="replace") would
        # otherwise mangle into U+FFFD. When masking, we accept that loss —
        # the masked chunks themselves are decoded strings from the SDK. The
        # join relies on _chunk_text's contract that every input byte ends
        # up in some chunk.
        return "".join(out) if any_masked else text

    # ---- main path ----

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        prompt_text = self._messages_to_text(messages)
        scanned_prompt = self._scan_text(prompt_text, direction="prompt")

        if scanned_prompt != prompt_text:
            # Masked content collapses to one HumanMessage; we lose role
            # structure but guarantee the LLM never sees unmasked data.
            # For per-message masking, scan messages individually instead.
            logger.info(
                "AIRS masked prompt before LLM call",
                extra={"app_user": self.app_user, "app_name": self.app_name},
            )
            messages = [HumanMessage(content=scanned_prompt)]

        result = self.inner_llm._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

        for gen in result.generations:
            if not isinstance(gen, ChatGeneration):
                continue
            msg = gen.message
            # Plain-string content gets mask-substitution semantics.
            # Multimodal (list-of-parts) content is skipped — substitution
            # isn't meaningful and the block path still fires.
            if isinstance(msg.content, str) and msg.content:
                scanned = self._scan_text(msg.content, direction="response")
                if scanned != msg.content:
                    logger.info(
                        "AIRS masked response content",
                        extra={
                            "app_user": self.app_user,
                            "app_name": self.app_name,
                        },
                    )
                    msg.content = scanned
            # Tool-argument injection is a real attack vector, so scan
            # tool_calls too. Masked text can't be substituted back into
            # the structured tool_calls dict, so a mask verdict here is
            # treated as a block — otherwise the original sensitive value
            # would still leak through the unmodified tool_calls field.
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                self._scan_text(
                    str(tool_calls), direction="response", mask_as_block=True
                )

        return result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """Buffer the full response, scan it, then yield as one chunk.

        True streaming would bypass post-hoc scanning. The caller still gets
        a working .stream() interface, just not token-by-token output.
        """
        result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        for gen in result.generations:
            if isinstance(gen, ChatGeneration):
                # ChatGenerationChunk requires AIMessageChunk, not AIMessage.
                msg = gen.message
                chunk_msg = AIMessageChunk(
                    content=msg.content,
                    additional_kwargs=getattr(msg, "additional_kwargs", {}),
                    response_metadata=getattr(msg, "response_metadata", {}),
                    tool_calls=getattr(msg, "tool_calls", []),
                    id=getattr(msg, "id", None),
                )
                yield ChatGenerationChunk(
                    message=chunk_msg,
                    generation_info=gen.generation_info,
                )