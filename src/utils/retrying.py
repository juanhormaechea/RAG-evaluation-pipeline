"""Shared retry policy for transient LLM/embedding gateway failures.

Every LLM and embedding call in this project goes through the litellm gateway,
which occasionally returns transient errors (e.g. 503 "failure to get a peer
from the ring-balancer" when its upstream ring momentarily has no healthy peer,
429 rate limits, or dropped connections). These are safe to retry; a client
bug (400/401/404) or a programming error is not.

`api_retry` rides out a multi-minute gateway blip with jittered exponential
backoff, retrying ONLY the transient API error classes so real errors surface
fast instead of burning the full backoff. `reraise=True` surfaces the underlying
openai exception on exhaustion rather than wrapping it in tenacity's RetryError.
"""
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

# --- Gateway (litellm) transient errors ---
# RateLimitError, InternalServerError -> APIStatusError; APITimeoutError ->
# APIConnectionError. A gateway 503 surfaces as openai.InternalServerError.
_TRANSIENT: list[type[BaseException]] = [
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
]

# --- Graph-backend transient errors ---
# The Spanner (google.api_core) and LightRAG/Neo4j retrieve paths don't go
# through the gateway, so their transient outages need their own error classes
# or a blip there would crash a long benchmark sweep. Imported defensively so a
# backend that isn't installed can't break this module.
try:
    from google.api_core import exceptions as _g
    _TRANSIENT += [_g.ServiceUnavailable, _g.DeadlineExceeded, _g.Aborted,
                   _g.InternalServerError, _g.TooManyRequests, _g.GatewayTimeout]
except ImportError:  # pragma: no cover
    pass
try:
    import neo4j.exceptions as _n
    _TRANSIENT += [_n.ServiceUnavailable, _n.TransientError, _n.SessionExpired]
except ImportError:  # pragma: no cover
    pass

TRANSIENT_API_ERRORS = tuple(_TRANSIENT)

# ~8 jittered attempts capped at 60s each => rides out a multi-minute outage.
api_retry = retry(
    retry=retry_if_exception_type(TRANSIENT_API_ERRORS),
    wait=wait_exponential_jitter(initial=2, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)
