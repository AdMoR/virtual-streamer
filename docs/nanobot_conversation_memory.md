# Nanobot Conversation Memory: Analysis & Twitch Adaptation Guide

> Reference codebase: `temp/nanobot/`
> Focus: How memory works across channels, using WhatsApp as reference, and how to adapt for Twitch.

---

## 1. Two-Layer Memory Architecture

Nanobot uses two complementary stores for conversation persistence:

### Layer 1 — Session Messages (short-term)
- **Format**: JSONL file, one JSON object per line
- **Path**: `{workspace}/sessions/{channel}_{chat_id}.jsonl`
- **Content**: Raw message exchange — role, content, timestamps, tool calls
- **Purpose**: Immediate LLM context window. Append-only (never modified in place, for prompt caching efficiency).

### Layer 2 — Long-term Memory (MEMORY.md + HISTORY.md)
- **MEMORY.md**: Human-readable markdown of facts extracted by an LLM consolidation step. Persists across `/new` resets.
- **HISTORY.md**: Append-only log of conversation summaries, each prefixed with `[YYYY-MM-DD HH:MM]`. Acts as a searchable audit trail.
- Both are injected into the system prompt for every new LLM call.

---

## 2. Session Identification

### Key Format

```
"{channel}:{chat_id}"
```

Where `chat_id` is determined by each channel's implementation. This is the critical fork:

| Channel   | `chat_id` value       | Effective scope          |
|-----------|-----------------------|--------------------------|
| WhatsApp  | Sender's phone/LID    | **Per-user** (DMs)       |
| Email     | Sender address        | **Per-user**             |
| Telegram  | Group chat ID         | **Per-group** (or per-topic with override) |
| Slack     | DM or channel ID      | **Per-DM** / **Per-thread** (if `reply_in_thread` enabled) |
| Discord   | Channel ID            | **Per-channel** (shared) |
| Twitch    | Channel name          | **Per-channel** (shared) |

### Why WhatsApp is Per-User

In `channels/whatsapp.py`, the `chat_id` passed to `_handle_message()` is the sender's unique identifier (phone or LID). Every user gets their own session file and their own `MEMORY.md`.

### Why Twitch is Shared

In `channels/twitch.py` (lines 75–80), `chat_id` is the channel name — the same for every viewer. All messages from all users land in a single session: `twitch:{channel_name}`.

### Thread-Scoped Override

Channels can override the session key entirely:

```python
# Slack threads (slack.py:183–197)
session_key = f"slack:{chat_id}:{thread_ts}"

# Telegram topics (telegram.py:445–450)
session_key = f"telegram:{chat_id}:topic:{message_thread_id}"
```

This is the mechanism to leverage for Twitch per-user memory.

---

## 3. Context Window Management

### Configuration

```yaml
# nanobot config YAML
agents:
  defaults:
    memory_window: 100   # default
```

Set in `config/schema.py:241` (`AgentDefaults.memory_window`).

### What Gets Sent to the LLM

Three parts are assembled in `agent/context.py`:

1. **System prompt** — identity, skills, + full contents of `MEMORY.md`
2. **History slice** — recent unconsolidated messages from the session, starting from `last_consolidated` index
3. **Current user message** — with injected runtime context (timestamp, channel, chat_id)

The history slice is retrieved via `session.get_history(max_messages=500)` (manager.py:46–64), which:
- Slices from `last_consolidated` to end
- Trims to last `max_messages`
- Drops any leading non-user messages (to avoid orphaned tool_result blocks)

### Tool Result Truncation

Tool outputs are capped at 500 characters before saving to the session (loop.py:47, 463–464). This prevents large file reads or API responses from bloating the context.

---

## 4. Memory Consolidation

### Trigger

When unconsolidated message count ≥ `memory_window` (loop.py:396–412), a background consolidation task fires.

```
unconsolidated = len(session.messages) - session.last_consolidated
if unconsolidated >= memory_window:
    # spawn background consolidation
```

### What Consolidation Does (memory.py:69–157)

1. Extracts old messages (everything from `last_consolidated` up to `-keep_count`)
   - `keep_count = memory_window // 2` → keeps the most recent 50 messages (at default window=100)
2. Sends them to an LLM with a `save_memory` tool
3. LLM returns:
   - `history_entry`: 2–5 sentence summary → appended to `HISTORY.md`
   - `memory_update`: Full updated `MEMORY.md` content
4. Updates `session.last_consolidated` pointer (messages list is never modified)

### Concurrency Safety

A per-session `asyncio.Lock` (stored in a `WeakValueDictionary`) ensures only one consolidation runs per session at a time (loop.py:365, 399).

---

## 5. The Twitch Problem

Twitch is a high-volume, multi-user channel. The current shared-channel session has three issues:

1. **Context stuffing**: 100 messages from 50 different users fill the window quickly with irrelevant exchanges.
2. **No per-user continuity**: If the AI asks user A a follow-up question, user B's messages in between will be in context; the AI may confuse who said what.
3. **No user identity in consolidation**: MEMORY.md consolidates the whole channel, not per-user facts.

---

## 6. Recommended Adaptation for Twitch

### Option A — Per-User Session (cleanest isolation)

Override the session key so each Twitch user gets their own session:

```python
# channels/twitch.py — in _handle_raw() or equivalent
await self._handle_message(
    sender_id=username,
    chat_id=channel_id,
    content=text,
    session_key=f"twitch:{channel_id}:{username}",  # per-user key
    metadata={"channel_name": channel_id},
)
```

**Pros**: Full isolation; user-specific MEMORY.md; small context windows per user.
**Cons**: Loses channel-wide awareness (the AI won't know what other users have discussed).

### Option B — Hybrid: Shared Channel Memory + User Tag in Messages

Keep the shared session but prefix every message with the sender's name:

```python
content = f"[{username}]: {text}"
```

Configure a low `memory_window` (e.g. 30–50) to prevent bloat. The consolidation LLM will then naturally extract per-user facts into MEMORY.md (e.g., "User spammy_cat prefers clip highlights").

**Pros**: AI retains channel context; still learns per-user preferences over time.
**Cons**: Context fills faster; AI must parse "[username]:" to track who asked what.

### Option C — Selective Per-User Sessions (best of both)

Use a shared session for general chat, but switch to a per-user session when the AI enters a back-and-forth dialogue (e.g., it asked the user a question):

- Track `pending_question_for: dict[username, bool]` in the channel handler.
- When `pending_question_for[username]` is set, route that user's next message to `twitch:{channel_id}:{username}` until the dialogue resolves.
- All other messages go to the shared `twitch:{channel_id}` session.

This is the most complex but most faithful to Twitch's UX (most messages are broadcast; only some need dialogue context).

---

## 7. Configuration Recommendations for Twitch

```yaml
agents:
  defaults:
    memory_window: 40        # Lower than default 100 — Twitch moves fast
    max_tokens: 4096         # Keep responses short for chat
    max_tool_iterations: 10  # Limit tool loops in live context
```

Consider also truncating tool results more aggressively (default 500 chars is reasonable).

---

## 8. Key Files Reference

| File | Purpose |
|------|---------|
| `nanobot/agent/memory.py:69–157` | Consolidation logic |
| `nanobot/agent/loop.py:396–412` | Consolidation trigger |
| `nanobot/session/manager.py:16–214` | Session load/save/cache |
| `nanobot/agent/context.py:85–130` | How context is assembled for LLM |
| `nanobot/bus/events.py:8–24` | `InboundMessage.session_key` property |
| `nanobot/channels/twitch.py:75–80` | Where to inject per-user session key |
| `nanobot/channels/slack.py:183–197` | Reference: thread-scoped session override |
| `nanobot/config/schema.py:230–243` | `AgentDefaults` including `memory_window` |