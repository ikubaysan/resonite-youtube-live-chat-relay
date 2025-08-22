import argparse
import asyncio
import datetime
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass
from collections import deque
from typing import Deque, Set, Optional, Callable, Awaitable

import pytchat
import websockets


# -------------------- WebSocket Broadcaster --------------------

class WebSocketBroadcaster:
    """Runs a local WebSocket server in a background thread and broadcasts raw text to all clients."""
    def __init__(self, host: str = "127.0.0.1", port: int = 17865) -> None:
        self.host = host
        self.port = int(port)
        self._loop = asyncio.new_event_loop()
        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._started = threading.Event()
        self._server_task: Optional[asyncio.Task] = None
        # on_connect will be called with the websocket of the newly connected client.
        # It can be sync or async. Use it to send per-client messages before any broadcasts.
        self.on_connect: Optional[Callable[[websockets.WebSocketServerProtocol], Optional[Awaitable[None]]]] = None

    async def _handler(self, ws: websockets.WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        try:
            # If provided, allow a hook on connection (can be async or sync)
            if self.on_connect is not None:
                result = self.on_connect(ws)
                if asyncio.iscoroutine(result):
                    await result  # ensure per-client message is sent before anything else

            # Keep connection alive; we don't expect inbound messages
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)

    async def _serve(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port):
            self._started.set()
            # Run forever
            await asyncio.Future()

    def start(self) -> None:
        def runner():
            asyncio.set_event_loop(self._loop)
            self._server_task = self._loop.create_task(self._serve())
            self._loop.run_forever()

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        # Wait until server socket is bound
        self._started.wait()
        print(f"[WebSocket] Hosting on ws://{self.host}:{self.port}")

    async def _broadcast(self, text: str) -> None:
        if not self._clients:
            return
        stale = []
        for ws in list(self._clients):
            try:
                await ws.send(text)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)

    def broadcast(self, text: str) -> None:
        """Thread-safe schedule of a broadcast from any thread."""
        if self._loop.is_closed():
            return
        # Prefix all WS broadcasts
        payload = f"ChatBuffer::{text}"
        self._loop.call_soon_threadsafe(
            asyncio.create_task,
            self._broadcast(payload)
        )


# -------------------- Chat Buffer --------------------

@dataclass(frozen=True)
class ChatMessage:
    msg_id: str
    timestamp_ms: int  # unix time in ms
    author: str
    text: str

    def formatted_header(self) -> str:
        dt = datetime.datetime.fromtimestamp(self.timestamp_ms / 1000.0)
        return f"{dt:%H:%M:%S} - {self.author}: {self.text}"

    def formatted_header_bold_author(self) -> str:
        dt = datetime.datetime.fromtimestamp(self.timestamp_ms / 1000.0)
        return f"{dt:%H:%M:%S} - <b>{self.author}</b>: {self.text}"


class ChatBuffer:
    """
    Rolling chat buffer that:
      - Holds up to `max_messages`
      - Ignores duplicates by message ID in O(1) via a set (duplicates only within current buffer)
      - Prints the entire buffer after each accepted message
      - Word-wraps each printed message at `max_message_width` without splitting words when possible
      - Surrounds the buffer with separators at top and bottom
      - Optionally trims the rendered buffer to `max_lines` lines (from the top)
      - Optionally broadcasts the rendered buffer via a WebSocketBroadcaster
    """
    def __init__(
        self,
        max_messages: int = 10,
        max_message_width: int = 100,
        max_lines: int = 0,
        broadcaster: Optional[WebSocketBroadcaster] = None,
        sep_char: str = "─",
        sep_length: int = 100,
    ) -> None:
        # Basic safety clamps
        self.max_messages: int = max(1, int(max_messages))
        self.max_message_width: int = max(10, int(max_message_width))
        self.max_lines: int = max(0, int(max_lines))  # 0 = disabled
        self.sep_char: str = sep_char
        self.sep_length: int = max(1, int(sep_length))

        self._buffer: Deque[ChatMessage] = deque()
        self._ids: Set[str] = set()
        self._broadcaster = broadcaster

    def add_message(self, msg_id: str, timestamp_ms: int, author: str, text: str) -> bool:
        """Add a message if not already present. Returns True if added, False if duplicate."""
        if msg_id in self._ids:
            return False  # duplicate in current buffer → ignore

        # Remove oldest message if at capacity
        if len(self._buffer) >= self.max_messages:
            oldest = self._buffer.popleft()
            self._ids.discard(oldest.msg_id)

        msg = ChatMessage(msg_id=msg_id, timestamp_ms=timestamp_ms, author=author, text=text)
        self._buffer.append(msg)
        self._ids.add(msg_id)

        self.print_buffer()  # also triggers WS broadcast
        return True

    def _wrap(self, s: str) -> list[str]:
        """
        Wrap a string to `self.max_message_width`, preferring not to split words.
        - Words longer than `self.max_message_width` will be split (to keep the hard max width).
        - Hyphen-based breaks are disabled to avoid mid-hyphen splits.
        """
        return textwrap.wrap(
            s,
            width=self.max_message_width,
            expand_tabs=False,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,     # only splits if a single word exceeds max width
            break_on_hyphens=False,
        )

    def render_buffer(self, bold_usernames: bool = False) -> str:
        """Return the chat buffer as a formatted string (with separators, possibly trimmed)."""
        sep = self.sep_char * self.sep_length
        out_lines: list[str] = [sep]  # top border
        for idx, m in enumerate(self._buffer):
            if idx > 0:  # separator only between messages
                out_lines.append(sep)
            header = m.formatted_header_bold_author() if bold_usernames else m.formatted_header()
            out_lines.extend(self._wrap(header))
        out_lines.append(sep)  # bottom border

        # If max_lines is set, trim from the top
        if self.max_lines > 0 and len(out_lines) > self.max_lines:
            out_lines = out_lines[-self.max_lines:]

        return "\n".join(out_lines)

    def print_buffer(self) -> None:
        """Print the rendered chat buffer string and broadcast it if enabled."""
        # Console (plain usernames)
        output_plain = self.render_buffer(bold_usernames=False)
        print(output_plain + "\n")
        # Websocket (bold usernames)
        if self._broadcaster:
            output_bold = self.render_buffer(bold_usernames=True)
            self._broadcaster.broadcast(output_bold)


# -------------------- CLI & Main --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and broadcast a rolling, wrapped YouTube Live Chat buffer."
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID (required).")
    parser.add_argument("--max-messages", type=int, default=10,
                        help="Maximum number of chat messages to keep in the buffer (default: 10).")
    parser.add_argument("--max-message-width", type=int, default=100,
                        help="Maximum characters per printed line (default: 100).")
    parser.add_argument("--max-lines", type=int, default=0,
                        help=("Maximum number of lines to show in the rendered buffer (after wrapping & separators). "
                              "If set to 0 (default), this feature is disabled."))
    parser.add_argument("--ws-host", default="127.0.0.1",
                        help="WebSocket host to bind (default: 127.0.0.1).")
    parser.add_argument("--ws-port", type=int, default=17865,
                        help="WebSocket port to bind (default: 17865).")
    parser.add_argument("--sep-char", default="─",
                        help='Separator character (default: "─"). Alternatives: "━", "═", "█", "-".')
    parser.add_argument("--sep-length", type=int, default=100,
                        help="Number of separator characters per line (default: 100).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Prepare broadcaster (do NOT start yet)
    broadcaster = WebSocketBroadcaster(host=args.ws_host, port=args.ws_port)

    # Prepare buffer
    buffer = ChatBuffer(
        max_messages=args.max_messages,
        max_message_width=args.max_message_width,
        max_lines=args.max_lines,
        broadcaster=broadcaster,
        sep_char=args.sep_char,
        sep_length=args.sep_length,
    )

    # When a client connects:
    # 1) Send them "Data::<live url>" directly (only to that client)
    # 2) Then append a "Client Connected" system message to the buffer (broadcasts with ChatBuffer:: prefix)
    async def _on_connect(ws: websockets.WebSocketServerProtocol):
        await ws.send(f"Data::https://www.youtube.com/live/{args.video_id}")
        buffer.add_message(
            msg_id=f"sys-{uuid.uuid4().hex}",
            timestamp_ms=int(time.time() * 1000),
            author="System",
            text="Client Connected",
        )

    broadcaster.on_connect = _on_connect

    # Now start the WebSocket server
    broadcaster.start()

    chat = pytchat.create(video_id=args.video_id)

    try:
        while chat.is_alive():
            items = chat.get().items
            for item in items:
                buffer.add_message(item.id, item.timestamp, item.author.name, item.message)
    except KeyboardInterrupt:
        pass
