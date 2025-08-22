import argparse
import asyncio
import datetime
import textwrap
import time
import uuid
import contextlib
import sys
import os
import signal
from dataclasses import dataclass
from collections import deque
from typing import Deque, Set, Optional

import pytchat
import websockets


# -------------------- Hard-kill on Ctrl+C (Windows + POSIX) --------------------

def _install_hard_kill_handlers() -> None:
    """
    Guarantee immediate process termination on Ctrl+C (and friends).
    Windows: use SetConsoleCtrlHandler to catch CTRL_C_EVENT, CTRL_BREAK_EVENT, etc.
    POSIX:   force-exit on SIGINT.
    """
    def _hard_exit(reason: str) -> None:
        # Print once, then force exit (130 = terminated by Ctrl+C)
        try:
            print(f"\n[Exit] {reason}. Forcing immediate exit.", flush=True)
        except Exception:
            pass
        os._exit(130)

    # POSIX handler
    def _sigint_handler(signum, frame):
        _hard_exit("SIGINT received (Ctrl+C)")

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except Exception:
        pass

    # Windows console control handler
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            # Control event codes:
            CTRL_C_EVENT = 0
            CTRL_BREAK_EVENT = 1
            CTRL_CLOSE_EVENT = 2
            CTRL_LOGOFF_EVENT = 5
            CTRL_SHUTDOWN_EVENT = 6

            def _console_handler(ctrl_type):
                if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
                    _hard_exit(f"Console control event {ctrl_type}")
                    return True
                return False

            _handler_ref = HandlerRoutine(_console_handler)  # keep a ref so it isn't GC'd
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler_ref, True)
            # Stash the ref to prevent GC:
            globals()["_WIN_CONSOLE_HANDLER_REF"] = _handler_ref
        except Exception:
            # If this fails, we still have POSIX-style SIGINT handler above
            pass


# -------------------- Chat Message --------------------

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


# -------------------- Chat Buffer --------------------

class ChatBuffer:
    """
    Rolling chat buffer that:
      - Holds up to `max_messages`
      - Ignores duplicates by message ID (current buffer only)
      - Prints the entire buffer after each accepted message
      - Word-wraps each printed message at `max_message_width`
      - Surrounds the buffer with separators at top and bottom
      - Optionally trims the rendered buffer to `max_lines` lines (from the top)
      - Broadcasts the rendered buffer via websocket
    """
    def __init__(
        self,
        max_messages: int = 10,
        max_message_width: int = 100,
        max_lines: int = 0,
        sep_char: str = "─",
        sep_length: int = 100,
    ) -> None:
        self.max_messages: int = max(1, int(max_messages))
        self.max_message_width: int = max(10, int(max_message_width))
        self.max_lines: int = max(0, int(max_lines))  # 0 = disabled
        self.sep_char: str = sep_char
        self.sep_length: int = max(1, int(sep_length))

        self._buffer: Deque[ChatMessage] = deque()
        self._ids: Set[str] = set()

        # Websocket state (owned by main loop)
        self._clients: Set[websockets.WebSocketServerProtocol] = set()

    def add_ws_client(self, ws: websockets.WebSocketServerProtocol) -> None:
        self._clients.add(ws)

    def remove_ws_client(self, ws: websockets.WebSocketServerProtocol) -> None:
        self._clients.discard(ws)

    async def broadcast(self, text: str) -> None:
        """Broadcast to all clients; drop any that fail."""
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

    def add_message(self, msg_id: str, timestamp_ms: int, author: str, text: str) -> bool:
        """Add a message if not already present. Returns True if added, False if duplicate."""
        if msg_id in self._ids:
            return False

        if len(self._buffer) >= self.max_messages:
            oldest = self._buffer.popleft()
            self._ids.discard(oldest.msg_id)

        msg = ChatMessage(msg_id=msg_id, timestamp_ms=timestamp_ms, author=author, text=text)
        self._buffer.append(msg)
        self._ids.add(msg_id)
        return True

    def _wrap(self, s: str) -> list[str]:
        return textwrap.wrap(
            s,
            width=self.max_message_width,
            expand_tabs=False,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )

    def render_buffer(self, bold_usernames: bool = False) -> str:
        sep = self.sep_char * self.sep_length
        out_lines: list[str] = [sep]
        for idx, m in enumerate(self._buffer):
            if idx > 0:
                out_lines.append(sep)
            header = m.formatted_header_bold_author() if bold_usernames else m.formatted_header()
            out_lines.extend(self._wrap(header))
        out_lines.append(sep)

        if self.max_lines > 0 and len(out_lines) > self.max_lines:
            out_lines = out_lines[-self.max_lines:]

        return "\n".join(out_lines)

    async def print_and_broadcast(self) -> None:
        # Console (plain usernames)
        output_plain = self.render_buffer(bold_usernames=False)
        print(output_plain + "\n", flush=True)
        # Websocket (bold usernames)
        output_bold = self.render_buffer(bold_usernames=True)
        await self.broadcast(f"ChatBuffer::{output_bold}")


# -------------------- Helpers --------------------

async def interruptible_wait(total_seconds: float, tick: float = 0.1) -> None:
    """Async sleep in short chunks so cancellation is immediate."""
    end = time.monotonic() + max(0.0, total_seconds)
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(tick, remaining))


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and broadcast a rolling, wrapped YouTube Live Chat buffer."
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID (required).")
    parser.add_argument("--max-messages", type=int, default=20,
                        help="Maximum number of chat messages to keep in the buffer (default: 20).")
    parser.add_argument("--max-message-width", type=int, default=50,
                        help="Maximum characters per printed line (default: 50).")
    parser.add_argument("--max-lines", type=int, default=0,
                        help=("Maximum number of lines to show in the rendered buffer (after wrapping & separators). "
                              "If set to 0 (default), this feature is disabled."))
    parser.add_argument("--ws-host", default="127.0.0.1",
                        help="WebSocket host to bind (default: 127.0.0.1).")
    parser.add_argument("--ws-port", type=int, default=17865,
                        help="WebSocket port to bind (default: 17865).")
    parser.add_argument("--sep-char", default="─",
                        help='Separator character (default: "─"). Alternatives: "━", "═", "█", "-".')
    parser.add_argument("--sep-length", type=int, default=20,
                        help="Number of separator characters per line (default: 20).")
    parser.add_argument("--retry-wait", type=float, default=15.0,
                        help="Seconds to wait before retrying when chat is not alive (default: 15).")
    parser.add_argument("--stale-timeout", type=float, default=120.0,
                        help="If no chat items arrive for this many seconds, recreate the chat client (default: 120).")
    return parser.parse_args()


# -------------------- Main (single asyncio loop, no background event loop threads) --------------------

def _create_chat(video_id: str):
    # Wrap creation to keep a single place for logging and future options
    print(f"[Chat] Creating new pytchat client for video_id={video_id}", flush=True)
    return pytchat.create(video_id=video_id)

async def main_async(args: argparse.Namespace) -> None:
    buffer = ChatBuffer(
        max_messages=args.max_messages,
        max_message_width=args.max_message_width,
        max_lines=args.max_lines,
        sep_char=args.sep_char,
        sep_length=args.sep_length,
    )

    # Create pytchat client (recreated as needed)
    chat = _create_chat(args.video_id)
    last_item_ts = time.monotonic()

    async def ws_handler(ws: websockets.WebSocketServerProtocol):
        buffer.add_ws_client(ws)
        try:
            # Per-connection greeting
            await ws.send(f"Data::https://www.youtube.com/live/{args.video_id}")
            # Send current buffer immediately
            await buffer.broadcast(f"ChatBuffer::{buffer.render_buffer(bold_usernames=True)}")

            # Drain incoming messages; keep alive
            async for _ in ws:
                pass
        finally:
            buffer.remove_ws_client(ws)

    print(f"[WebSocket] Hosting on ws://{args.ws_host}:{args.ws_port}", flush=True)
    async with websockets.serve(ws_handler, args.ws_host, args.ws_port):
        try:
            while True:
                # 1) If pytchat reports dead, rebuild it.
                if not chat.is_alive():
                    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                          f"Chat stream at {args.video_id} reports NOT ALIVE. Recreating client after {args.retry_wait:.1f}s...",
                          flush=True)
                    with contextlib.suppress(Exception):
                        chat.terminate()
                    await interruptible_wait(args.retry_wait)
                    chat = _create_chat(args.video_id)
                    # After recreation, continue loop (don’t hammer get() immediately)
                    await asyncio.sleep(0.1)
                    continue

                # 2) Try reading items; if get() explodes, recreate client.
                try:
                    items = await asyncio.to_thread(lambda: chat.get().items)
                except Exception as e:
                    print(f"[Chat] get() error: {e}. Recreating client after {args.retry_wait:.1f}s", flush=True)
                    with contextlib.suppress(Exception):
                        chat.terminate()
                    await interruptible_wait(args.retry_wait)
                    chat = _create_chat(args.video_id)
                    await asyncio.sleep(0.1)
                    continue

                # 3) Process items
                updated = False
                if items:
                    last_item_ts = time.monotonic()
                for item in items:
                    added = buffer.add_message(item.id, item.timestamp, item.author.name, item.message)
                    if added:
                        updated = True

                if updated:
                    await buffer.print_and_broadcast()

                # 4) Stale watchdog: if no items for a while, rebuild the client even if is_alive() is True
                idle = time.monotonic() - last_item_ts
                if args.stale_timeout > 0 and idle >= args.stale_timeout:
                    print(f"[Chat] No items for {idle:.0f}s (>= {args.stale_timeout:.0f}). "
                          f"Assuming stale client; recreating after {args.retry_wait:.1f}s.", flush=True)
                    with contextlib.suppress(Exception):
                        chat.terminate()
                    await interruptible_wait(args.retry_wait)
                    chat = _create_chat(args.video_id)
                    last_item_ts = time.monotonic()
                    await asyncio.sleep(0.1)
                    continue

                # Small tick to keep loop responsive but not hot
                await asyncio.sleep(0.01)

        finally:
            with contextlib.suppress(Exception):
                chat.terminate()
            # Close all websocket clients
            close_tasks = []
            for ws in list(buffer._clients):
                close_tasks.append(asyncio.create_task(ws.close(code=1001, reason="Server shutting down")))
            if close_tasks:
                with contextlib.suppress(Exception):
                    await asyncio.gather(*close_tasks, return_exceptions=True)


if __name__ == "__main__":
    _install_hard_kill_handlers()
    args = parse_args()

    # Run everything on a single asyncio loop in the main thread.
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        # This block might not run on Windows if the console handler already called os._exit.
        # It's here for POSIX completeness.
        print("\n[Main] KeyboardInterrupt received. Exiting now.", flush=True)
        os._exit(130)
