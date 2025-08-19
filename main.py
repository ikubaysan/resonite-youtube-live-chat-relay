"""
Sample usage:

  python chat_buffer.py VXNNDRzVMmg
  python chat_buffer.py VXNNDRzVMmg --max-messages 15
  python chat_buffer.py VXNNDRzVMmg --max-message-width 120
  python chat_buffer.py VXNNDRzVMmg --max-messages 20 --max-message-width 80
"""

import argparse
import datetime
import textwrap
from dataclasses import dataclass
from collections import deque
from typing import Deque, Set
import pytchat


@dataclass(frozen=True)
class ChatMessage:
    msg_id: str
    timestamp_ms: int  # unix time in ms
    author: str
    text: str

    def formatted_header(self) -> str:
        dt = datetime.datetime.fromtimestamp(self.timestamp_ms / 1000.0)
        return f"{dt:%Y-%m-%d %H:%M:%S} - {self.author}: {self.text}"


class ChatBuffer:
    """
    Rolling chat buffer that:
      - Holds up to `max_messages`
      - Ignores duplicates by message ID in O(1) via a set (duplicates only within current buffer)
      - Prints the entire buffer after each accepted message
      - Word-wraps each printed message at `max_message_width` without splitting words when possible
      - Surrounds the buffer with separators at top and bottom
    """
    def __init__(self, max_messages: int = 10, max_message_width: int = 100) -> None:
        # Basic safety clamps
        self.max_messages: int = max(1, int(max_messages))
        self.max_message_width: int = max(10, int(max_message_width))

        self._buffer: Deque[ChatMessage] = deque()
        self._ids: Set[str] = set()

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

        self.print_buffer()
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

    def print_buffer(self) -> None:
        sep = "-" * self.max_message_width  # frame width follows message width
        out_lines: list[str] = [sep]  # top border
        for idx, m in enumerate(self._buffer):
            if idx > 0:  # separator only between messages
                out_lines.append(sep)
            out_lines.extend(self._wrap(m.formatted_header()))
        out_lines.append(sep)  # bottom border
        print("\n".join(out_lines))
        print()  # extra newline after the entire chatbuffer print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a rolling, wrapped YouTube Live Chat buffer."
    )
    parser.add_argument(
        "--video-id",
        required=True,
        help="YouTube video ID (required)."
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=10,
        help="Maximum number of chat messages to keep in the buffer (default: 10)."
    )
    parser.add_argument(
        "--max-message-width",
        type=int,
        default=100,
        help="Maximum characters per printed line (default: 100)."
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    chat = pytchat.create(video_id=args.video_id)
    buffer = ChatBuffer(max_messages=args.max_messages, max_message_width=args.max_message_width)

    try:
        while chat.is_alive():
            items = chat.get().items
            for item in items:
                buffer.add_message(item.id, item.timestamp, item.author.name, item.message)
    except KeyboardInterrupt:
        pass
