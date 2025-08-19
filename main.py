import datetime
import textwrap
from dataclasses import dataclass
from collections import deque
from typing import Deque, Set
import pytchat


MAX_WIDTH = 100  # max characters per printed line


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
      - Ignores duplicates by message ID in O(1) via a set
      - Prints the entire buffer after each accepted message
      - Word-wraps each printed message at MAX_WIDTH without splitting words when possible
    """
    def __init__(self, max_messages: int = 10) -> None:
        self.max_messages: int = max_messages
        self._buffer: Deque[ChatMessage] = deque()
        self._ids: Set[str] = set()

    def add_message(self, msg_id: str, timestamp_ms: int, author: str, text: str) -> bool:
        """Add a message if not already present. Returns True if added, False if duplicate."""
        if msg_id in self._ids:
            return False  # duplicate → ignore

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
        Wrap a string to MAX_WIDTH, preferring not to split words.
        - Words longer than MAX_WIDTH will be split (to keep the hard max width).
        - Hyphen-based breaks are disabled to avoid mid-hyphen splits.
        """
        return textwrap.wrap(
            s,
            width=MAX_WIDTH,
            expand_tabs=False,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,     # only splits if a single word exceeds MAX_WIDTH
            break_on_hyphens=False,
        )

    def print_buffer(self) -> None:
        sep = "-" * 50
        out_lines: list[str] = []
        for m in self._buffer:
            out_lines.append(sep)
            for line in self._wrap(m.formatted_header()):
                out_lines.append(line)
        print("\n".join(out_lines))
        print("\n\n")  # extra newlines after the entire chatbuffer print


if __name__ == "__main__":
    video_id = "vxNNDRzVMmg"
    chat = pytchat.create(video_id)

    buffer = ChatBuffer(max_messages=10)

    while chat.is_alive():
        items = chat.get().items
        for item in items:
            buffer.add_message(item.id, item.timestamp, item.author.name, item.message)
