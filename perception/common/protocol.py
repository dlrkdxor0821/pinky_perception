"""UDP framing for sending JPEG frames and receiving detection results.

A single JPEG frame is usually larger than one UDP datagram (max 65507 bytes),
so we split it into chunks. Each datagram carries a small header:

    frame_id (uint32) | total_chunks (uint16) | chunk_idx (uint16) | payload...

The receiver reassembles chunks by frame_id. Because UDP can drop/reorder
packets, an incomplete frame is simply discarded (that is part of what the
benchmark measures).
"""
import struct

HEADER_FMT = "!IHH"  # frame_id, total_chunks, chunk_idx
HEADER_SIZE = struct.calcsize(HEADER_FMT)
DEFAULT_CHUNK = 60000  # payload bytes per datagram (under UDP 65507 limit)

RESULT_FMT = "!I"  # frame_id prefix on the detection reply
RESULT_SIZE = struct.calcsize(RESULT_FMT)


def encode_frame(frame_id, payload, chunk_size=DEFAULT_CHUNK):
    """Split `payload` (JPEG bytes) into a list of datagrams ready to sendto()."""
    chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)] or [b""]
    total = len(chunks)
    fid = frame_id & 0xFFFFFFFF
    return [struct.pack(HEADER_FMT, fid, total, idx) + chunk for idx, chunk in enumerate(chunks)]


class Reassembler:
    """Collects chunks until a frame is complete. Call push() per datagram."""

    def __init__(self, max_pending=64):
        self._buf = {}
        self._max_pending = max_pending

    def push(self, datagram):
        """Return (frame_id, payload) once all chunks arrive, else None."""
        if len(datagram) < HEADER_SIZE:
            return None
        frame_id, total, idx = struct.unpack(HEADER_FMT, datagram[:HEADER_SIZE])
        slot = self._buf.setdefault(frame_id, {})
        slot[idx] = datagram[HEADER_SIZE:]
        if len(slot) >= total:
            data = b"".join(slot[i] for i in range(total) if i in slot)
            del self._buf[frame_id]
            return frame_id, data
        # bound memory: drop the oldest incomplete frame if we fall behind
        if len(self._buf) > self._max_pending:
            self._buf.pop(next(iter(self._buf)))
        return None


def encode_result(frame_id, json_bytes):
    return struct.pack(RESULT_FMT, frame_id & 0xFFFFFFFF) + json_bytes


def decode_result(datagram):
    frame_id = struct.unpack(RESULT_FMT, datagram[:RESULT_SIZE])[0]
    return frame_id, datagram[RESULT_SIZE:]
