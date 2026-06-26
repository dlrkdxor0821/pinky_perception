"""UDP framing for sending JPEG frames and receiving detection results.

A JPEG frame is usually larger than one UDP datagram (max 65507 bytes), so we
split it into chunks. Each datagram carries a header:

    frame_id (uint32) | total_chunks (uint16) | chunk_idx (uint16) | crc32 (uint32)

followed by the chunk payload. Two robustness guarantees (required for the
AI-server path):

  1. **Checksum** — every datagram carries a CRC32 of its payload. A datagram
     whose payload doesn't match its CRC is dropped (corruption / stray packet).
  2. **Drop stale frames** — the reassembler always works on the newest frame.
     When a chunk of a newer frame_id arrives, any older incomplete frames are
     discarded, and late packets from an older frame are ignored. This avoids
     mixing chunks across frames and keeps latency low (no stale backlog).
"""
import struct
import zlib

HEADER_FMT = "!IHHI"  # frame_id, total_chunks, chunk_idx, crc32(payload)
HEADER_SIZE = struct.calcsize(HEADER_FMT)
DEFAULT_CHUNK = 1400  # payload bytes per datagram — kept under the ~1500 Ethernet/
# WiFi MTU (minus IP/UDP/app headers) so each datagram fits in ONE IP packet and is
# NOT IP-fragmented. A big (e.g. 60 KB) datagram is split into ~40 IP fragments by the
# kernel, and losing any single fragment silently drops the whole datagram before the
# app's per-chunk CRC can help. MTU-sized chunks make that CRC + reassembly meaningful
# and behave predictably on lossy WiFi.

RESULT_FMT = "!I"  # frame_id prefix on the detection reply
RESULT_SIZE = struct.calcsize(RESULT_FMT)


def encode_frame(frame_id, payload, chunk_size=DEFAULT_CHUNK):
    """Split `payload` (JPEG bytes) into datagrams ready to sendto()."""
    chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)] or [b""]
    total = len(chunks)
    fid = frame_id & 0xFFFFFFFF
    out = []
    for idx, chunk in enumerate(chunks):
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        out.append(struct.pack(HEADER_FMT, fid, total, idx, crc) + chunk)
    return out


class Reassembler:
    """Collects chunks until a frame is complete. Call push() per datagram.

    Always tracks the newest frame_id seen; older incomplete frames are dropped
    and late packets from older frames are ignored.
    """

    def __init__(self):
        self._buf = {}
        self._latest = -1

    def push(self, datagram):
        """Return (frame_id, payload) once the newest frame completes, else None.

        Returns None (and drops the datagram) on CRC mismatch or stale packet.
        """
        if len(datagram) < HEADER_SIZE:
            return None
        frame_id, total, idx, crc = struct.unpack(HEADER_FMT, datagram[:HEADER_SIZE])
        payload = datagram[HEADER_SIZE:]

        # 1) checksum: reject corrupted / mismatched payloads
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            return None

        # 2) drop stale frames
        if frame_id < self._latest:
            return None  # late packet from an already-superseded frame
        if frame_id > self._latest:
            self._latest = frame_id
            for old in [k for k in self._buf if k < frame_id]:
                del self._buf[old]

        slot = self._buf.setdefault(frame_id, {})
        slot[idx] = payload
        if len(slot) >= total:
            data = b"".join(slot[i] for i in range(total) if i in slot)
            del self._buf[frame_id]
            return frame_id, data
        return None


def encode_result(frame_id, json_bytes):
    return struct.pack(RESULT_FMT, frame_id & 0xFFFFFFFF) + json_bytes


def decode_result(datagram):
    frame_id = struct.unpack(RESULT_FMT, datagram[:RESULT_SIZE])[0]
    return frame_id, datagram[RESULT_SIZE:]
