"""Shared input/framing and deterministic allocation primitives for NES's seven-method study.

This module deliberately does not implement any embedding algorithm. It provides
one common multi-message wire format and allocation contract so method-specific
adapters can be compared without silently changing their native carrier logic.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

MAGIC = b"NES7"
PROTOCOL_VERSION = 1
# magic, version, record count
_CORPUS_HEADER = struct.Struct(">4sBH")
# id byte length, payload byte length
_RECORD_HEADER = struct.Struct(">HI")
_DIGEST_SIZE = hashlib.sha256().digest_size

METHODS = (
    "sign",
    "magnitude_aware",
    "qae",
    "lwe_grid_parity",
    "split_sign_parity",
    "qse",
    "dce",
)

# Keep provenance/representation differences visible in every report.
METHOD_METADATA = {
    "sign": {"family": "residual_stream", "receiver": "matching sign extractor"},
    "magnitude_aware": {"family": "residual_stream", "receiver": "matching magnitude-aware extractor"},
    "qae": {"family": "residual_stream", "receiver": "QAE adapter/extractor; not nf4_qae"},
    "lwe_grid_parity": {"family": "residual_stream", "receiver": "LWE-inspired parity/grid extractor"},
    "split_sign_parity": {"family": "residual_stream", "receiver": "split extractor; parity carriers then sign carriers"},
    "qse": {"family": "packed_nf4_candidate", "receiver": "candidate-specific; artifact-only status must be recorded"},
    "dce": {"family": "packed_nf4_candidate", "receiver": "packed NF4 code indices and keyed carrier positions"},
}


@dataclass(frozen=True)
class MessageRecord:
    """A named UTF-8 string to be carried as part of a corpus."""

    id: str
    text: str


@dataclass(frozen=True)
class BitAllocation:
    """A contiguous slice of the framed bitstream assigned to one tensor."""

    tensor_name: str
    bit_offset: int
    bits: tuple[int, ...]

    @property
    def bit_length(self) -> int:
        return len(self.bits)


def normalize_records(records: Iterable[MessageRecord | Mapping[str, object]]) -> list[MessageRecord]:
    """Normalize records, enforce unique non-empty IDs, and validate UTF-8 encodability."""
    out: list[MessageRecord] = []
    seen: set[str] = set()
    for index, item in enumerate(records):
        if isinstance(item, MessageRecord):
            record = item
        else:
            if "text" not in item:
                raise ValueError(f"message record {index} has no 'text' field")
            record_id = item.get("id", f"msg-{index + 1:04d}")
            if not isinstance(record_id, str) or not isinstance(item["text"], str):
                raise TypeError(f"message record {index}: id and text must be strings")
            record = MessageRecord(record_id, item["text"])
        if not record.id:
            raise ValueError(f"message record {index} has an empty ID")
        if record.id in seen:
            raise ValueError(f"duplicate message ID: {record.id!r}")
        # Force strict UTF-8 validation now rather than midway through an experiment.
        record.id.encode("utf-8", errors="strict")
        record.text.encode("utf-8", errors="strict")
        seen.add(record.id)
        out.append(record)
    if len(out) > 65535:
        raise ValueError("at most 65535 messages are supported by protocol v1")
    return out


def load_jsonl(path: str | Path) -> list[MessageRecord]:
    """Load a UTF-8 JSONL corpus; blank lines are ignored, malformed lines fail loudly."""
    source = Path(path)
    records: list[MessageRecord] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{source}:{line_number}: each JSONL row must be an object")
            records.append(item)
    return normalize_records(records)


def encode_corpus(records: Iterable[MessageRecord | Mapping[str, object]]) -> bytes:
    """Encode an ordered corpus with per-record SHA-256 integrity digests.

    Wire format: corpus header, then for each record its lengths, UTF-8 ID,
    UTF-8 text, and SHA-256(text). Empty texts are valid. IDs must be unique.
    """
    rows = normalize_records(records)
    chunks = [_CORPUS_HEADER.pack(MAGIC, PROTOCOL_VERSION, len(rows))]
    for record in rows:
        rid = record.id.encode("utf-8")
        payload = record.text.encode("utf-8")
        if len(rid) > 65535:
            raise ValueError(f"message ID too long: {record.id!r}")
        if len(payload) > 0xFFFFFFFF:
            raise ValueError(f"message payload too long: {record.id!r}")
        chunks.append(_RECORD_HEADER.pack(len(rid), len(payload)))
        chunks.extend((rid, payload, hashlib.sha256(payload).digest()))
    return b"".join(chunks)


def decode_corpus(blob: bytes) -> list[MessageRecord]:
    """Decode and validate a corpus. Raises ValueError on truncation/corruption."""
    view = memoryview(blob)
    if len(view) < _CORPUS_HEADER.size:
        raise ValueError("corpus too short for header")
    magic, version, count = _CORPUS_HEADER.unpack(view[:_CORPUS_HEADER.size])
    if magic != MAGIC:
        raise ValueError("invalid corpus magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {version}")
    offset = _CORPUS_HEADER.size
    records: list[MessageRecord] = []
    for index in range(count):
        if len(view) - offset < _RECORD_HEADER.size:
            raise ValueError(f"truncated record header at index {index}")
        id_len, text_len = _RECORD_HEADER.unpack(view[offset:offset + _RECORD_HEADER.size])
        offset += _RECORD_HEADER.size
        expected = id_len + text_len + _DIGEST_SIZE
        if len(view) - offset < expected:
            raise ValueError(f"truncated record body at index {index}")
        id_bytes = bytes(view[offset:offset + id_len])
        offset += id_len
        text_bytes = bytes(view[offset:offset + text_len])
        offset += text_len
        digest = bytes(view[offset:offset + _DIGEST_SIZE])
        offset += _DIGEST_SIZE
        if not hmac.compare_digest(hashlib.sha256(text_bytes).digest(), digest):
            raise ValueError(f"integrity digest mismatch for record index {index}")
        try:
            record_id = id_bytes.decode("utf-8", errors="strict")
            text = text_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 in record index {index}") from exc
        records.append(MessageRecord(record_id, text))
    if offset != len(view):
        raise ValueError(f"unexpected trailing bytes: {len(view) - offset}")
    return normalize_records(records)


def bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to MSB-first bits, preserving all bytes exactly."""
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    """Convert a byte-aligned MSB-first bit sequence to bytes."""
    if len(bits) % 8:
        raise ValueError(f"bit count must be byte-aligned, got {len(bits)}")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")
    out = bytearray()
    for start in range(0, len(bits), 8):
        value = 0
        for bit in bits[start:start + 8]:
            value = (value << 1) | int(bit)
        out.append(value)
    return bytes(out)


def allocate_bits(bits: Sequence[int], tensor_capacities: Mapping[str, int]) -> list[BitAllocation]:
    """Split a stream sequentially across an explicitly ordered tensor-capacity map.

    Python mapping insertion order is the allocation order and must be saved in
    the experiment manifest. Capacity values count payload bits, not tensor
    elements. No capacity is silently discarded.
    """
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")
    if not tensor_capacities:
        if bits:
            raise ValueError("no tensor capacities supplied for non-empty payload")
        return []
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for name, capacity in tensor_capacities.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be non-empty strings")
        if name in seen:
            raise ValueError(f"duplicate tensor name: {name!r}")
        if not isinstance(capacity, int) or capacity < 0:
            raise ValueError(f"capacity for {name!r} must be a non-negative integer")
        seen.add(name)
        normalized.append((name, capacity))
    total_capacity = sum(cap for _, cap in normalized)
    if len(bits) > total_capacity:
        raise ValueError(f"payload needs {len(bits)} bits but tensor capacity is {total_capacity}")
    allocations: list[BitAllocation] = []
    offset = 0
    for name, capacity in normalized:
        if offset >= len(bits):
            break
        if capacity == 0:
            continue
        end = min(len(bits), offset + capacity)
        allocations.append(BitAllocation(name, offset, tuple(int(b) for b in bits[offset:end])))
        offset = end
    if offset != len(bits):
        raise AssertionError("internal allocation error: not all payload bits were allocated")
    return allocations


def keyed_positions(key: bytes, tensor_name: str, n_values: int, n_bits: int) -> list[int]:
    """Return deterministic, unique carrier positions domain-separated by tensor name."""
    if not isinstance(key, bytes) or not key:
        raise ValueError("key must be non-empty bytes")
    if not tensor_name:
        raise ValueError("tensor_name must be non-empty")
    if n_values < 0 or n_bits < 0:
        raise ValueError("n_values and n_bits must be non-negative")
    if n_bits > n_values:
        raise ValueError(f"requested {n_bits} carriers from only {n_values} values")
    out: list[int] = []
    seen: set[int] = set()
    counter = 0
    domain = b"NES-seven-method-v1\0" + tensor_name.encode("utf-8") + b"\0"
    while len(out) < n_bits:
        block = hmac.new(key, domain + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
        for start in range(0, len(block), 4):
            candidate = int.from_bytes(block[start:start + 4], "big") % n_values if n_values else 0
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
                if len(out) == n_bits:
                    break
    return out


def corpus_summary(records: Iterable[MessageRecord | Mapping[str, object]]) -> dict:
    """Return a stable, JSON-serializable manifest summary without message contents."""
    rows = normalize_records(records)
    blob = encode_corpus(rows)
    return {
        "protocol": "NES-seven-method-corpus-v1",
        "message_count": len(rows),
        "messages": [
            {
                "id": record.id,
                "utf8_bytes": len(record.text.encode("utf-8")),
                "payload_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
            }
            for record in rows
        ],
        "framed_bytes": len(blob),
        "framed_bits": len(blob) * 8,
        "corpus_sha256": hashlib.sha256(blob).hexdigest(),
    }
