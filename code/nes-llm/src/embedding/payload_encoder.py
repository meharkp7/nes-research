"""
Payload encoder — converts text/bytes ↔ bit sequences.

Adds a 32-bit length header so the extractor knows precisely how many
bits belong to the payload (avoids padding ambiguity).

Wire format (bits):
    [0:32]    length header  — number of payload bits as uint32
    [32:32+N] payload bits   — UTF-8 encoded message
"""

import struct


class PayloadEncoder:
    """
    Text / bytes ↔ bit-list conversion with a 32-bit length header.

    encode(text)  →  [length_header_bits ... payload_bits ...]
    decode(bits)  →  text

    The length header stores the number of payload bits (not bytes),
    so the decoder reads exactly that many bits after the header.
    """

    HEADER_BITS = 32   # uint32 length header

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def text_to_bits(text: str) -> list:
        """
        Convert text string to bit list WITH 32-bit length header.

        Returns:
            [header_bit_0 ... header_bit_31, payload_bit_0 ... payload_bit_N]
        """
        raw_bytes    = text.encode("utf-8")
        payload_bits = PayloadEncoder._bytes_to_bits(raw_bytes)
        header_bits  = PayloadEncoder._uint32_to_bits(len(payload_bits))
        return header_bits + payload_bits

    @staticmethod
    def bytes_to_bits(data: bytes) -> list:
        """
        Convert raw bytes to bit list WITH 32-bit length header.
        """
        payload_bits = PayloadEncoder._bytes_to_bits(data)
        header_bits  = PayloadEncoder._uint32_to_bits(len(payload_bits))
        return header_bits + payload_bits

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @staticmethod
    def bits_to_text(bits: list) -> str:
        """
        Decode bit list (with length header) back to text string.

        Reads the 32-bit header, then extracts exactly that many payload bits.
        """
        raw = PayloadEncoder._decode_bits(bits)
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def bits_to_bytes(bits: list) -> bytes:
        """
        Decode bit list (with length header) back to raw bytes.
        """
        return PayloadEncoder._decode_bits(bits)

    # ------------------------------------------------------------------
    # Low-level helpers (public for direct use in tests / crypto layer)
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_bits(bits: list) -> bytes:
        """Internal: strip header, extract payload, return bytes."""
        header = PayloadEncoder.HEADER_BITS
        if len(bits) < header:
            raise ValueError(f"Bit stream too short to contain header ({len(bits)} bits)")

        payload_len = PayloadEncoder._bits_to_uint32(bits[:header])
        payload_bits = bits[header: header + payload_len]

        if len(payload_bits) < payload_len:
            raise ValueError(
                f"Bit stream truncated: expected {payload_len} payload bits, "
                f"got {len(payload_bits)}"
            )
        return PayloadEncoder._bits_to_bytes(payload_bits)

    @staticmethod
    def _bytes_to_bits(data: bytes) -> list:
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_bytes(bits: list) -> bytes:
        remainder = len(bits) % 8
        if remainder:
            bits = list(bits) + [0] * (8 - remainder)
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            out.append(byte)
        return bytes(out)

    @staticmethod
    def _uint32_to_bits(value: int) -> list:
        packed = struct.pack(">I", value)   # big-endian uint32
        return PayloadEncoder._bytes_to_bits(packed)

    @staticmethod
    def _bits_to_uint32(bits: list) -> int:
        raw = PayloadEncoder._bits_to_bytes(bits[:32])
        return struct.unpack(">I", raw)[0]