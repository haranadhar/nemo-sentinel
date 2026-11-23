"""Deterministic decoding of 'encrypted' data found in challenges and tool results.

Black-box puzzles commonly hide text behind base64, hex, binary, ROT13 /
Caesar shifts, reversed strings, URL- or HTML-encoding, or Morse code. An LLM
often decodes these wrongly; ordinary Python decodes them exactly, for free,
and in microseconds.

Everything here is pure string manipulation. Nothing is ever executed.
Results are offered to the solver as *hints* with a plausibility score; the
solver decides what to trust.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import html
import re
import string
from dataclasses import dataclass
from urllib.parse import unquote

_COMMON_WORDS = {
    "the", "and", "is", "of", "to", "in", "key", "answer", "tool", "use", "code",
    "secret", "password", "flag", "call", "with", "for", "you", "this", "that",
    "next", "clue", "find", "value", "hidden", "name", "id", "submit", "word",
    "a", "an", "it", "be", "are", "on", "at", "by", "or", "from", "as", "not",
}

MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
}

_B64_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/_-]{12,}={0,2})(?![A-Za-z0-9+/=_-])")
_HEX_RE = re.compile(r"(?<![0-9A-Fa-f])((?:0x)?(?:[0-9A-Fa-f]{2}[\s:]?){6,})(?![0-9A-Fa-f])")
_BIN_RE = re.compile(r"((?:[01]{8}[\s]?){3,})")
_MORSE_RE = re.compile(r"((?:[.\-]{1,6}[ /]+){3,}[.\-]{1,6})")
_URLENC_RE = re.compile(r"((?:%[0-9A-Fa-f]{2}|[A-Za-z0-9._~-]){6,})")


@dataclass(frozen=True)
class Decoded:
    method: str
    source: str
    text: str
    score: float  # 0..1 plausibility that this is meaningful English/ASCII

    def as_hint(self) -> str:
        src = self.source if len(self.source) <= 60 else self.source[:57] + "..."
        return f"[decoded {self.method}] {src!r} -> {self.text!r}"


def plausibility(text: str) -> float:
    """Heuristic 0..1 score of how much ``text`` looks like readable content."""
    if not text:
        return 0.0
    printable = sum(ch in string.printable for ch in text) / len(text)
    if printable < 0.95:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    spaces = text.count(" ")
    words = re.findall(r"[a-zA-Z]+", text.lower())
    common = sum(1 for w in words if w in _COMMON_WORDS)
    score = 0.35 * printable
    score += 0.25 * min(1.0, letters / max(1, len(text)) * 1.4)
    score += 0.2 * (1.0 if spaces or len(words) <= 2 else 0.4)
    score += 0.2 * min(1.0, common / max(1, len(words)) * 3)
    # JSON / key=value payloads are meaningful even without English words.
    if text.strip().startswith(("{", "[")) or re.search(r"\w+\s*[:=]\s*\w+", text):
        score = max(score, 0.75)
    return round(min(1.0, score), 3)


def caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def atbash(text: str) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr(122 - (ord(ch) - 97)))
        elif "A" <= ch <= "Z":
            out.append(chr(90 - (ord(ch) - 65)))
        else:
            out.append(ch)
    return "".join(out)


def _try_b64(token: str) -> str | None:
    if len(token) < 12:
        return None
    candidate = token.replace("-", "+").replace("_", "/")
    candidate += "=" * (-len(candidate) % 4)
    try:
        raw = base64.b64decode(candidate, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _try_hex(token: str) -> str | None:
    cleaned = re.sub(r"[\s:]|0x", "", token)
    if len(cleaned) % 2 or len(cleaned) < 12:
        return None
    try:
        return bytes.fromhex(cleaned).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _try_binary(token: str) -> str | None:
    bits = re.findall(r"[01]{8}", token)
    if len(bits) < 3:
        return None
    try:
        return bytes(int(b, 2) for b in bits).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _try_morse(token: str) -> str | None:
    words = re.split(r"\s*/\s*|\s{3,}", token.strip())
    out_words = []
    for word in words:
        letters = []
        for sym in word.split():
            if sym not in MORSE:
                return None
            letters.append(MORSE[sym])
        out_words.append("".join(letters))
    text = " ".join(w for w in out_words if w)
    return text or None


def decode_candidates(text: str, *, min_score: float = 0.55, limit: int = 12) -> list[Decoded]:
    """Find encoded substrings in ``text`` and decode them deterministically."""
    if not text or not isinstance(text, str):
        return []
    results: list[Decoded] = []
    seen: set[tuple[str, str]] = set()

    def add(method: str, source: str, decoded: str | None, bonus: float = 0.0) -> None:
        if not decoded:
            return
        decoded = decoded.strip()
        if not decoded or decoded == source.strip():
            return
        key = (method, decoded)
        if key in seen:
            return
        score = min(1.0, plausibility(decoded) + bonus)
        if score >= min_score:
            seen.add(key)
            results.append(Decoded(method, source.strip(), decoded, score))

    for m in _B64_RE.finditer(text):
        token = m.group(1)
        # Skip plain words / identifiers that happen to be b64-alphabet.
        if token.isalpha() and token.islower():
            continue
        decoded = _try_b64(token)
        add("base64", token, decoded, 0.1)
        # Nested: base64 of base64 / of hex.
        if decoded:
            add("base64>base64", token, _try_b64(decoded.strip()), 0.1)
            add("base64>hex", token, _try_hex(decoded.strip()), 0.1)
            add("base64>rot13", token, codecs.decode(decoded, "rot_13"), 0.0)
            add("base64>reverse", token, decoded[::-1], 0.0)
    for m in _HEX_RE.finditer(text):
        add("hex", m.group(1), _try_hex(m.group(1)), 0.1)
    for m in _BIN_RE.finditer(text):
        add("binary", m.group(1), _try_binary(m.group(1)), 0.1)
    for m in _MORSE_RE.finditer(text):
        add("morse", m.group(1), _try_morse(m.group(1)), 0.1)
    if "%" in text:
        for m in _URLENC_RE.finditer(text):
            if "%" in m.group(1):
                add("url", m.group(1), unquote(m.group(1)), 0.1)
    if "&" in text and ";" in text:
        add("html", text, html.unescape(text), 0.0)

    # Whole-string classical ciphers, only when explicitly hinted at or the
    # text looks like gibberish letters (to avoid noise on normal prose).
    lowered = text.lower()
    hinted = any(k in lowered for k in ("rot", "caesar", "shift", "cipher", "encrypt", "atbash", "reverse"))
    letters_only = re.sub(r"[^A-Za-z ]", "", text)
    if hinted or (letters_only and plausibility(text) < 0.5):
        segments = _cipher_segments(text)
        for seg in segments:
            add("rot13", seg, codecs.decode(seg, "rot_13"))
            add("atbash", seg, atbash(seg))
            add("reverse", seg, seg[::-1])
            best_shift, best_text, best_score = 0, "", 0.0
            for shift in range(1, 26):
                if shift == 13:
                    continue
                cand = caesar(seg, shift)
                s = plausibility(cand)
                if s > best_score:
                    best_shift, best_text, best_score = shift, cand, s
            if best_text:
                add(f"caesar-{26 - best_shift}", seg, best_text)

    results.sort(key=lambda d: d.score, reverse=True)
    return results[:limit]


def _cipher_segments(text: str) -> list[str]:
    """Quoted strings, or 'ciphertext: XYZ' segments, else the whole text."""
    segs = re.findall(r"[\"'`]([^\"'`]{4,200})[\"'`]", text)
    segs += re.findall(r"(?i)(?:cipher(?:text)?|encrypted|encoded|message)\s*[:=]\s*([A-Za-z ,.!?]{4,200})", text)
    if not segs and len(text) <= 300:
        segs = [text]
    out, seen = [], set()
    for s in segs:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:6]


def decode_hints(texts: list[str], *, limit: int = 10) -> list[str]:
    """Decode hints across several texts, best first, as solver-ready lines."""
    found: list[Decoded] = []
    for t in texts:
        found.extend(decode_candidates(t))
    found.sort(key=lambda d: d.score, reverse=True)
    lines, seen = [], set()
    for d in found:
        if d.text in seen:
            continue
        seen.add(d.text)
        lines.append(d.as_hint())
        if len(lines) >= limit:
            break
    return lines
