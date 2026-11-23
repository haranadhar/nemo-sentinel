"""Answer extraction, output-format detection, validation and normalisation.

Quality is 70% of the score and a correct answer in the wrong format can be
judged wrong, so every candidate answer passes through here before it is
saved as a draft or submitted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(r"^\s*(?:\*\*)?(?:final\s+answer|answer)(?:\*\*)?\s*[:：]\s*(.*)$", re.IGNORECASE)
_BLOCKED = ("<toolcall", "</toolcall", '"tool_calls"', "function_call", "<think")


@dataclass
class AnswerFormat:
    kind: str = "free"            # free | json | integer | number | choice | yesno | word | list
    case: str = ""                # upper | lower | ""
    choices: list[str] = field(default_factory=list)
    decimals: int | None = None
    separator: str = ""
    raw_hint: str = ""            # the rules sentence that defined the format

    def describe(self) -> str:
        parts = {
            "json": "a single valid JSON value (no prose, no code fences)",
            "integer": "only an integer, digits only (with sign if negative)",
            "number": "only a number" + (f" with exactly {self.decimals} decimal places" if self.decimals is not None else ""),
            "choice": f"exactly one of: {', '.join(self.choices)}",
            "yesno": "exactly 'yes' or 'no'",
            "word": "a single word",
            "list": f"a list separated by '{self.separator or ', '}'",
            "free": "the shortest exact answer that satisfies the rules",
        }
        text = parts.get(self.kind, parts["free"])
        if self.case == "upper":
            text += ", in UPPERCASE"
        elif self.case == "lower":
            text += ", in lowercase"
        return text


def detect_format(rules: str, description: str = "") -> AnswerFormat:
    text = f"{rules or ''}\n{description or ''}"
    low = text.lower()
    fmt = AnswerFormat()
    m = re.search(r"(?im)^.*\b(format|respond|answer|output|return)\b.*$", rules or "")
    fmt.raw_hint = m.group(0).strip()[:240] if m else ""

    if re.search(r"\bjson\b", low):
        fmt.kind = "json"
    elif re.search(r"\b(yes or no|yes/no|\"yes\" or \"no\")\b", low):
        fmt.kind = "yesno"
    elif re.search(r"\b(integer|whole number|digits only|number only|only the number|numeric answer|a number)\b", low):
        fmt.kind = "integer" if re.search(r"\b(integer|whole number|digits only)\b", low) else "number"
    elif re.search(r"\b(single word|one word)\b", low):
        fmt.kind = "word"
    elif re.search(r"\b(comma[- ]separated|separated by commas)\b", low):
        fmt.kind, fmt.separator = "list", ", "

    d = re.search(r"(\d+)\s+decimal places?", low)
    if d and fmt.kind in {"free", "number", "integer"}:
        fmt.kind, fmt.decimals = "number", int(d.group(1))

    choices = _detect_choices(text)
    if choices and fmt.kind in {"free", "word"}:
        fmt.kind, fmt.choices = "choice", choices

    if re.search(r"\b(uppercase|upper case|all caps|capital letters)\b", low):
        fmt.case = "upper"
    elif re.search(r"\b(lowercase|lower case)\b", low):
        fmt.case = "lower"
    return fmt


def _detect_choices(text: str) -> list[str]:
    # (A) foo  (B) bar   |  A) foo  B) bar  |  A. foo
    letters = re.findall(r"(?m)(?:^|\s)\(?([A-H])[\).:]\s+\S", text)
    if len(letters) >= 2 and letters[0] == "A":
        uniq = []
        for ch in letters:
            if ch not in uniq:
                uniq.append(ch)
        return uniq
    m = re.search(r"(?i)\b(?:one of|choose from|options?)\s*[:\-]?\s*\[?([^\n\]]{3,200})", text)
    if m:
        opts = [o.strip(" '\"`.") for o in re.split(r",|\bor\b|/|\|", m.group(1))]
        opts = [o for o in opts if o and len(o) <= 40]
        if 2 <= len(opts) <= 10:
            return opts
    return []


# ------------------------------------------------------------- extraction
def strip_think(raw: str) -> str:
    return _THINK_RE.sub("", raw or "").strip()


def is_invalid(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    low = text.lower()
    if low in {"unknown", "n/a", "none", "null", "i don't know", "idk", "<final answer>", "<answer>"}:
        return True
    if low.startswith(("reasoning:", "analysis:", "action:", "toolcall")):
        return True
    return any(tok in low for tok in _BLOCKED)


def extract_answer(raw: str, fmt: AnswerFormat | None = None) -> str:
    """Pull the final answer out of a model response."""
    text = strip_think(raw)
    if not text:
        return ""
    lines = text.splitlines()
    found = ""
    for i, line in enumerate(lines):
        m = _ANSWER_LINE_RE.match(line)
        if not m:
            continue
        value = m.group(1).strip()
        # Multi-line JSON after "ANSWER:"
        if (not value or value[:1] in "{[") and fmt and fmt.kind == "json":
            block = _balanced_json("\n".join([value] + lines[i + 1:]))
            if block:
                value = block
        if not value and i + 1 < len(lines):
            value = lines[i + 1].strip()
        value = value.strip().strip("`").strip()
        if value and not is_invalid(value):
            found = value  # keep the last valid ANSWER line
    if found:
        return found
    if fmt and fmt.kind == "json":
        block = _balanced_json(text)
        if block:
            return block
    # Single short line with no label: accept it.
    nonempty = [ln.strip() for ln in lines if ln.strip()]
    if (
        len(nonempty) == 1
        and len(nonempty[0]) <= 200
        and not _ANSWER_LINE_RE.match(nonempty[0])
        and not is_invalid(nonempty[0])
    ):
        return nonempty[0].strip("`").strip()
    return ""


def _balanced_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for j in range(start, len(text)):
                ch = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : j + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    return ""


# ------------------------------------------------------ validation/normalise
def normalize(answer: str, fmt: AnswerFormat) -> str:
    a = str(answer or "").strip()
    if not a:
        return a
    # Strip wrapping quotes/backticks/bold
    a = re.sub(r"^(\*\*|__)(.*)\1$", r"\2", a).strip()
    if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'`" and fmt.kind != "json":
        a = a[1:-1].strip()

    if fmt.kind == "json":
        block = _balanced_json(a) or a
        try:
            return json.dumps(json.loads(block), ensure_ascii=False, separators=(", ", ": "))
        except json.JSONDecodeError:
            return a
    if fmt.kind in {"integer", "number"}:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", a)
        if nums:
            n = nums[-1] if len(nums) > 1 and re.search(r"(?i)\b(is|=|total|answer)\b", a) else nums[0]
            n = n.replace(",", "")
            if fmt.kind == "integer":
                try:
                    f = float(n)
                    if f.is_integer():
                        n = str(int(f))
                except ValueError:
                    pass
            elif fmt.decimals is not None:
                try:
                    n = f"{float(n):.{fmt.decimals}f}"
                except ValueError:
                    pass
            a = n
    elif fmt.kind == "yesno":
        low = a.lower()
        if re.match(r"^\s*yes\b", low):
            a = "yes"
        elif re.match(r"^\s*no\b", low):
            a = "no"
    elif fmt.kind == "choice" and fmt.choices:
        a = _match_choice(a, fmt.choices)
    elif fmt.kind == "word":
        a = a.rstrip(".!?")
    else:
        # Free text: drop a lone trailing period on short answers.
        if len(a) <= 60 and a.endswith(".") and a.count(".") == 1:
            a = a[:-1]

    if fmt.case == "upper":
        a = a.upper()
    elif fmt.case == "lower":
        a = a.lower()
    return a


def _match_choice(a: str, choices: list[str]) -> str:
    stripped = a.strip().strip("()[]. ")
    for c in choices:
        if stripped.lower() == c.lower():
            return c
    m = re.match(r"^\(?([A-H])\)?(?:[\).:\s]|$)", a.strip())
    if m and m.group(1) in choices:
        return m.group(1)
    for c in choices:
        if len(c) > 1 and re.search(rf"\b{re.escape(c)}\b", a, re.IGNORECASE):
            return c
    return a


def validate(answer: str, fmt: AnswerFormat) -> list[str]:
    """Return a list of problems; empty means the answer looks well-formed."""
    problems: list[str] = []
    a = str(answer or "").strip()
    if is_invalid(a):
        return ["empty or placeholder answer"]
    if len(a) > 2000:
        problems.append("answer is very long")
    if fmt.kind == "json":
        try:
            json.loads(a)
        except json.JSONDecodeError:
            problems.append("not valid JSON")
    elif fmt.kind == "integer" and not re.fullmatch(r"-?\d+", a):
        problems.append("not an integer")
    elif fmt.kind == "number":
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", a):
            problems.append("not a number")
        elif fmt.decimals is not None and fmt.decimals > 0 and not re.fullmatch(rf"-?\d+\.\d{{{fmt.decimals}}}", a):
            problems.append(f"needs exactly {fmt.decimals} decimal places")
    elif fmt.kind == "yesno" and a.lower() not in {"yes", "no"}:
        problems.append("not yes/no")
    elif fmt.kind == "choice" and fmt.choices and a not in fmt.choices:
        problems.append(f"not one of {fmt.choices}")
    elif fmt.kind == "word" and len(a.split()) != 1:
        problems.append("not a single word")
    if fmt.case == "upper" and a != a.upper():
        problems.append("not uppercase")
    if fmt.case == "lower" and a != a.lower():
        problems.append("not lowercase")
    return problems


def finalize(raw_or_answer: str, fmt: AnswerFormat, *, already_extracted: bool = False) -> tuple[str, list[str]]:
    """Extract (if needed), normalise and validate. Returns (answer, problems)."""
    ans = raw_or_answer if already_extracted else extract_answer(raw_or_answer, fmt)
    ans = normalize(ans, fmt)
    return ans, validate(ans, fmt)
