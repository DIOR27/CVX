import re
import unicodedata

MONTH_ALIASES = {
    1: {"ene", "enero", "jan", "january"},
    2: {"feb", "febrero", "february"},
    3: {"mar", "marzo", "march"},
    4: {"abr", "abril", "apr", "april"},
    5: {"may", "mayo"},
    6: {"jun", "junio", "june"},
    7: {"jul", "julio", "july"},
    8: {"ago", "agosto", "aug", "august"},
    9: {"sep", "sept", "septiembre", "setiembre", "september"},
    10: {"oct", "octubre", "october"},
    11: {"nov", "noviembre", "november"},
    12: {"dic", "diciembre", "dec", "december"},
}

CURRENT_WORDS = {"actualidad", "presente", "actual", "current", "hoy", "vigente"}
LONG_DATE_PATTERN = re.compile(
    r"(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚáéíóúÑñ.]+)\s+(?:del\s+)?(\d{4})",
    re.IGNORECASE,
)
MONTH_NAME_PATTERN = (
    r"(?:ene(?:ro)?|feb(?:rero)?|mar(?:zo)?|abr(?:il)?|may(?:o)?|jun(?:io)?|"
    r"jul(?:io)?|ago(?:sto)?|sep(?:tiembre)?|sept(?:iembre)?|setiembre|"
    r"oct(?:ubre)?|nov(?:iembre)?|dic(?:iembre)?|jan(?:uary)?|feb(?:ruary)?|"
    r"mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DATE_VALUE_PATTERN = (
    rf"{MONTH_NAME_PATTERN}[.,]?\s+\d{{4}}|\d{{4}}|actualidad|presente|actual|current|hoy|vigente"
)
DATE_RANGE_PATTERN = re.compile(
    rf"({DATE_VALUE_PATTERN})\s*(?:-|–|a|al|hasta|to)\s*({DATE_VALUE_PATTERN})",
    re.IGNORECASE,
)


def sanitize_date_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.,]", "", text or "")).strip()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", normalized)


def parse_month(text: str) -> int | None:
    if not text:
        return None

    normalized = text.strip().lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z]", "", normalized)

    for month_number, aliases in MONTH_ALIASES.items():
        if normalized in aliases:
            return month_number

    return None


def to_date_dict(day=None, month=None, year=None) -> dict | None:
    if day is None and month is None and year is None:
        return None
    if year is not None and not (1900 <= int(year) <= 2100):
        return None
    return {
        "day": int(day) if day else None,
        "month": int(month) if month else None,
        "year": int(year) if year else None,
    }


def parse_date_string(text: str) -> tuple[dict | None, dict | None | str]:
    if not text:
        return None, None

    text = text.strip()
    normalized = sanitize_date_text(text)

    if matches := list(LONG_DATE_PATTERN.finditer(text)):
        start = matches[0]
        start_date = to_date_dict(
            day=start.group(1),
            month=parse_month(start.group(2)),
            year=start.group(3),
        )

        if len(matches) > 1:
            end = matches[1]
            end_date = to_date_dict(
                day=end.group(1),
                month=parse_month(end.group(2)),
                year=end.group(3),
            )
            return start_date, end_date

        if re.search(
            r"(?:-|–|a|al|hasta|to)\s*(actualidad|presente|actual|current|hoy|vigente)",
            normalized,
            re.IGNORECASE,
        ):
            return start_date, "Current"

        return start_date, None

    if m := re.match(
        r"(\w+)\s+(\d{4})\s*(?:-|–|a|al|hasta|to)\s*(\w+)\s+(\d{4})",
        normalized,
        re.IGNORECASE,
    ):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(2)),
            to_date_dict(month=parse_month(m.group(3)), year=m.group(4)),
        )

    if m := re.match(
        r"(\w+)\s+(\d{4})\s*[-–]\s*(actualidad|presente|actual|hoy|vigente)",
        normalized,
        re.IGNORECASE,
    ):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(2)),
            "Current",
        )

    if m := re.match(r"^(\d{4})\s*[-–]\s*(\d{4})$", normalized):
        return to_date_dict(year=m.group(1)), to_date_dict(year=m.group(2))

    if m := re.match(r"(\w+)\s+a\s+(\w+)\s+(\d{4})", normalized, re.IGNORECASE):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(3)),
            to_date_dict(month=parse_month(m.group(2)), year=m.group(3)),
        )

    if m := re.match(r"^(\w+)\s+(\d{4})$", normalized, re.IGNORECASE):
        return to_date_dict(month=parse_month(m.group(1)), year=m.group(2)), None

    if m := re.match(r"^(\d{4})$", normalized):
        return to_date_dict(year=m.group(1)), None

    return None, None


def convert_date_value(value):
    if value is None:
        return None
    if value == "Current":
        return "Current"
    if isinstance(value, dict):
        return to_date_dict(
            day=value.get("day"),
            month=value.get("month"),
            year=value.get("year"),
        )
    if isinstance(value, str):
        if normalize_text(value) in CURRENT_WORDS:
            return "Current"
        start, _ = parse_date_string(value)
        return start
    return None


def extract_date_range_from_text(
    text: str,
) -> tuple[dict | None, dict | None | str, tuple[int, int] | None]:
    if not text:
        return None, None, None

    cleaned = sanitize_date_text(text)
    if match := DATE_RANGE_PATTERN.search(cleaned):
        start_date = convert_date_value(match.group(1))
        end_date = convert_date_value(match.group(2))
        return start_date, end_date, match.span()

    if match := re.search(rf"({DATE_VALUE_PATTERN})", cleaned, re.IGNORECASE):
        start_date = convert_date_value(match.group(1))
        return start_date, None, match.span()

    return None, None, None
