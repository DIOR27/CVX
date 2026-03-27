import httpx
import json
import re
import pdfplumber
import tempfile
import os
import unicodedata
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="CV Analyzer API", version="5.1.0")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "https://pruebask8s.ucuenca.edu.ec/qwen/api/generate"
)
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "30"))
OLLAMA_READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "45"))
OLLAMA_WRITE_TIMEOUT = float(os.getenv("OLLAMA_WRITE_TIMEOUT", "30"))
OLLAMA_EXTRACTION_READ_TIMEOUT = float(
    os.getenv("OLLAMA_EXTRACTION_READ_TIMEOUT", "25")
)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
VERIFY_MAX_TOKENS = int(os.getenv("VERIFY_MAX_TOKENS", "5"))
EXTRACTION_MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", "1200"))
EXPERIENCE_SECTION_MAX_CHARS = int(os.getenv("EXPERIENCE_SECTION_MAX_CHARS", "3000"))
PDF_LEFT_CROP_RATIO = float(os.getenv("PDF_LEFT_CROP_RATIO", "0.68"))
PDF_RIGHT_CROP_START_RATIO = float(os.getenv("PDF_RIGHT_CROP_START_RATIO", "0.30"))
ENABLE_REMOTE_EXTRACTION = (
    os.getenv("ENABLE_REMOTE_EXTRACTION", "true").strip().lower() == "true"
)


def get_ollama_timeout(read_timeout: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(
        connect=OLLAMA_CONNECT_TIMEOUT,
        read=read_timeout or OLLAMA_READ_TIMEOUT,
        write=OLLAMA_WRITE_TIMEOUT,
        pool=OLLAMA_CONNECT_TIMEOUT,
    )

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

    # "day month year - day month year"
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

    # "month year - month year" or "month year a month year"
    if m := re.match(
        r"(\w+)\s+(\d{4})\s*(?:-|–|a|al|hasta|to)\s*(\w+)\s+(\d{4})",
        normalized,
        re.IGNORECASE,
    ):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(2)),
            to_date_dict(month=parse_month(m.group(3)), year=m.group(4)),
        )

    # "month year - current/present"
    if m := re.match(
        r"(\w+)\s+(\d{4})\s*[-–]\s*(actualidad|presente|actual|hoy|vigente)",
        normalized,
        re.IGNORECASE,
    ):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(2)),
            "Current",
        )

    # "year-year"
    if m := re.match(r"^(\d{4})\s*[-–]\s*(\d{4})$", normalized):
        return to_date_dict(year=m.group(1)), to_date_dict(year=m.group(2))

    # "month a month year" -> "March to November 2021"
    if m := re.match(r"(\w+)\s+a\s+(\w+)\s+(\d{4})", normalized, re.IGNORECASE):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(3)),
            to_date_dict(month=parse_month(m.group(2)), year=m.group(3)),
        )

    # "month year" only
    if m := re.match(r"^(\w+)\s+(\d{4})$", normalized, re.IGNORECASE):
        return to_date_dict(month=parse_month(m.group(1)), year=m.group(2)), None

    # year only
    if m := re.match(r"^(\d{4})$", normalized):
        return to_date_dict(year=m.group(1)), None

    return None, None


def extract_text_from_pdf(
    file_path: str,
    crop_ratio: float | None = None,
    crop_start_ratio: float | None = None,
) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            source_page = page
            if crop_ratio is not None or crop_start_ratio is not None:
                x0, top, x1, bottom = page.bbox
                start_x = x0 + ((x1 - x0) * (crop_start_ratio or 0.0))
                end_x = x0 + ((x1 - x0) * (crop_ratio or 1.0))
                source_page = page.crop((start_x, top, end_x, bottom))

            page_text = source_page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_experience_section(full_text: str, max_chars: int = 5000) -> str:
    exp_headers = re.compile(
        r"(^|\n)\s*(experiencia(?:\s+laboral)?|experiencia\s+como\s+director\s+de\s+proyectos|work\s+experience)\s*(\n|$)",
        re.IGNORECASE,
    )
    exp_end = re.compile(
        r"\n\s*(titulaci[oó]n\s+acad|formaci[oó]n\s+acad|certificacion|"
        r"competencia\s+profesion|referencias\s+person|aptitudes|"
        r"habilidades\s*\n|idiomas\s*\n|cursos\s+y\s+|educaci[oó]n\s*\n|"
        r"principales\s+pericias|proyectos\s+destacados|desarrollo\s+m[oó]vil|arquitectura\s*&\s*patrones|backend\s*&\s*databases)",
        re.IGNORECASE,
    )

    if start_match := exp_headers.search(full_text):
        start_idx = start_match.start(2)
        text_from_exp = full_text[start_idx:]
        if end_match := exp_end.search(text_from_exp[50:]):
            section = text_from_exp[: 50 + end_match.start()]
        else:
            section = text_from_exp
        return section[:max_chars] if len(section) > 200 else full_text[:max_chars]

    return full_text[:max_chars]


MONTH_NAME_PATTERN = (
    r"(?:ene(?:ro)?|feb(?:rero)?|mar(?:zo)?|abr(?:il)?|may(?:o)?|jun(?:io)?|"
    r"jul(?:io)?|ago(?:sto)?|sep(?:tiembre)?|sept(?:iembre)?|setiembre|"
    r"oct(?:ubre)?|nov(?:iembre)?|dic(?:iembre)?|jan(?:uary)?|feb(?:ruary)?|"
    r"mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DATE_VALUE_PATTERN = rf"{MONTH_NAME_PATTERN}[.,]?\s+\d{{4}}|\d{{4}}|actualidad|presente|actual|current|hoy|vigente"
DATE_RANGE_PATTERN = re.compile(
    rf"({DATE_VALUE_PATTERN})\s*(?:-|–|a|al|hasta|to)\s*({DATE_VALUE_PATTERN})",
    re.IGNORECASE,
)
COMPANY_DATE_PATTERN = re.compile(
    rf"^(?:en\s+)?(.+?)\s*(?:\||-|–)\s*({DATE_VALUE_PATTERN}\s*(?:-|–|a|al|hasta|to)\s*{DATE_VALUE_PATTERN}|{DATE_VALUE_PATTERN})\.?\s*$",
    re.IGNORECASE,
)
DATE_LINE_PATTERN = re.compile(
    rf"^\s*({MONTH_NAME_PATTERN}[.,]?\s+\d{{4}})\s+(.+?)\s*$", re.IGNORECASE
)
DATE_RANGE_LINE_PATTERN = re.compile(
    rf"^\s*-\s*({MONTH_NAME_PATTERN}[.,]?\s+\d{{4}}|actualidad|presente|actual|current|hoy|vigente)(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
CURRENT_ONLY_LINE_PATTERN = re.compile(r"^\s*-\s*$")
NON_COMPANY_PREFIXES = (
    "desarrollo",
    "analisis",
    "análisis",
    "diseno",
    "diseño",
    "automatizacion",
    "automatización",
    "soporte",
    "mantenimiento",
    "implementacion",
    "implementación",
    "monitorizacion",
    "monitorización",
    "tratamiento",
    "ci/cd",
    "pipelines",
    "rest",
    "actividades realizadas",
    "doy mi consentimiento",
    "este curriculum",
    "idiomas",
    "contacto",
    "mas informacion",
    "m?s informaci?n",
    "perfil",
    "sobre mi",
    "sobre mí",
)
SECTION_END_PREFIXES = (
    "formacion",
    "formación",
    "educacion",
    "educación",
    "certificaciones",
    "cursos",
    "competencias",
    "habilidades",
    "idiomas",
    "referencias",
    "contacto",
)
JOB_TITLE_KEYWORDS = (
    "analista",
    "developer",
    "desarrollador",
    "ingeniero",
    "consultor",
    "director",
    "docente",
    "administrador",
    "backend",
    "frontend",
    "fullstack",
    "full stack",
    "software",
    "programador",
    "qa",
    "quality",
    "arquitecto",
    "lider",
    "líder",
    "ejecutivo",
    "pasante",
    "pasantias",
    "pasantías",
    "asistente",
)


def sanitize_date_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.,]", "", text or "")).strip()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", normalized)


def clean_job_title(text: str) -> str:
    title = re.sub(r"\s+", " ", text.strip(" -|."))
    title = re.sub(r"^(laboral\s+)", "", title, flags=re.IGNORECASE)
    return re.sub(r"^(en\s+)", "", title, flags=re.IGNORECASE).strip()


def clean_company_name(text: str) -> str:
    company = re.sub(r"\s+", " ", text.strip(" -|."))
    company = re.sub(r"^(en\s+)", "", company, flags=re.IGNORECASE).strip()
    return company


def looks_like_section_end(text: str) -> bool:
    normalized = normalize_text(text)
    return any(normalized.startswith(prefix) for prefix in SECTION_END_PREFIXES)


def refine_job_title(text: str) -> str:
    title = clean_job_title(text)
    if looks_like_job_title_line(title) and len(title.split()) <= 8:
        return title

    patterns = [
        r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ'/-]*(?:\s+(?:de|del|y|&|/|-)?\s*[A-ZÁÉÍÓÚÑ]?[\wÁÉÍÓÚÑáéíóúñ'/-]*){0,6})$",
        r"((?:Backend|Frontend|Fullstack|Full Stack|Software|Wordpress|Data|QA|Analista|Desarrollador|Ingeniero|Consultor|Director|Administrador|Arquitecto|Docente|Pasante)[^|]{0,80})$",
    ]
    for pattern in patterns:
        if match := re.search(pattern, title, re.IGNORECASE):
            candidate = clean_job_title(match.group(1))
            if looks_like_job_title_line(candidate):
                return candidate

    return title


def looks_like_job_title_line(text: str) -> bool:
    candidate = text.strip()
    if not candidate or len(candidate) < 3 or len(candidate) > 120:
        return False

    normalized = normalize_text(candidate)
    if normalized.startswith(("-", "?", "actividades realizadas")):
        return False
    if looks_like_section_end(candidate) or normalized.startswith(NON_COMPANY_PREFIXES):
        return False
    if normalized in {"experiencia", "experiencia laboral", "work experience"}:
        return False
    if candidate.endswith((",", ":", ";")):
        return False
    if "@" in candidate or "http" in normalized:
        return False

    if any(keyword in normalized for keyword in JOB_TITLE_KEYWORDS):
        return True

    words = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ][A-Za-zÁÉÍÓÚáéíóúÑñ'/-]*", candidate)
    if not words:
        return False

    title_like_words = sum(1 for word in words if word[:1].isupper() or word.isupper())
    return (title_like_words / len(words)) >= 0.6


def extract_date_range_from_text(text: str) -> tuple[dict | None, dict | None | str, tuple[int, int] | None]:
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


def parse_company_date_line(text: str) -> tuple[str, dict | None, dict | None | str]:
    candidate = text.strip()
    if not candidate:
        return "", None, None

    candidate = re.sub(r"^(en\s+)", "", candidate, flags=re.IGNORECASE).strip()
    if "|" in candidate:
        company_part, date_part = candidate.split("|", 1)
        start_date, end_date = parse_date_string(sanitize_date_text(date_part))
        return clean_company_name(company_part), start_date, end_date

    match = re.match(
        rf"^(.+?)\s+({DATE_VALUE_PATTERN}\s*(?:-|–|a|al|hasta|to)\s*{DATE_VALUE_PATTERN}|{DATE_VALUE_PATTERN})\.?\s*$",
        candidate,
        re.IGNORECASE,
    )
    if not match:
        return "", None, None

    start_date, end_date = parse_date_string(sanitize_date_text(match.group(2)))
    return clean_company_name(match.group(1)), start_date, end_date


def looks_like_company_line(text: str) -> bool:
    candidate = text.strip()
    if not candidate or len(candidate) < 4:
        return False

    lowered = normalize_text(candidate)
    if "@" in candidate or "http" in lowered:
        return False
    if lowered in {"experiencia", "educacion", "certificaciones", "cursos y talleres"}:
        return False
    if lowered.startswith(NON_COMPANY_PREFIXES):
        return False
    if looks_like_section_end(candidate):
        return False
    if extract_date_range_from_text(candidate)[0]:
        return False

    return True


def extract_experience_blocks(text: str) -> list:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    results = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if looks_like_section_end(line):
            break

        start_date = None
        end_date = None
        job_title = ""
        company = ""
        consumed = 1

        inline_range = extract_date_range_from_text(line)
        inline_match = DATE_RANGE_PATTERN.search(sanitize_date_text(line))
        if inline_match and inline_match.start() > 0:
            start_date, end_date, span = inline_range
            job_title = refine_job_title(sanitize_date_text(line)[: span[0]])
            if looks_like_job_title_line(job_title) and i + 1 < len(lines) and looks_like_company_line(lines[i + 1]):
                company = clean_company_name(lines[i + 1])
                consumed = 2
            else:
                job_title = ""
        else:
            match = DATE_LINE_PATTERN.match(line)
            if match and looks_like_job_title_line(match.group(2)):
                start_date = convert_date_value(sanitize_date_text(match.group(1)))
                job_title = refine_job_title(match.group(2))
                j = i + 1

                if j < len(lines) and CURRENT_ONLY_LINE_PATTERN.match(lines[j]):
                    end_date = "Current"
                    j += 1
                elif j < len(lines):
                    range_match = DATE_RANGE_LINE_PATTERN.match(lines[j])
                    if range_match:
                        range_value = sanitize_date_text(range_match.group(1))
                        trailing_company = clean_company_name(range_match.group(2) or "")
                        end_date = (
                            "Current"
                            if normalize_text(range_value) in CURRENT_WORDS
                            else convert_date_value(range_value)
                        )
                        if trailing_company:
                            company = trailing_company
                        j += 1

                if not company and j < len(lines) and looks_like_company_line(lines[j]):
                    company = clean_company_name(lines[j])
                    j += 1

                consumed = max(j - i, 1)
            else:
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                next_next_line = lines[i + 2] if i + 2 < len(lines) else ""

                if looks_like_job_title_line(line):
                    next_range = extract_date_range_from_text(next_line)
                    parsed_company, parsed_start, parsed_end = parse_company_date_line(
                        next_line
                    )
                    if parsed_company and parsed_start and (
                        "|" in next_line or next_line.lower().startswith("en ")
                    ):
                        job_title = refine_job_title(line)
                        company = parsed_company
                        start_date, end_date = parsed_start, parsed_end
                        consumed = 2
                    elif next_range[0] and re.match(
                        rf"^\s*(?:{DATE_VALUE_PATTERN})\s*(?:-|–|a|al|hasta|to)",
                        sanitize_date_text(next_line),
                        re.IGNORECASE,
                    ):
                        job_title = refine_job_title(line)
                        start_date, end_date, _ = next_range
                        if looks_like_company_line(next_next_line):
                            company = clean_company_name(next_next_line)
                            consumed = 3
                    elif looks_like_company_line(next_line):
                        company_and_date = COMPANY_DATE_PATTERN.match(next_next_line)
                        if company_and_date and len(line) > 2:
                            job_title = clean_job_title(line)
                            company = clean_company_name(company_and_date.group(1))
                            start_date, end_date = parse_date_string(
                                sanitize_date_text(company_and_date.group(2))
                            )
                            consumed = 3

        if job_title and company and start_date:
            results.append(
                {
                    "job_title": job_title,
                    "company": company,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            i += consumed
            continue

        i += 1

    return results


def call_ollama(prompt: str, max_tokens: int = 2000, read_timeout: float | None = None) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    try:
        with httpx.Client(timeout=get_ollama_timeout(read_timeout=read_timeout)) as client:
            response = client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "The remote Ollama server took too long to respond."
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "The remote Ollama server returned an unexpected error."
        ) from exc

    if isinstance(data, dict):
        return str(data.get("response", "")).strip()
    return ""


CV_KEYWORDS = [
    "experiencia laboral",
    "formación académica",
    "hoja de vida",
    "currículum",
    "curriculum",
    "datos personales",
    "perfil profesional",
    "sobre mí",
    "referencias",
    "aptitudes",
    "habilidades",
    "desarrollador",
    "ingeniero",
    "licenciado",
    "analista",
    "programador",
    "gerente",
    "director",
    "experience",
    "education",
    "skills",
    "resume",
    "employment",
    "work experience",
    "professional profile",
]


def verify_is_cv(text: str) -> bool:
    text_lower = text.lower()
    matches = sum(1 for kw in CV_KEYWORDS if kw in text_lower)
    if matches >= 2:
        return True
    prompt = (
        f'Is this text a resume? Answer only "YES" or "NO":\n\n{text[:500]}\n\nAnswer:'
    )
    response = call_ollama(prompt, max_tokens=VERIFY_MAX_TOKENS).strip().upper()
    return any(w in response for w in ["SI", "SÍ", "YES", "Y"])


EXTRACTION_PROMPT = """Extrae experiencias laborales del CV y responde SOLO con un arreglo JSON valido.
Cada item debe tener:
job_title, company, start_date, end_date

Reglas:
- job_title es el cargo, no la empresa.
- company es la organizacion.
- Usa fechas como {"day":null,"month":5,"year":2022}
- Si termina en presente/actualidad/current usa "Current"
- No inventes informacion.
- Excluye educacion, cursos, certificaciones y habilidades.

Formato:
[{"job_title":"...","company":"...","start_date":{"day":null,"month":5,"year":2022},"end_date":"Current"}]

CV:
{cv_text}
JSON:"""


def parse_json_response(response: str) -> list:
    response = re.sub(r"```(?:json)?\s*", "", response).strip()

    if m := re.search(r"\[.*\]", response, re.DOTALL):
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    if m := re.search(r"\{.*\}", response, re.DOTALL):
        try:
            obj = json.loads(m.group())
            for key in [
                "experiencia",
                "experience",
                "experiencia_laboral",
                "jobs",
                "work_experience",
            ]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
        except json.JSONDecodeError:
            pass

    for suffix in ["]", "\n]"]:
        try:
            data = json.loads(response + suffix)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


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
        if value.lower() in CURRENT_WORDS:
            return "Current"
        start, _ = parse_date_string(value)
        return start
    return None


def clean_items(items: list) -> list:
    result = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        job_title = (item.get("job_title") or item.get("cargo") or "").strip()
        company = (item.get("company") or item.get("empresa") or "").strip()
        start_date = item.get("start_date") or item.get("fecha_inicio")
        end_date = item.get("end_date") or item.get("fecha_fin")

        # Remove "En " prefix
        for prefix in ["En ", "en "]:
            if company.startswith(prefix):
                company = company[len(prefix) :]
            if job_title.startswith(prefix):
                job_title = job_title[len(prefix) :]

        # Convert dates to numeric format
        start_date = convert_date_value(start_date)
        end_date = convert_date_value(end_date)

        # Split job title by "|"
        titles = (
            [t.strip() for t in job_title.split("|")]
            if "|" in job_title
            else [job_title]
        )

        for title in titles:
            if not title or not company:
                continue
            key = (title.lower()[:40], company.lower()[:40], str(start_date))
            if key not in seen:
                seen.add(key)
                result.append(
                    {
                        "job_title": title,
                        "company": company,
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                )

    return result


def is_suspicious_item(item: dict) -> bool:
    job_title = normalize_text(item.get("job_title", ""))
    company = normalize_text(item.get("company", ""))

    if not job_title or not company:
        return True
    if "@" in company or "http" in company:
        return True
    if looks_like_section_end(job_title) or looks_like_section_end(company):
        return True
    if company.startswith(NON_COMPANY_PREFIXES) or job_title.startswith(NON_COMPANY_PREFIXES):
        return True
    if extract_date_range_from_text(company)[0]:
        return True
    if job_title in CURRENT_WORDS:
        return True
    if company in {"email", "telefono", "teléfono", "linkedin", "contacto"}:
        return True
    if item.get("company", "")[:1].islower():
        return True
    if len(company.split()) > 8:
        return True
    if job_title == company:
        return True

    return False


def score_items(items: list) -> int:
    if not items:
        return 0

    suspicious = sum(1 for item in items if is_suspicious_item(item))
    return (len(items) * 10) - (suspicious * 8)


def extract_with_regex(text: str) -> list:
    results = extract_experience_blocks(text)
    seen = set()

    for item in results:
        key = (item["job_title"].lower()[:30], item["company"].lower()[:30])
        seen.add(key)

    pattern = re.compile(
        r"\b(Director[^,\n]{0,60}?|Gerente[^,\n]{0,60}?|Especialista[^,\n]{0,60}?|"
        r"Asesor[^,\n]{0,60}?|Director\s+T[eé]cnico[^,\n]{0,40}?)"
        r",\s*"
        r"(\d{4}[-–]\d{4}|\w+\s+a\s+\w+\s+(?:de\s+)?\d{4}|"
        r"\w+\s+\d{4}\s*[-–]\s*\w+\s+\d{4}|\d{4})"
        r",?\s*(?:en\s+)?"
        r"([A-ZÁÉÍÓÚÑÜ][^\n.]{5,80})",
        re.IGNORECASE,
    )

    for m in pattern.finditer(text):
        job_title = m.group(1).strip().rstrip(",")
        date_raw = m.group(2).strip()
        company_raw = m.group(3).strip()

        # Clean company
        company = re.split(r"\s*[\(\[]|\s{2,}", company_raw)[0].strip()
        company = re.sub(r"\.$", "", company).strip()

        # Parse dates
        start_date, end_date = parse_date_string(date_raw)

        key = (job_title.lower()[:30], company.lower()[:30])
        if key not in seen and len(job_title) > 3 and len(company) > 3:
            seen.add(key)
            results.append(
                {
                    "job_title": job_title,
                    "company": company,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )

    return results


def extract_structured_role_entries(text: str) -> list:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    results = []

    for i, line in enumerate(lines):
        if looks_like_section_end(line):
            break

        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        # Pattern: ROLE / COMPANY | DATE
        parsed_company, start_date, end_date = parse_company_date_line(next_line)
        if (
            looks_like_job_title_line(line)
            and parsed_company
            and start_date
            and "|" in next_line
        ):
            results.append(
                {
                    "job_title": refine_job_title(line),
                    "company": parsed_company,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            continue

        # Pattern: Role, 2014-2016 en Company
        if match := re.match(
            rf"^(.+?),\s*({DATE_VALUE_PATTERN}\s*(?:-|–|a|al|hasta|to)\s*{DATE_VALUE_PATTERN}|{DATE_VALUE_PATTERN})\s+en\s+(.+)$",
            line,
            re.IGNORECASE,
        ):
            job_title = refine_job_title(match.group(1))
            start_date, end_date = parse_date_string(sanitize_date_text(match.group(2)))
            company = clean_company_name(match.group(3))
            if looks_like_job_title_line(job_title) and company and start_date:
                results.append(
                    {
                        "job_title": job_title,
                        "company": company,
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                )
                continue

        # Pattern for uppercase role line followed by company/date on next line
        parsed_company, start_date, end_date = parse_company_date_line(next_line)
        alpha_chars = [ch for ch in line if ch.isalpha()]
        if (
            parsed_company
            and start_date
            and alpha_chars
            and sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars) > 0.7
            and len(line.split()) >= 2
        ):
            results.append(
                {
                    "job_title": refine_job_title(line.title()),
                    "company": parsed_company,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )

    return results


def extract_consulting_entries(text: str) -> list:
    normalized_text = re.sub(r"\s+", " ", text)
    results = []

    patterns = [
        re.compile(
            rf"(Gerente\s+de\s+Contrato|Director\s+General\s+de\s+Proyecto|Director\s+de\s+Control\s+y\s+Programaci[oó]n|Especialista\s+en\s+Control\s+y\s+Planillas|Director\s+de\s+Proyecto(?:\s*\([^)]+\))?)"
            rf",\s*({DATE_VALUE_PATTERN}\s*(?:-|–|a|al|hasta|to)\s*{DATE_VALUE_PATTERN}|{DATE_VALUE_PATTERN})"
            rf"(?:\s+en)?\s*,?\s*([^.;]{8,180})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(Director\s+T[eé]cnico(?:\s*\([^)]+\))?)\s+en\s+([^,.;]{{5,160}}),\s*({DATE_VALUE_PATTERN}\s*(?:-|–|a|al|hasta|to)\s*{DATE_VALUE_PATTERN}|{DATE_VALUE_PATTERN})",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        for match in pattern.finditer(normalized_text):
            if pattern is patterns[1]:
                job_title = refine_job_title(match.group(1))
                company = clean_company_name(match.group(2))
                date_raw = sanitize_date_text(match.group(3))
            else:
                job_title = refine_job_title(match.group(1))
                date_raw = sanitize_date_text(match.group(2))
                company = clean_company_name(match.group(3))

            start_date, end_date = parse_date_string(date_raw)
            if looks_like_job_title_line(job_title) and company and start_date:
                results.append(
                    {
                        "job_title": job_title,
                        "company": company,
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                )

    return results


def extract_work_experience(full_text: str, allow_remote_refine: bool = False) -> list:
    section = extract_experience_section(full_text, max_chars=EXPERIENCE_SECTION_MAX_CHARS)

    # Keep a local fallback ready in case the remote model is slow or unavailable.
    regex_items = (
        extract_with_regex(section)
        + extract_structured_role_entries(section)
        + extract_consulting_entries(section)
    )
    regex_cleaned = clean_items(regex_items)
    local_is_reliable = score_items(regex_cleaned) >= (len(regex_cleaned) * 8)
    if regex_cleaned and (
        not allow_remote_refine or local_is_reliable
    ):
        return regex_cleaned

    if not ENABLE_REMOTE_EXTRACTION:
        return regex_cleaned if local_is_reliable else []

    prompt = EXTRACTION_PROMPT.replace("{cv_text}", section)
    try:
        response = call_ollama(
            prompt,
            max_tokens=EXTRACTION_MAX_TOKENS,
            read_timeout=OLLAMA_EXTRACTION_READ_TIMEOUT,
        )
        items = parse_json_response(response)
        results = clean_items(items)
    except RuntimeError:
        return regex_cleaned if local_is_reliable else []

    existing_keys = {
        (r["job_title"].lower()[:30], r["company"].lower()[:30]) for r in results
    }
    for item in regex_cleaned:
        key = (item["job_title"].lower()[:30], item["company"].lower()[:30])
        if key not in existing_keys:
            results.append(item)
            existing_keys.add(key)

    if results:
        return results
    return regex_cleaned if local_is_reliable else []


@app.post("/analizar-cv")
async def analyze_cv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)

        if not text or len(text) < 50:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF. It may be a scanned PDF without OCR.",
            )

        if not verify_is_cv(text):
            return JSONResponse(
                content={
                    "es_cv": False,
                    "message": "The document does not appear to be a resume or CV.",
                    "experiencia_laboral": [],
                }
            )

        left_text = extract_text_from_pdf(tmp_path, crop_ratio=PDF_LEFT_CROP_RATIO)
        right_text = extract_text_from_pdf(
            tmp_path, crop_start_ratio=PDF_RIGHT_CROP_START_RATIO
        )
        local_left = clean_items(
            extract_with_regex(
                extract_experience_section(
                    left_text, max_chars=EXPERIENCE_SECTION_MAX_CHARS
                )
            )
        )
        local_full = clean_items(
            extract_with_regex(
                extract_experience_section(text, max_chars=EXPERIENCE_SECTION_MAX_CHARS)
            )
        )
        local_right = clean_items(
            extract_with_regex(
                extract_experience_section(
                    right_text, max_chars=EXPERIENCE_SECTION_MAX_CHARS
                )
            )
        )
        candidates = [
            (score_items(local_left), local_left, left_text),
            (score_items(local_right), local_right, right_text),
            (score_items(local_full), local_full, text),
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, experiencia, selected_text = candidates[0]

        if not experiencia or score_items(experiencia) < (len(experiencia) * 8):
            remote_source = selected_text if best_score > 0 else text
            experiencia = extract_work_experience(
                remote_source, allow_remote_refine=True
            )

        return JSONResponse(
            content={
                "es_cv": True,
                "file": file.filename,
                "total_experiencias": len(experiencia),
                "experiencia_laboral": experiencia,
            }
        )
    except RuntimeError as exc:
        message = str(exc)
        status_code = 504 if "too long" in message.lower() else 502
        raise HTTPException(status_code=status_code, detail=message) from exc

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    try:
        tags_url = OLLAMA_URL.rsplit("/", 1)[0] + "/tags"
        with httpx.Client(timeout=5.0) as client:
            r = client.get(tags_url)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
        return {
            "status": "ok",
            "ollama": "connected",
            "ollama_url": OLLAMA_URL,
            "configured_model": MODEL,
            "available_models": models,
        }
    except Exception as e:
        return {
            "status": "error",
            "ollama_url": OLLAMA_URL,
            "configured_model": MODEL,
            "ollama": str(e),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
