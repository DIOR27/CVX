import re

from app.config import ENABLE_REMOTE_EXTRACTION, OLLAMA_EXTRACTION_READ_TIMEOUT
from app.date_parsing import (
    CURRENT_WORDS,
    DATE_RANGE_PATTERN,
    DATE_VALUE_PATTERN,
    LONG_DATE_PATTERN,
    convert_date_value,
    extract_date_range_from_text,
    normalize_text,
    parse_date_string,
    sanitize_date_text,
)
from app.ollama_client import call_ollama, extraction_prompt, parse_json_response
from app.pdf_processing import extract_experience_section

DATE_LINE_PATTERN = re.compile(
    rf"^\s*({DATE_VALUE_PATTERN})\s+(.+?)\s*$", re.IGNORECASE
)
DATE_RANGE_LINE_PATTERN = re.compile(
    rf"^\s*-\s*({DATE_VALUE_PATTERN})(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
CURRENT_ONLY_LINE_PATTERN = re.compile(r"^\s*-\s*$")
COMPANY_DATE_PATTERN = re.compile(
    rf"^(?:en\s+)?(.+?)\s*(?:\||-|–)\s*({DATE_VALUE_PATTERN}\s*(?:-|–|a|al|hasta|to)\s*{DATE_VALUE_PATTERN}|{DATE_VALUE_PATTERN})\.?\s*$",
    re.IGNORECASE,
)

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
            if (
                looks_like_job_title_line(job_title)
                and i + 1 < len(lines)
                and looks_like_company_line(lines[i + 1])
            ):
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


def extract_with_regex(text: str) -> list:
    results = extract_experience_blocks(text)
    seen = set()
    for item in results:
        seen.add((item["job_title"].lower()[:30], item["company"].lower()[:30]))

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

    for match in pattern.finditer(text):
        job_title = match.group(1).strip().rstrip(",")
        date_raw = match.group(2).strip()
        company_raw = match.group(3).strip()
        company = re.split(r"\s*[\(\[]|\s{2,}", company_raw)[0].strip()
        company = re.sub(r"\.$", "", company).strip()
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
        parsed_company, start_date, end_date = parse_company_date_line(next_line)
        if looks_like_job_title_line(line) and parsed_company and start_date and "|" in next_line:
            results.append(
                {
                    "job_title": refine_job_title(line),
                    "company": parsed_company,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            continue

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
            rf"(?:\s+en)?\s*,?\s*([^.;]{{8,180}})",
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

        for prefix in ["En ", "en "]:
            if company.startswith(prefix):
                company = company[len(prefix) :]
            if job_title.startswith(prefix):
                job_title = job_title[len(prefix) :]

        start_date = convert_date_value(start_date)
        end_date = convert_date_value(end_date)

        titles = [t.strip() for t in job_title.split("|")] if "|" in job_title else [job_title]
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


def extract_local_candidates(text: str) -> list:
    section = extract_experience_section(text)
    return clean_items(
        extract_with_regex(section)
        + extract_structured_role_entries(section)
        + extract_consulting_entries(section)
    )


def extract_work_experience(full_text: str, allow_remote_refine: bool = False) -> list:
    section = extract_experience_section(full_text)
    regex_cleaned = extract_local_candidates(section)
    local_is_reliable = score_items(regex_cleaned) >= (len(regex_cleaned) * 8)
    if regex_cleaned and (not allow_remote_refine or local_is_reliable):
        return regex_cleaned

    if not ENABLE_REMOTE_EXTRACTION:
        return regex_cleaned

    prompt = extraction_prompt(section)
    try:
        response = call_ollama(
            prompt,
            read_timeout=OLLAMA_EXTRACTION_READ_TIMEOUT,
        )
        results = clean_items(parse_json_response(response))
    except RuntimeError:
        return regex_cleaned

    existing_keys = {
        (item["job_title"].lower()[:30], item["company"].lower()[:30]) for item in results
    }
    for item in regex_cleaned:
        key = (item["job_title"].lower()[:30], item["company"].lower()[:30])
        if key not in existing_keys:
            results.append(item)
            existing_keys.add(key)

    return results if results else regex_cleaned
