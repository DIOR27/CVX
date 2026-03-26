import httpx
import json
import re
import pdfplumber
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="CV Analyzer API", version="5.1.0")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

MONTH_MAP = {
    "enero": 1,
    "ene": 1,
    "january": 1,
    "jan": 1,
    "febrero": 2,
    "feb": 2,
    "february": 2,
    "marzo": 3,
    "mar": 3,
    "march": 3,
    "abril": 4,
    "abr": 4,
    "april": 4,
    "mayo": 5,
    "may": 5,
    "junio": 6,
    "jun": 6,
    "june": 6,
    "julio": 7,
    "jul": 7,
    "july": 7,
    "agosto": 8,
    "ago": 8,
    "august": 8,
    "septiembre": 9,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "octubre": 10,
    "oct": 10,
    "october": 10,
    "noviembre": 11,
    "nov": 11,
    "november": 11,
    "diciembre": 12,
    "dic": 12,
    "december": 12,
    "dec": 12,
}

CURRENT_WORDS = {"actualidad", "presente", "actual", "current", "hoy", "vigente"}


def parse_month(text: str) -> int | None:
    return MONTH_MAP.get(text.strip().lower())


def to_date_dict(day=None, month=None, year=None) -> dict | None:
    if day is None and month is None and year is None:
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

    # "month year - month year" or "month year a month year"
    if m := re.match(
        r"(\w+)\s+(\d{4})\s*(?:-|–|a|al|hasta)\s*(\w+)\s+(\d{4})", text, re.IGNORECASE
    ):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(2)),
            to_date_dict(month=parse_month(m.group(3)), year=m.group(4)),
        )

    # "month year - current/present"
    if m := re.match(
        r"(\w+)\s+(\d{4})\s*[-–]\s*(actualidad|presente|actual|hoy|vigente)",
        text,
        re.IGNORECASE,
    ):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(2)),
            "Current",
        )

    # "year-year"
    if m := re.match(r"^(\d{4})\s*[-–]\s*(\d{4})$", text):
        return to_date_dict(year=m.group(1)), to_date_dict(year=m.group(2))

    # "month a month year" -> "March to November 2021"
    if m := re.match(r"(\w+)\s+a\s+(\w+)\s+(\d{4})", text, re.IGNORECASE):
        return (
            to_date_dict(month=parse_month(m.group(1)), year=m.group(3)),
            to_date_dict(month=parse_month(m.group(2)), year=m.group(3)),
        )

    # "month year" only
    if m := re.match(r"^(\w+)\s+(\d{4})$", text, re.IGNORECASE):
        return to_date_dict(month=parse_month(m.group(1)), year=m.group(2)), None

    # year only
    if m := re.match(r"^(\d{4})$", text):
        return to_date_dict(year=m.group(1)), None

    return None, None


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_experience_section(full_text: str, max_chars: int = 5000) -> str:
    exp_headers = re.compile(
        r"(experiencia\s+(?:laboral|como|profesional|de\s+trabajo)|work\s+experience)",
        re.IGNORECASE,
    )
    exp_end = re.compile(
        r"\n\s*(titulaci[oó]n\s+acad|formaci[oó]n\s+acad|certificacion|"
        r"competencia\s+profesion|referencias\s+person|aptitudes|"
        r"habilidades\s*\n|idiomas\s*\n|cursos\s+y\s+|educaci[oó]n\s*\n|"
        r"principales\s+pericias)",
        re.IGNORECASE,
    )

    if start_match := exp_headers.search(full_text):
        start_idx = start_match.start()
        text_from_exp = full_text[start_idx:]
        if end_match := exp_end.search(text_from_exp[50:]):
            section = text_from_exp[: 50 + end_match.start()]
        else:
            section = text_from_exp
        return section[:max_chars] if len(section) > 200 else full_text[:max_chars]

    return full_text[:max_chars]


def call_ollama(prompt: str, max_tokens: int = 2000) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            "num_ctx": 4096,
        },
    }
    full_response = ""
    with httpx.Client(timeout=180.0) as client:
        with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    full_response += data.get("response", "")
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    return full_response.strip()


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
    response = call_ollama(prompt, max_tokens=5).strip().upper()
    return any(w in response for w in ["SI", "SÍ", "YES", "Y"])


EXTRACTION_PROMPT = """Extract ALL work experience from this CV. Return ONLY a valid JSON array, no extra text, no markdown.

CRITICAL RULES:
1. job_title = exact position title. NEVER company name.
2. company = full organization name, without "En" prefix. NO truncation.
3. ALWAYS separate start_date and end_date as objects with day, month, year:
   - "2014-2016" → start:{day:null, month:null, year:2014} end:{day:null, month:null, year:2016}
   - "may 2022" → start:{day:null, month:5, year:2022} end:null
   - "march 2017 to june 2017" → start:{day:null, month:3, year:2017} end:{day:null, month:6, year:2017}
   - "15/03/2020 - 20/08/2022" → start:{day:15, month:3, year:2020} end:{day:20, month:8, year:2022}
   - "present/current" → end:"Current"
4. If no date identifiable, use {day:null, month:null, year:null}
5. DO NOT invent dates. If not in text, use null.
6. Include: jobs, internships, consulting, project management
7. Exclude: courses, certifications, academic education, volunteering without dates

MONTH MAP (number):
1=january, 2=february, 3=march, 4=april, 5=may, 6=june, 7=july, 8=august, 9=september, 10=october, 11=november, 12=december

EXACT FORMAT (each date is object with day, month, year):
[{"job_title":"...","company":"...","start_date":{"day":null,"month":5,"year":2022},"end_date":{"day":null,"month":null,"year":2024}}]

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


def extract_with_regex(text: str) -> list:
    results = []
    seen = set()

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


def extract_work_experience(full_text: str) -> list:
    section = extract_experience_section(full_text, max_chars=5000)

    prompt = EXTRACTION_PROMPT.replace("{cv_text}", section)
    response = call_ollama(prompt, max_tokens=2000)
    items = parse_json_response(response)
    results = clean_items(items)

    # Complement with regex for entries model might have missed
    regex_items = extract_with_regex(section)
    regex_cleaned = clean_items(regex_items)

    existing_keys = {
        (r["job_title"].lower()[:30], r["company"].lower()[:30]) for r in results
    }
    for item in regex_cleaned:
        key = (item["job_title"].lower()[:30], item["company"].lower()[:30])
        if key not in existing_keys:
            results.append(item)
            existing_keys.add(key)

    return results


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

        experiencia = extract_work_experience(text)

        return JSONResponse(
            content={
                "es_cv": True,
                "file": file.filename,
                "total_experiencias": len(experiencia),
                "experiencia_laboral": experiencia,
            }
        )

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {"status": "ok", "ollama": "connected", "available_models": models}
    except Exception as e:
        return {"status": "error", "ollama": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
