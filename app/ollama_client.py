import json
import re

import httpx

from app.config import (
    EXTRACTION_MAX_TOKENS,
    MODEL,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_EXTRACTION_READ_TIMEOUT,
    OLLAMA_NUM_CTX,
    OLLAMA_READ_TIMEOUT,
    OLLAMA_URL,
    OLLAMA_WRITE_TIMEOUT,
    VERIFY_MAX_TOKENS,
)


def get_ollama_timeout(read_timeout: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(
        connect=OLLAMA_CONNECT_TIMEOUT,
        read=read_timeout or OLLAMA_READ_TIMEOUT,
        write=OLLAMA_WRITE_TIMEOUT,
        pool=OLLAMA_CONNECT_TIMEOUT,
    )


def call_ollama(
    prompt: str,
    max_tokens: int = EXTRACTION_MAX_TOKENS,
    read_timeout: float | None = None,
) -> str:
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


def parse_json_response(response: str) -> list:
    response = re.sub(r"```(?:json)?\s*", "", response).strip()

    if match := re.search(r"\[.*\]", response, re.DOTALL):
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    if match := re.search(r"\{.*\}", response, re.DOTALL):
        try:
            obj = json.loads(match.group())
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


def verify_is_cv(text: str, cv_keywords: list[str]) -> bool:
    text_lower = text.lower()
    matches = sum(1 for kw in cv_keywords if kw in text_lower)
    if matches >= 2:
        return True
    prompt = (
        f'Is this text a resume? Answer only "YES" or "NO":\n\n{text[:500]}\n\nAnswer:'
    )
    response = call_ollama(prompt, max_tokens=VERIFY_MAX_TOKENS).strip().upper()
    return any(word in response for word in ["SI", "SÍ", "YES", "Y"])


def extraction_prompt(cv_text: str) -> str:
    return f"""Extrae experiencias laborales del CV y responde SOLO con un arreglo JSON valido.
Cada item debe tener:
job_title, company, start_date, end_date

Reglas:
- job_title es el cargo, no la empresa.
- company es la organizacion.
- Usa fechas como {{"day":null,"month":5,"year":2022}}
- Si termina en presente/actualidad/current usa "Current"
- No inventes informacion.
- Excluye educacion, cursos, certificaciones y habilidades.

Formato:
[{{"job_title":"...","company":"...","start_date":{{"day":null,"month":5,"year":2022}},"end_date":"Current"}}]

CV:
{cv_text}
JSON:"""
