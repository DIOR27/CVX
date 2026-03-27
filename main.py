import os
import tempfile

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import (
    APP_TITLE,
    APP_VERSION,
    EXPERIENCE_SECTION_MAX_CHARS,
    MODEL,
    OLLAMA_URL,
    PDF_LEFT_CROP_RATIO,
    PDF_RIGHT_CROP_START_RATIO,
)
from app.date_parsing import (
    CURRENT_WORDS,
    LONG_DATE_PATTERN,
    MONTH_ALIASES,
    convert_date_value,
    extract_date_range_from_text,
    normalize_text,
    parse_date_string,
    parse_month,
    sanitize_date_text,
    to_date_dict,
)
from app.extraction import (
    CV_KEYWORDS,
    clean_company_name,
    clean_items,
    clean_job_title,
    extract_consulting_entries,
    extract_experience_blocks,
    extract_local_candidates,
    extract_structured_role_entries,
    extract_with_regex,
    extract_work_experience,
    is_suspicious_item,
    looks_like_company_line,
    looks_like_job_title_line,
    looks_like_section_end,
    parse_company_date_line,
    refine_job_title,
    score_items,
)
from app.ollama_client import (
    call_ollama,
    extraction_prompt,
    get_ollama_timeout,
    parse_json_response,
    verify_is_cv,
)
from app.pdf_processing import extract_experience_section, extract_text_from_pdf

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


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

        if not verify_is_cv(text, CV_KEYWORDS):
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

        local_left = extract_local_candidates(left_text)
        local_right = extract_local_candidates(right_text)
        local_full = extract_local_candidates(text)

        candidates = [
            (score_items(local_left), local_left, left_text),
            (score_items(local_right), local_right, right_text),
            (score_items(local_full), local_full, text),
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, experience, selected_text = candidates[0]

        if not experience or score_items(experience) < (len(experience) * 8):
            remote_source = selected_text if best_score > 0 else text
            experience = extract_work_experience(remote_source, allow_remote_refine=True)

        return JSONResponse(
            content={
                "es_cv": True,
                "file": file.filename,
                "total_experiencias": len(experience),
                "experiencia_laboral": experience,
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
            response = client.get(tags_url)
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
        return {
            "status": "ok",
            "ollama": "connected",
            "ollama_url": OLLAMA_URL,
            "configured_model": MODEL,
            "available_models": models,
        }
    except Exception as exc:
        return {
            "status": "error",
            "ollama_url": OLLAMA_URL,
            "configured_model": MODEL,
            "ollama": str(exc),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
