import re

import pdfplumber


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
