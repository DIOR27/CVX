import httpx
import json
import re
import pdfplumber
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="CV Analyzer API", version="4.0.0")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

# ---------------------------------------------------------------------------
# MAPEO DE MESES
# ---------------------------------------------------------------------------
MESES = {
    "enero": "Enero", "ene": "Enero",
    "febrero": "Febrero", "feb": "Febrero",
    "marzo": "Marzo", "mar": "Marzo",
    "abril": "Abril", "abr": "Abril",
    "mayo": "Mayo", "may": "Mayo",
    "junio": "Junio", "jun": "Junio",
    "julio": "Julio", "jul": "Julio",
    "agosto": "Agosto", "ago": "Agosto",
    "septiembre": "Septiembre", "sep": "Septiembre", "sept": "Septiembre",
    "octubre": "Octubre", "oct": "Octubre",
    "noviembre": "Noviembre", "nov": "Noviembre",
    "diciembre": "Diciembre", "dic": "Diciembre",
}

def normalizar_mes(texto: str) -> str:
    """Convierte abreviatura o nombre de mes a forma capitalizada."""
    t = texto.strip().lower()
    return MESES.get(t, texto.strip().capitalize())

def parsear_fecha(texto: str):
    """
    Recibe un string de fecha y devuelve (fecha_inicio, fecha_fin).
    Maneja todos los patrones encontrados en los 3 CVs.
    """
    if not texto:
        return None, None

    t = texto.strip()

    # Patrón: "mes año - mes año"  o  "mes año a mes año"
    m = re.match(
        r'(\w+)\s+(\d{4})\s*(?:-|a|al|hasta)\s*(\w+)\s+(\d{4})',
        t, re.IGNORECASE
    )
    if m:
        return f"{normalizar_mes(m.group(1))} {m.group(2)}", f"{normalizar_mes(m.group(3))} {m.group(4)}"

    # Patrón: "mes año - actualidad/presente"
    m = re.match(
        r'(\w+)\s+(\d{4})\s*[-–]\s*(actualidad|presente|actual|hoy|vigente)',
        t, re.IGNORECASE
    )
    if m:
        return f"{normalizar_mes(m.group(1))} {m.group(2)}", "Actualidad"

    # Patrón: "mes año - presente" abreviado (may 2024 - presente)
    m = re.match(
        r'(\w{3,})\s+(\d{4})\s*[-–]\s*(presente|actualidad|actual)',
        t, re.IGNORECASE
    )
    if m:
        return f"{normalizar_mes(m.group(1))} {m.group(2)}", "Actualidad"

    # Patrón: "año-año" o "año - año"
    m = re.match(r'(\d{4})\s*[-–]\s*(\d{4})', t)
    if m:
        return m.group(1), m.group(2)

    # Patrón: "mes a mes año" → "Marzo a Noviembre 2021"
    m = re.match(r'(\w+)\s+a\s+(\w+)\s+(\d{4})', t, re.IGNORECASE)
    if m:
        anio = m.group(3)
        return f"{normalizar_mes(m.group(1))} {anio}", f"{normalizar_mes(m.group(2))} {anio}"

    # Patrón: "mes año" solo
    m = re.match(r'(\w+)\s+(\d{4})$', t, re.IGNORECASE)
    if m:
        return f"{normalizar_mes(m.group(1))} {m.group(2)}", None

    # Solo año
    m = re.match(r'^(\d{4})$', t)
    if m:
        return m.group(1), None

    return t, None  # Devolver tal cual si no matchea nada


# ---------------------------------------------------------------------------
# 1. EXTRACCIÓN DE TEXTO
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_experience_section(full_text: str, max_chars: int = 5000) -> str:
    """Extrae solo la sección de experiencia laboral."""
    EXP_HEADERS = re.compile(
        r'(experiencia\s+(?:laboral|como|profesional|de\s+trabajo)|work\s+experience)',
        re.IGNORECASE
    )
    EXP_END = re.compile(
        r'\n\s*(titulaci[oó]n\s+acad|formaci[oó]n\s+acad|certificacion|'
        r'competencia\s+profesion|referencias\s+person|aptitudes|'
        r'habilidades\s*\n|idiomas\s*\n|cursos\s+y\s+|educaci[oó]n\s*\n|'
        r'principales\s+pericias)',
        re.IGNORECASE
    )

    start_match = EXP_HEADERS.search(full_text)
    if not start_match:
        return full_text[:max_chars]

    start_idx = start_match.start()
    text_from_exp = full_text[start_idx:]

    end_match = EXP_END.search(text_from_exp[50:])
    if end_match:
        section = text_from_exp[:50 + end_match.start()]
    else:
        section = text_from_exp

    return section[:max_chars] if len(section) > 200 else full_text[:max_chars]


# ---------------------------------------------------------------------------
# 2. LLAMADA A OLLAMA
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, max_tokens: int = 2000) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            "num_ctx": 4096,
        }
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


# ---------------------------------------------------------------------------
# 3. VERIFICACIÓN: ¿ES UN CV?
# ---------------------------------------------------------------------------

CV_KEYWORDS = [
    "experiencia laboral", "formación académica", "hoja de vida",
    "currículum", "curriculum", "datos personales", "perfil profesional",
    "sobre mí", "referencias", "aptitudes", "habilidades",
    "desarrollador", "ingeniero", "licenciado", "analista", "programador",
    "gerente", "director", "experience", "education", "skills", "resume",
    "employment", "work experience", "professional profile",
]

def verify_is_cv(text: str) -> bool:
    text_lower = text.lower()
    matches = sum(1 for kw in CV_KEYWORDS if kw in text_lower)
    if matches >= 2:
        return True
    prompt = f'¿Es este texto una hoja de vida? Responde solo "SI" o "NO":\n\n{text[:500]}\n\nRespuesta:'
    response = call_ollama(prompt, max_tokens=5).strip().upper()
    return any(w in response for w in ["SI", "SÍ", "YES"])


# ---------------------------------------------------------------------------
# 4. EXTRACCIÓN HÍBRIDA: REGEX PRIMERO, MODELO COMO RESPALDO
# ---------------------------------------------------------------------------

# Patrón para CVs tipo "Cargo, año-año, Empresa" (estilo Iván Andrade)
# Ej: "Gerente de Contrato, 2014-2016 en (CFE) la Comisión Federal..."
REGEX_PATTERN_NARRATIVE = re.compile(
    r'^([A-ZÁÉÍÓÚÑÜ][^,\n]{3,60}?),?\s*'           # Cargo (empieza con mayúscula)
    r'(\d{4}[-–]\d{4}|'                              # fecha año-año
    r'\w+\s+a\s+\w+\s+(?:de\s+)?\d{4}|'             # mes a mes año
    r'\w+\s+\d{4}\s*[-–a]\s*\w+\s+\d{4}|'           # mes año - mes año
    r'\w+\s+\d{4}|'                                  # mes año
    r'\d{4})'                                        # solo año
    r'[,\s]+(?:en\s+)?(.+?)(?:\.|$)',                # empresa
    re.MULTILINE | re.IGNORECASE
)

def extract_with_regex(text: str) -> list:
    """
    Extracción por regex para CVs con experiencia narrada en párrafos.
    Patrón: 'Cargo, fecha, en Empresa descripción...'
    """
    results = []
    seen = set()

    # Patrones específicos del formato narrativo de Iván Andrade
    patterns = [
        # "Director de Proyecto, 2009, Constructora MALDONADO FIALLO..."
        re.compile(
            r'\b(Director[^,\n]{0,60}?|Gerente[^,\n]{0,60}?|Especialista[^,\n]{0,60}?|'
            r'Asesor[^,\n]{0,60}?|Director\s+T[eé]cnico[^,\n]{0,40}?)'
            r',\s*'
            r'(\d{4}[-–]\d{4}|\w+\s+a\s+\w+\s+(?:de\s+)?\d{4}|'
            r'\w+\s+\d{4}\s*[-–]\s*\w+\s+\d{4}|\d{4})'
            r',?\s*(?:en\s+)?'
            r'([A-ZÁÉÍÓÚÑÜ][^\n.]{5,80})',
            re.IGNORECASE
        ),
    ]

    for pattern in patterns:
        for m in pattern.finditer(text):
            cargo = m.group(1).strip().rstrip(',')
            fecha_raw = m.group(2).strip()
            empresa_raw = m.group(3).strip()

            # Limpiar empresa (quitar texto después de punto o paréntesis largo)
            empresa = re.split(r'\s*[\(\[]|\s{2,}', empresa_raw)[0].strip()
            empresa = re.sub(r'\.$', '', empresa).strip()

            # Parsear fechas
            f_inicio, f_fin = parsear_fecha(fecha_raw)

            key = (cargo.lower()[:30], empresa.lower()[:30])
            if key not in seen and len(cargo) > 3 and len(empresa) > 3:
                seen.add(key)
                results.append({
                    "cargo": cargo,
                    "empresa": empresa,
                    "fecha_inicio": f_inicio,
                    "fecha_fin": f_fin,
                })

    return results


EXTRACTION_PROMPT = """Extrae TODA la experiencia laboral de este CV. Devuelve SOLO un JSON array válido, sin texto extra, sin markdown.

REGLAS CRÍTICAS:
1. cargo = título exacto del puesto. NUNCA nombre de empresa.
2. empresa = nombre completo de la organización, sin "En" al inicio. NO truncar.
3. Si cargo tiene "|", crear DOS entradas con misma empresa y fechas.
4. SIEMPRE separar fecha_inicio y fecha_fin:
   "2014-2016" → inicio:"2014" fin:"2016"
   "2012-2014" → inicio:"2012" fin:"2014"  
   "2010-2012" → inicio:"2010" fin:"2012"
   "2009-2010" → inicio:"2009" fin:"2010"
   "2002-2004" → inicio:"2002" fin:"2004"
   "1998-2002" → inicio:"1998" fin:"2002"
   "mayo a sep 2022" → inicio:"Mayo 2022" fin:"Septiembre 2022"
   "may 2024 - presente" → inicio:"Mayo 2024" fin:"Actualidad"
   "marzo 2017 a junio 2017" → inicio:"Marzo 2017" fin:"Junio 2017"
   "abril a octubre de 2016" → inicio:"Abril 2016" fin:"Octubre 2016"
   "2009" (solo año) → inicio:"2009" fin:null
5. Si NO hay fecha identificable para una entrada, usar fecha_inicio:null fecha_fin:null
6. NO inventar fechas. Si no está en el texto, es null.
7. Incluir: empleos, pasantías, consultoría, dirección de proyectos, peritajes
8. Excluir: cursos, certificaciones, formación académica, voluntariado sin fecha

FORMATO EXACTO:
[{"cargo":"...","empresa":"...","fecha_inicio":"...","fecha_fin":"..."}]

CV:
{cv_text}

JSON:"""


def parse_json_response(response: str) -> list:
    response = re.sub(r"```(?:json)?\s*", "", response).strip()

    m = re.search(r"\[.*\]", response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            for key in ["experiencia", "experience", "experiencia_laboral", "jobs"]:
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


NORMALIZAR_FIN = {"actualidad", "presente", "actual", "present",
                  "current", "a la fecha", "hoy", "today", "vigente"}

def clean_and_fix_dates(items: list) -> list:
    """Limpia empresas, normaliza fechas y divide cargos con |"""
    result = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        cargo   = (item.get("cargo") or "").strip()
        empresa = (item.get("empresa") or "").strip()
        f_ini   = (item.get("fecha_inicio") or "").strip() or None
        f_fin   = (item.get("fecha_fin") or "").strip() or None

        # Limpiar "En " al inicio
        for prefix in ["En ", "en "]:
            if empresa.startswith(prefix):
                empresa = empresa[len(prefix):]
            if cargo.startswith(prefix):
                cargo = cargo[len(prefix):]

        # Si fecha_inicio contiene un rango "XXXX-XXXX", separar
        if f_ini and not f_fin:
            fi2, ff2 = parsear_fecha(f_ini)
            if ff2:  # Se encontró un rango
                f_ini, f_fin = fi2, ff2

        # Normalizar "presente/actualidad"
        if f_fin and f_fin.lower() in NORMALIZAR_FIN:
            f_fin = "Actualidad"

        # Dividir cargo con "|"
        cargos = [c.strip() for c in cargo.split("|")] if "|" in cargo else [cargo]

        for c in cargos:
            if not c or not empresa:
                continue
            key = (c.lower()[:40], empresa.lower()[:40], str(f_ini))
            if key not in seen:
                seen.add(key)
                result.append({
                    "cargo": c,
                    "empresa": empresa,
                    "fecha_inicio": f_ini,
                    "fecha_fin": f_fin,
                })

    return result


# ---------------------------------------------------------------------------
# 5. EXTRACCIÓN PRINCIPAL
# ---------------------------------------------------------------------------

def extract_work_experience(full_text: str) -> list:
    section = extract_experience_section(full_text, max_chars=5000)

    # Paso 1: intentar extracción con el modelo
    prompt = EXTRACTION_PROMPT.replace("{cv_text}", section)
    response = call_ollama(prompt, max_tokens=2000)
    items = parse_json_response(response)
    results = clean_and_fix_dates(items)

    # Paso 2: complementar con regex para entradas que el modelo haya perdido
    regex_items = extract_with_regex(section)
    regex_cleaned = clean_and_fix_dates(regex_items)

    # Merge: agregar entradas de regex que no estén ya en los resultados del modelo
    existing_keys = {
        (r["cargo"].lower()[:30], r["empresa"].lower()[:30])
        for r in results
    }
    for item in regex_cleaned:
        key = (item["cargo"].lower()[:30], item["empresa"].lower()[:30])
        if key not in existing_keys:
            results.append(item)
            existing_keys.add(key)

    return results


# ---------------------------------------------------------------------------
# 6. ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/analizar-cv")
async def analizar_cv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)

        if not text or len(text) < 50:
            raise HTTPException(
                status_code=422,
                detail="No se pudo extraer texto del PDF. Puede ser un PDF escaneado sin OCR."
            )

        if not verify_is_cv(text):
            return JSONResponse(content={
                "es_cv": False,
                "mensaje": "El documento no parece ser una hoja de vida o CV.",
                "experiencia_laboral": []
            })

        experiencia = extract_work_experience(text)

        return JSONResponse(content={
            "es_cv": True,
            "archivo": file.filename,
            "total_experiencias": len(experiencia),
            "experiencia_laboral": experiencia
        })

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {"status": "ok", "ollama": "conectado", "modelos_disponibles": models}
    except Exception as e:
        return {"status": "error", "ollama": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
