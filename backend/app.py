"""Flask backend entrypoint for text and image search."""

from __future__ import annotations

import os
import sys
import re
import csv
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.face_search import (
    create_face_models,
    load_documents as load_face_documents,
    load_face_index,
    load_face_meta,
    search_face,
)
from retrieval.text_search import _load_documents, _load_index, search, search_text_documents
from retrieval.web_search import WebIndexCache, search_web_documents
from backend.services.llm_service import (
    GroqError,
    generate_placement_response,
    generate_response,
    generate_scholarship_response,
    generate_web_response,
    parse_prompt_to_filters,
)


STATIC_DIR = ROOT / "frontend" / "web"
IMAGES_DIR = ROOT / "faculty_images"
SCHOLARSHIP_IMAGES_DIR = ROOT / "scholarship_images"
PLACEMENT_DOCS_PATH = ROOT / "processed" / "placement_documents.json"
SCHOLARSHIP_DOCS_PATH = ROOT / "processed" / "scholarship_documents.json"
PLACEMENT_INDEX_PATH = ROOT / "vector_db" / "faiss_placement.index"
SCHOLARSHIP_INDEX_PATH = ROOT / "vector_db" / "faiss_scholarship.index"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

    documents = _load_documents(ROOT / "processed" / "faculty_documents.json")
    text_index = _load_index(ROOT / "vector_db" / "faiss_text.index")

    placement_documents: list[Dict[str, Any]] = []
    scholarship_documents: list[Dict[str, Any]] = []
    placement_index = None
    scholarship_index = None
    if PLACEMENT_DOCS_PATH.exists():
        placement_documents = _load_documents(PLACEMENT_DOCS_PATH)
    if SCHOLARSHIP_DOCS_PATH.exists():
        scholarship_documents = _load_documents(SCHOLARSHIP_DOCS_PATH)
    if PLACEMENT_INDEX_PATH.exists():
        placement_index = _load_index(PLACEMENT_INDEX_PATH)
    if SCHOLARSHIP_INDEX_PATH.exists():
        scholarship_index = _load_index(SCHOLARSHIP_INDEX_PATH)
    web_cache = WebIndexCache()

    placements_path = ROOT / "data" / "placement_details.csv"
    scholarships_path = ROOT / "data" / "scholarship_details.json"
    placement_records: list[Dict[str, str]] = []
    scholarship_records: list[Dict[str, Any]] = []

    if placements_path.exists():
        with placements_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                placement_records.append(
                    {
                        "student_name": (row.get("Student_Name") or "").strip(),
                        "enrollment_number": (row.get("Enrollment_Number") or "").strip(),
                        "discipline": (row.get("Discipline") or "").strip(),
                        "year_of_passing": (row.get("Year_of_Passing") or "").strip(),
                        "on_off_campus": (row.get("On_Off_Campus") or "").strip(),
                        "employer": (row.get("Employer") or "").strip(),
                        "academic_year": (row.get("Academic_Year") or "").strip(),
                    }
                )

    if scholarships_path.exists():
        with scholarships_path.open("r", encoding="utf-8") as handle:
            scholarship_records = json.load(handle)

    face_documents = load_face_documents(ROOT / "processed" / "faculty_documents.json")
    face_index = None
    face_meta = None
    mtcnn = None
    resnet = None
    device = None
    image_lookup = {}
    if IMAGES_DIR.exists():
        for path in IMAGES_DIR.iterdir():
            if path.is_file():
                stem = path.stem.lower()
                image_lookup[stem] = path.name

    def _ensure_face_resources() -> None:
        nonlocal face_index, face_meta, mtcnn, resnet, device
        face_index_path = ROOT / "vector_db" / "faiss_face.index"
        face_meta_path = ROOT / "embeddings" / "face_embeddings_meta.json"
        if not face_index_path.exists() or not face_meta_path.exists():
            raise FileNotFoundError(
                "Face index or metadata not found. Run: python main.py build (or python main.py build-faces)"
            )
        if face_index is None:
            face_index = load_face_index(face_index_path)
        if face_meta is None:
            face_meta = load_face_meta(face_meta_path)
        if mtcnn is None or resnet is None:
            mtcnn, resnet, device = create_face_models()

    def _find_image_for_id(faculty_id: Optional[str]) -> Optional[str]:
        if not faculty_id:
            return None
        raw_id = str(faculty_id).strip()
        candidates = [raw_id]
        try:
            id_int = int(raw_id)
            candidates.extend(
                [
                    f"{id_int:03d}",
                    f"{id_int:04d}",
                    f"fac{id_int:04d}",
                ]
            )
        except ValueError:
            pass
        for candidate in candidates:
            filename = image_lookup.get(candidate.lower())
            if filename:
                return filename
        return None

    def _vector_candidates(
        prompt_text: str,
        docs: list[Dict[str, Any]],
        index: Any,
        model_name: str,
        top_k: int,
    ) -> list[Dict[str, Any]]:
        if not docs:
            return []
        if not prompt_text.strip() or index is None:
            return docs if top_k is None else docs[:top_k]
        try:
            return search_text_documents(
                query=prompt_text,
                documents=docs,
                index=index,
                model_name=model_name,
                top_k=top_k,
            )
        except Exception:  # noqa: BLE001
            return docs if top_k is None else docs[:top_k]

    def _normalize_name_text(value: str) -> str:
        text = value.lower()
        text = re.sub(r"\b(mr|mrs|ms|dr|prof|professor)\.?\b", " ", text)
        text = re.sub(
            r"\b(ph\.?d|m\.?tech|mtech|m\.?e|me|m\.?sc|msc|b\.?tech|btech|b\.?e|be)\b",
            " ",
            text,
        )
        text = re.sub(r"[^a-z\s]", " ", text)
        return " ".join(text.split())

    def _name_tokens(value: str) -> list[str]:
        tokens = [token for token in _normalize_name_text(value).split() if len(token) > 1]
        return tokens

    def _doc_name_match(name: str) -> bool:
        normalized = _normalize_name_text(name)
        return any(_normalize_name_text(str(doc.get("name"))) == normalized for doc in documents)

    def _match_name_prompt(prompt_text: str, doc_name: str) -> bool:
        if not doc_name:
            return False
        prompt_norm = _normalize_name_text(prompt_text)
        if not prompt_norm:
            return False
        stopwords = {
            "faculty",
            "teacher",
            "teachers",
            "professor",
            "assistant",
            "associate",
            "department",
            "university",
            "of",
            "from",
            "with",
            "in",
            "for",
            "degree",
            "highest",
            "experience",
            "joining",
            "date",
            "cse",
            "cs",
            "ece",
            "ee",
            "it",
            "ash",
            "ju",
        }
        prompt_tokens = [
            token for token in prompt_norm.split() if token not in stopwords
        ]
        prompt_initials = {token for token in prompt_tokens if len(token) == 1}

        name_tokens = _name_tokens(doc_name)
        if not name_tokens:
            return False
        first = name_tokens[0]
        last = name_tokens[-1]

        if len(prompt_tokens) >= 2:
            if all(token in prompt_tokens for token in name_tokens):
                return True
            if last in prompt_tokens and prompt_initials and first[0] in prompt_initials:
                return True
            return False

        if len(prompt_tokens) == 1:
            token = prompt_tokens[0]
            return token == first or token == last

        return False

    def _find_name_matches(prompt_text: str) -> list[Dict[str, Any]]:
        matches: list[Dict[str, Any]] = []
        for doc in documents:
            if _match_name_prompt(prompt_text, str(doc.get("name", ""))):
                entry = dict(doc)
                filename = _find_image_for_id(entry.get("id"))
                if filename:
                    entry["image_url"] = f"/images/{filename}"
                matches.append(entry)
        return matches

    def _extract_name_from_prompt(prompt_text: str) -> str:
        prompt_norm = _normalize_name_text(prompt_text)
        if not prompt_norm:
            return ""
        prompt_tokens = set(_name_tokens(prompt_text))
        best_match = ""
        best_score = 0
        for doc in documents:
            name = doc.get("name")
            if not name:
                continue
            name_tokens = _name_tokens(str(name))
            if not name_tokens:
                continue
            if all(token in prompt_tokens for token in name_tokens):
                score = len(name_tokens)
                if score > best_score:
                    best_score = score
                    best_match = str(name)
                    continue
            name_norm = _normalize_name_text(str(name))
            if name_norm and name_norm in prompt_norm and best_score == 0:
                best_match = str(name)
        return best_match

    def _normalize_field_text(value: str) -> str:
        text = value.lower().strip()
        text = re.sub(r"[^a-z0-9\s&.]", " ", text)
        return " ".join(text.split())

    def _placement_name_tokens(value: str) -> list[str]:
        return [token for token in _normalize_name_text(value).split() if len(token) > 1]

    def _match_person_tokens(prompt_tokens: list[str], name_tokens: list[str]) -> bool:
        if not prompt_tokens or not name_tokens:
            return False
        if len(prompt_tokens) >= 2:
            return all(token in name_tokens for token in prompt_tokens)
        token = prompt_tokens[0]
        return token == name_tokens[0] or token == name_tokens[-1]

    def _extract_years_from_prompt(prompt_text: str) -> Optional[str]:
        match = re.search(r"\b(20\d{2})\b", prompt_text)
        if match:
            return match.group(1)
        return None

    def _is_web_intent(prompt_text: str) -> bool:
        prompt_lower = prompt_text.lower()
        keywords = [
            "admission",
            "admissions",
            "apply",
            "application",
            "eligibility",
            "intake",
            "course",
            "courses",
            "program",
            "programs",
            "curriculum",
            "syllabus",
            "b.tech",
            "m.tech",
            "mca",
            "bba",
            "bca",
            "fee",
            "fees",
            "tuition",
            "timetable",
            "time table",
            "routine",
            "calendar",
            "holiday",
            "library",
            "hostel",
            "facility",
            "facilities",
            "central facilities",
            "laboratory",
            "lab",
            "sports",
            "canteen",
            "medical",
            "contact",
            "address",
            "phone",
            "email",
        ]
        return any(keyword in prompt_lower for keyword in keywords)

    placement_employers = sorted({p.get("employer", "") for p in placement_records if p.get("employer")})
    placement_disciplines = sorted({p.get("discipline", "") for p in placement_records if p.get("discipline")})
    scholarship_states = sorted(
        {
            (s.get("state") or "").strip()
            for s in scholarship_records
            if isinstance(s, dict)
        }
    )

    def _extract_employer_from_prompt(prompt_text: str) -> str:
        prompt_norm = _normalize_field_text(prompt_text)
        prompt_key = prompt_norm.replace(" ", "")
        for employer in sorted(placement_employers, key=len, reverse=True):
            if not employer:
                continue
            employer_norm = _normalize_field_text(employer)
            employer_key = employer_norm.replace(" ", "")
            if employer_key and employer_key in prompt_key:
                return employer
        return ""

    def _extract_discipline_from_prompt(prompt_text: str) -> str:
        prompt_lower = prompt_text.lower()
        for abbrev, full in {
            "cse": "CSE",
            "ece": "ECE",
            "ee": "EE",
            "it": "IT",
        }.items():
            if re.search(rf"\b{abbrev}\b", prompt_lower):
                return full
        for discipline in placement_disciplines:
            if discipline and re.search(rf"\b{re.escape(discipline.lower())}\b", prompt_lower):
                return discipline
        return ""

    def _search_placements(
        prompt_text: str,
        records: Optional[list[Dict[str, Any]]] = None,
    ) -> list[Dict[str, Any]]:
        prompt_norm = _normalize_field_text(prompt_text)
        stopwords = {
            "show",
            "list",
            "all",
            "available",
            "record",
            "records",
            "detail",
            "details",
            "placement",
            "placements",
            "placed",
            "student",
            "students",
            "from",
            "in",
            "of",
            "by",
            "at",
            "for",
            "company",
            "employer",
            "discipline",
            "department",
            "year",
            "academic",
            "on",
            "off",
            "campus",
            "cse",
            "ece",
            "ee",
            "it",
            "computer",
            "science",
            "engineering",
            "electronics",
            "communication",
            "electrical",
            "information",
            "technology",
            "applied",
            "humanities",
        }
        prompt_tokens = [
            token
            for token in prompt_norm.split()
            if token not in stopwords and not token.isdigit()
        ]

        employer = _extract_employer_from_prompt(prompt_text)
        discipline = _extract_discipline_from_prompt(prompt_text)
        year = _extract_years_from_prompt(prompt_text)
        on_off = ""
        if "on campus" in prompt_norm or "on-campus" in prompt_norm:
            on_off = "On"
        if "off campus" in prompt_norm or "off-campus" in prompt_norm:
            on_off = "Off"

        if employer:
            employer_norm = _normalize_field_text(employer)
            employer_tokens = set(employer_norm.split())
            employer_key = employer_norm.replace(" ", "")
            if employer_tokens:
                prompt_tokens = [
                    token
                    for token in prompt_tokens
                    if token not in employer_tokens and token != employer_key
                ]

        if discipline:
            discipline_tokens = set(_normalize_field_text(discipline).split())
            if discipline_tokens:
                prompt_tokens = [
                    token for token in prompt_tokens if token not in discipline_tokens
                ]

        results = []
        source_records = records if records is not None else placement_records
        for record in source_records:
            if employer and employer.lower().replace(" ", "") not in record.get("employer", "").lower().replace(" ", ""):
                continue
            if discipline and discipline.lower() != record.get("discipline", "").lower():
                continue
            if year:
                year_of_passing = record.get("year_of_passing", "")
                academic_year = record.get("academic_year", "")
                if year != year_of_passing and year not in academic_year:
                    continue
            if on_off and on_off.lower() != record.get("on_off_campus", "").lower():
                continue

            name_tokens = _placement_name_tokens(record.get("student_name", ""))
            if prompt_tokens and not _match_person_tokens(prompt_tokens, name_tokens):
                continue

            results.append(
                {
                    **record,
                    "result_type": "placement",
                }
            )

        return results

    def _search_scholarships(
        prompt_text: str,
        records: Optional[list[Dict[str, Any]]] = None,
    ) -> list[Dict[str, Any]]:
        prompt_norm = _normalize_field_text(prompt_text)
        stopwords = {
            "show",
            "list",
            "all",
            "available",
            "scholarship",
            "scholarships",
            "scheme",
            "schemes",
            "grant",
            "stipend",
            "fee",
            "waiver",
            "tuition",
            "for",
            "with",
            "to",
            "of",
            "in",
            "on",
            "by",
            "student",
            "students",
        }
        tokens = {token for token in prompt_norm.split() if token not in stopwords}

        generic_alias_tokens = {
            "scholarship",
            "scheme",
            "schemes",
            "government",
            "state",
            "central",
            "india",
            "west",
            "bengal",
            "education",
            "technical",
            "professional",
            "fund",
            "relief",
        }

        prompt_keywords = {
            token
            for token in prompt_norm.split()
            if token and token not in stopwords and token not in generic_alias_tokens
        }

        target_keywords = {
            "minority",
            "sc",
            "st",
            "obc",
            "girls",
            "female",
            "women",
            "disabled",
            "specially",
            "abled",
            "muslim",
            "muslims",
            "sikh",
            "sikhs",
            "christian",
            "christians",
            "buddhist",
            "buddhists",
            "jain",
            "jains",
            "parsi",
            "parsis",
        }

        filter_keywords = {
            "maximum",
            "minimum",
            "min",
            "max",
            "total",
            "overall",
            "household",
            "family",
            "income",
            "tuition",
            "tution",
            "fee",
            "fees",
            "limit",
            "amount",
            "monthly",
            "annual",
            "per",
            "year",
            "month",
            "benefit",
            "benefits",
            "waiver",
            "eligibility",
        }

        name_keywords = {
            token
            for token in prompt_keywords
            if token not in target_keywords
            and token not in filter_keywords
            and not token.isdigit()
        }

        if records is None:
            candidate_records: list[Dict[str, Any]] = [
                record for record in scholarship_records if isinstance(record, dict)
            ]
        else:
            candidate_records = [record for record in records if isinstance(record, dict)]

        if name_keywords:
            strict_candidates: list[Dict[str, Any]] = []
            for record in candidate_records:
                name_blob = " ".join(
                    [
                        str(record.get("title", "")),
                        str(record.get("short_name", "")),
                        str(record.get("id", "")),
                    ]
                )
                name_norm = _normalize_field_text(name_blob)
                name_tokens = set(name_norm.split())
                if name_keywords.issubset(name_tokens):
                    strict_candidates.append(record)
            if not strict_candidates:
                return []
            candidate_records = strict_candidates

        type_filter = ""
        if "central" in tokens:
            type_filter = "Central"
        if "state" in tokens:
            type_filter = "State"

        state_filter = ""
        for state in scholarship_states:
            if state and state.lower() in prompt_norm:
                state_filter = state
                break

        income_match = re.search(r"(\d+[\d,]*)\s*(lakh)?", prompt_norm)
        income_limit = None
        if income_match:
            raw = income_match.group(1).replace(",", "")
            income_limit = int(raw)
            if income_match.group(2):
                income_limit *= 100000

        tuition_limit = None
        if "tuition" in prompt_norm or "tution" in prompt_norm:
            tuition_match = re.search(r"(?:tuition|tution)[^\d]*(\d+[\d,]*)", prompt_norm)
            if tuition_match:
                tuition_limit = int(tuition_match.group(1).replace(",", ""))

        fee_waiver_requested = "fee waiver" in prompt_norm or "tuition waiver" in prompt_norm
        numeric_filter_active = income_limit is not None or tuition_limit is not None
        target_filter_active = bool(target_keywords.intersection(tokens))
        type_state_filter_active = bool(type_filter or state_filter)

        results: list[Dict[str, Any]] = []
        for record in candidate_records:
            if type_filter and type_filter.lower() not in (record.get("scholarship_type", "").lower()):
                continue
            if state_filter and state_filter.lower() not in (record.get("state", "").lower()):
                continue

            blob = " ".join(
                [
                    str(record.get("title", "")),
                    str(record.get("short_name", "")),
                    str(record.get("overview", "")),
                    " ".join(record.get("keywords", []) or []),
                    " ".join(record.get("categories", []) or []),
                    " ".join(record.get("target_groups", []) or []),
                    " ".join(record.get("eligible_communities", []) or []),
                    " ".join(record.get("eligible_courses", []) or []),
                    str(record.get("provider", "")),
                ]
            ).lower()

            requested_targets = target_keywords.intersection(tokens)
            blob_tokens = set(_normalize_field_text(blob).split())
            if requested_targets and not requested_targets.intersection(blob_tokens):
                continue

            eligibility = record.get("eligibility", {}) or {}
            max_income = eligibility.get("maximum_family_income") or eligibility.get("sc_st_maximum_family_income")
            if income_limit is not None:
                if max_income is None:
                    continue
                if income_limit > int(max_income):
                    continue

            benefits = record.get("benefits", {}) or {}
            tuition_fee_limit = benefits.get("tuition_fee_limit")
            if tuition_limit is not None:
                if tuition_fee_limit is None:
                    continue
                try:
                    fee_limit_value = int(tuition_fee_limit)
                except (TypeError, ValueError):
                    continue
                if tuition_limit > fee_limit_value:
                    continue
            if fee_waiver_requested and not benefits.get("tuition_fee_waiver") and "waiver" not in blob:
                continue

            if not target_filter_active and not type_state_filter_active and not numeric_filter_active and not fee_waiver_requested:
                if tokens and not any(token in blob for token in tokens):
                    continue

            results.append(
                {
                    **record,
                    "result_type": "scholarship",
                }
            )

        return results

    def _university_key_set(value: str) -> set[str]:
        normalized = _normalize_field_text(value)
        if not normalized:
            return set()
        tokens = [
            token
            for token in normalized.replace("&", " ").split()
            if token not in {"university", "of", "the"}
        ]
        initials = "".join(token[0] for token in tokens if token)
        keys = {normalized}
        if initials:
            keys.add(initials)
        return keys

    def _university_query_keys(value: str) -> set[str]:
        normalized = _normalize_field_text(value)
        if not normalized:
            return set()
        keys = _university_key_set(normalized)
        if "calcutta" in normalized:
            keys.add("cu")
        if "jadavpur" in normalized:
            keys.add("ju")
        return keys

    university_key_index: set[str] = set()
    for doc in documents:
        university_key_index.update(_university_key_set(str(doc.get("university", ""))))

    def _extract_university_from_prompt(prompt_text: str) -> str:
        match = re.search(
            r"(?:from\s+)?university\s+(?:of\s+)?([a-zA-Z.& ]+)",
            prompt_text,
            re.IGNORECASE,
        )
        if match:
            token = match.group(1).strip()
            if token:
                return token
        match = re.search(r"from\s+([a-zA-Z.& ]+)", prompt_text, re.IGNORECASE)
        if match:
            token = match.group(1).strip()
            if "university" in token.lower():
                token = token.lower().replace("university", "").strip().title()
                return f"{token} University" if token else ""
            return token
        return ""

    def _extract_department_from_prompt(prompt_text: str, departments: List[str]) -> str:
        prompt_lower = prompt_text.lower()
        abbrev_map = {
            "cse": "Computer Science & Engineering",
            "cs": "Computer Science & Engineering",
            "ece": "Electronics & Communication Engineering",
            "ee": "Electrical Engineering",
            "it": "Information Technology",
            "ash": "Applied Science & Humanities",
        }
        for abbrev, full in abbrev_map.items():
            if re.search(rf"\b{re.escape(abbrev)}\b", prompt_lower):
                return full
        for dept in departments:
            if dept and dept.lower() in prompt_lower:
                return dept
        return ""

    def _extract_designation_from_prompt(prompt_text: str, designations: List[str]) -> str:
        prompt_lower = prompt_text.lower()
        for designation in designations:
            if designation and designation.lower() in prompt_lower:
                return designation
        return ""

    def _extract_degree_from_prompt(prompt_text: str) -> str:
        match = re.search(r"(ph\.?d|m\.?tech|mtech|m\.?e|me|m\.?sc|msc|b\.?tech|btech|b\.?e|be)", prompt_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_experience_years(value: str) -> Optional[int]:
        if not value:
            return None
        match = re.search(r"(\d+)\s*year", value, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_min_years(prompt_text: str) -> Optional[int]:
        match = re.search(r"(more than|over|greater than)\s+(\d+)\s*year", prompt_text, re.IGNORECASE)
        if match:
            return int(match.group(2)) + 1
        match = re.search(r"(at least|>=|more than or equal to)\s+(\d+)\s*year", prompt_text, re.IGNORECASE)
        if match:
            return int(match.group(2))
        match = re.search(r"(\d+)\s*\+\s*year", prompt_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    # Head of Department mapping: canonical department name -> faculty id.
    hod_by_department = {
        "Computer Science & Engineering": "1",
        "Electronics & Communication Engineering": "28",
        "Electrical Engineering": "46",
        "Information Technology": "56",
        "Applied Science & Humanities": "86",
    }
    # Preserve a stable display order for "all HODs" queries.
    hod_department_order = [
        "Computer Science & Engineering",
        "Information Technology",
        "Electronics & Communication Engineering",
        "Electrical Engineering",
        "Applied Science & Humanities",
    ]

    # Ordered (most specific first) department patterns for HOD queries.
    # Word boundaries prevent short codes (it/ee/cs) matching inside other words.
    hod_department_patterns = [
        (
            "Computer Science & Engineering",
            [r"\bcse\b", r"\bcomputer science\b", r"\bcomputer\b", r"\bcs\b"],
        ),
        ("Information Technology", [r"\binformation technology\b", r"\bit\b"]),
        ("Electronics & Communication Engineering", [r"\bece\b", r"\belectronics\b"]),
        ("Electrical Engineering", [r"\bee\b", r"\belectrical\b"]),
        (
            "Applied Science & Humanities",
            [r"\bash\b", r"\bapplied science", r"\bhumanities\b"],
        ),
    ]

    def _is_hod_intent(prompt_text: str) -> bool:
        text = prompt_text.lower()
        if re.search(r"\bhods?\b", text):
            return True
        if re.search(r"\bheads?\s+of\s+(the\s+)?departments?\b", text):
            return True
        return bool(re.search(r"\bdepartments?\s+heads?\b", text))

    def _resolve_hod_department(prompt_text: str) -> str:
        text = prompt_text.lower()
        for department, patterns in hod_department_patterns:
            if any(re.search(pattern, text) for pattern in patterns):
                return department
        return ""

    def _hod_doc_for_department(department: str) -> Optional[Dict[str, Any]]:
        faculty_id = hod_by_department.get(department)
        if not faculty_id:
            return None
        for doc in documents:
            if str(doc.get("id")) == faculty_id:
                entry = dict(doc)
                filename = _find_image_for_id(entry.get("id"))
                if filename:
                    entry["image_url"] = f"/images/{filename}"
                return entry
        return None

    @app.get("/")
    def index() -> Any:
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/filters")
    def filters() -> Any:
        designations = sorted(
            {doc.get("present_designation", "").strip() for doc in documents if doc.get("present_designation")}
        )
        departments = sorted({doc.get("department", "").strip() for doc in documents if doc.get("department")})
        return jsonify({"designations": designations, "departments": departments})

    def _get_filter_lists() -> tuple[list[str], list[str]]:
        designations = sorted(
            {doc.get("present_designation", "").strip() for doc in documents if doc.get("present_designation")}
        )
        departments = sorted({doc.get("department", "").strip() for doc in documents if doc.get("department")})
        return designations, departments

    @app.get("/images/<path:filename>")
    def images(filename: str) -> Any:
        return send_from_directory(IMAGES_DIR, filename)

    @app.get("/scholarship-images/<path:filename>")
    def scholarship_images(filename: str) -> Any:
        return send_from_directory(SCHOLARSHIP_IMAGES_DIR, filename)

    @app.post("/api/search/text")
    def search_text() -> Any:
        payload: Dict[str, Any] = request.get_json(force=True) or {}
        query = (payload.get("query") or "").strip()
        name = payload.get("name")
        designation = payload.get("designation")
        department = payload.get("department")
        if not query and not any([name, designation, department]):
            return jsonify({"error": "Provide a query or at least one filter."}), 400
        try:
            results = search(
                query=query,
                documents=documents,
                index=text_index,
                model_name=payload.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                top_k=int(payload.get("top_k")) if payload.get("top_k") else None,
                name=name,
                designation=designation,
                department=department,
            )
            for item in results:
                filename = _find_image_for_id(item.get("id"))
                if filename:
                    item["image_url"] = f"/images/{filename}"
            return jsonify(results)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/assistant/text")
    def assistant_text() -> Any:
        payload: Dict[str, Any] = request.get_json(force=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        context_id = payload.get("context_id")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        try:
            _greetings = {
                "hi", "hello", "hey", "hiya", "howdy", "greetings", "sup", "yo",
                "good morning", "good afternoon", "good evening", "good night",
            }
            _prompt_clean = re.sub(r"[!.,?]+$", "", prompt.lower().strip())
            if _prompt_clean in _greetings:
                return jsonify({
                    "answer": (
                        "Hi! I'm the BPPIMT Campus Assistant. I can help you with:\n"
                        "- Faculty — search by name, department, or designation\n"
                        "- Placements — records by company, discipline, or year\n"
                        "- Scholarships — find schemes by eligibility\n"
                        "- Campus info — admissions, courses, facilities, and more\n\n"
                        "What would you like to know?"
                    ),
                    "results": [],
                    "filters": {},
                })

            prompt_lower = prompt.lower()

            if _is_hod_intent(prompt):
                hod_department = _resolve_hod_department(prompt)
                if hod_department:
                    hod_doc = _hod_doc_for_department(hod_department)
                    if hod_doc:
                        answer = (
                            f"The Head of Department of {hod_department} is "
                            f"{hod_doc.get('name')}."
                        )
                        return jsonify({"answer": answer, "results": [hod_doc], "filters": {}})
                    return jsonify(
                        {
                            "answer": f"No Head of Department is on record for {hod_department}.",
                            "results": [],
                            "filters": {},
                        }
                    )
                hod_docs = [
                    doc
                    for department in hod_department_order
                    if (doc := _hod_doc_for_department(department)) is not None
                ]
                if hod_docs:
                    lines = [
                        f"- {doc.get('department')}: {doc.get('name')}" for doc in hod_docs
                    ]
                    answer = "Heads of Department:\n" + "\n".join(lines)
                    return jsonify({"answer": answer, "results": hod_docs, "filters": {}})

            list_intent = any(
                keyword in prompt_lower
                for keyword in ["show", "list", "all", "display", "faculty", "teachers"]
            )
            attribute_query = any(
                keyword in prompt_lower
                for keyword in ["university", "highest degree", "degree", "experience", "years"]
            )

            placement_intent = any(
                keyword in prompt_lower
                for keyword in ["placement", "placed", "employer", "company", "on campus", "off campus"]
            )
            scholarship_intent = any(
                keyword in prompt_lower
                for keyword in ["scholarship", "scheme", "grant", "stipend", "fee waiver", "tuition"]
            )
            web_intent = _is_web_intent(prompt)

            if placement_intent:
                offset = payload.get("placement_offset") or payload.get("offset") or 0
                limit = payload.get("placement_limit") or payload.get("limit") or 50
                try:
                    offset = max(0, int(offset))
                except (TypeError, ValueError):
                    offset = 0
                try:
                    limit = max(1, int(limit))
                except (TypeError, ValueError):
                    limit = 50
                model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")
                placement_candidates = placement_records
                _has_keyword_filter = bool(
                    _extract_employer_from_prompt(prompt)
                    or _extract_discipline_from_prompt(prompt)
                    or _extract_years_from_prompt(prompt)
                )
                if not _has_keyword_filter and placement_documents and placement_index is not None:
                    vector_hits = _vector_candidates(
                        prompt,
                        placement_documents,
                        placement_index,
                        model_name,
                        top_k=min(len(placement_documents), max(limit * 5, 200)),
                    )
                    if vector_hits:
                        placement_candidates = vector_hits

                placements = [
                    {**item, "result_type": "placement"}
                    for item in _search_placements(prompt, records=placement_candidates)
                    if isinstance(item, dict)
                ]
                total = len(placements)
                limited = placements[offset : offset + limit]
                answer = f"Found {total} placement records."
                if limited:
                    start_idx = offset + 1
                    end_idx = offset + len(limited)
                    answer = f"Showing {start_idx}-{end_idx} of {total} placement records."
                if total == 0:
                    answer = "No matching placement records found."
                next_offset = offset + limit if offset + limit < total else None
                if total > len(limited) and next_offset is not None and not limited:
                    answer = "No more placement records found."
                try:
                    answer = generate_placement_response(prompt, limited, total, offset, limit)
                except GroqError:
                    pass
                return jsonify(
                    {
                        "answer": answer,
                        "results": limited,
                        "filters": {},
                        "result_type": "placement",
                        "total": total,
                        "offset": offset,
                        "limit": limit,
                        "next_offset": next_offset,
                    }
                )

            if scholarship_intent:
                model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")
                scholarship_candidates = scholarship_records
                if scholarship_documents and scholarship_index is not None:
                    vector_hits = _vector_candidates(
                        prompt,
                        scholarship_documents,
                        scholarship_index,
                        model_name,
                        top_k=min(len(scholarship_documents), 120),
                    )
                    if vector_hits:
                        scholarship_candidates = vector_hits

                scholarships = [
                    {**item, "result_type": "scholarship"}
                    for item in _search_scholarships(prompt, records=scholarship_candidates)
                    if isinstance(item, dict)
                ]
                total = len(scholarships)
                answer = f"Found {total} scholarship schemes."
                if total == 0:
                    answer = "No matching scholarship schemes found."
                try:
                    answer = generate_scholarship_response(prompt, scholarships, total)
                except GroqError:
                    pass
                return jsonify(
                    {
                        "answer": answer,
                        "results": scholarships,
                        "filters": {},
                        "result_type": "scholarship",
                        "total": total,
                    }
                )

            if web_intent:
                model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")
                try:
                    web_cache.refresh(model_name=model_name)
                except Exception:  # noqa: BLE001
                    web_cache.documents = []
                    web_cache.index = None
                web_results = search_web_documents(
                    query=prompt,
                    documents=web_cache.documents,
                    index=web_cache.index,
                    model_name=model_name,
                    top_k=6,
                )
                answer = "I couldn't find relevant information on the BPPIMT website."
                try:
                    answer = generate_web_response(prompt, web_results)
                except GroqError:
                    if web_results:
                        answer = "Here are the closest matches from the BPPIMT website."

                public_results = []
                for item in web_results:
                    public_results.append(
                        {
                            "title": item.get("title"),
                            "url": item.get("url"),
                            "snippet": item.get("snippet"),
                            "pdf_links": item.get("pdf_links") or [],
                            "source": item.get("source") or "bppimt_web",
                            "result_type": "web",
                        }
                    )

                return jsonify(
                    {
                        "answer": answer,
                        "results": public_results,
                        "filters": {},
                        "result_type": "web",
                        "total": len(public_results),
                    }
                )

            inferred_name_for_context = _extract_name_from_prompt(prompt)
            context_doc = None
            if context_id is not None and not inferred_name_for_context and not list_intent and not attribute_query:
                for doc in documents:
                    if str(doc.get("id")) == str(context_id):
                        context_doc = dict(doc)
                        filename = _find_image_for_id(context_doc.get("id"))
                        if filename:
                            context_doc["image_url"] = f"/images/{filename}"
                        break

            field_map = [
                (["department"], "Department", "department"),
                (["highest degree", "degree"], "Highest Degree", "highest_degree"),
                (["university"], "University", "university"),
                (["area of specialization", "specialization"], "Specialization", "specialization"),
                (["date of joining", "joining date"], "Date of Joining", "date_of_joining"),
                (
                    [
                        "experience",
                        "years of experience",
                        "work experience",
                        "years of working",
                        "working years",
                        "years of service",
                        "tenure",
                    ],
                    "Experience",
                    "experience",
                ),
                (["designation at joining", "designation when joined"], "Designation at Joining", "designation_at_joining"),
                (["present designation", "current designation"], "Present Designation", "present_designation"),
                (["date of promotion", "promotion date"], "Date of Promotion", "date_of_promotion"),
                (["association type"], "Association Type", "association_type"),
                (["contract type"], "Contract Type", "contract_type"),
                (["currently associated", "currently associated?"], "Currently Associated", "currently_associated"),
            ]

            prompt_lower = prompt.lower()
            if context_doc:
                requested = []
                for keywords, label, key in field_map:
                    if any(keyword in prompt_lower for keyword in keywords):
                        requested.append((label, key))
                if requested:
                    lines = []
                    for label, key in requested:
                        value = context_doc.get(key) or "Not available"
                        lines.append(f"{label}: {value}")
                    answer = f"{context_doc.get('name')}: " + "; ".join(lines) + "."
                    return jsonify({"answer": answer, "results": [context_doc], "filters": {}})

            requested = []
            for keywords, label, key in field_map:
                if any(keyword in prompt_lower for keyword in keywords):
                    requested.append((label, key))

            name_matches = _find_name_matches(prompt)
            if name_matches and requested:
                if requested and len(name_matches) == 1:
                    doc = name_matches[0]
                    lines = []
                    for label, key in requested:
                        value = doc.get(key) or "Not available"
                        lines.append(f"{label}: {value}")
                    answer = f"{doc.get('name')}: " + "; ".join(lines) + "."
                    return jsonify({"answer": answer, "results": name_matches, "filters": {}})
                if requested and len(name_matches) > 1:
                    return jsonify(
                        {
                            "answer": "Multiple matches found. Please specify the full name.",
                            "results": name_matches,
                            "filters": {},
                        }
                    )

            designations, departments = _get_filter_lists()
            filters = parse_prompt_to_filters(prompt, departments, designations)
            inferred_name = inferred_name_for_context
            if inferred_name and (not filters.get("name") or not _doc_name_match(filters.get("name", ""))):
                filters["name"] = inferred_name

            dept_override = _extract_department_from_prompt(prompt, departments)
            desig_override = _extract_designation_from_prompt(prompt, designations)

            designation_hits = [
                d for d in designations if d and d.lower() in prompt_lower
            ]
            department_hits = [
                d for d in departments if d and d.lower() in prompt_lower
            ]

            if filters.get("name"):
                if not designation_hits:
                    filters["designation"] = ""
                if not department_hits:
                    filters["department"] = ""

            if list_intent or attribute_query:
                if list_intent and not inferred_name_for_context:
                    filters["name"] = ""
                filters["designation"] = desig_override or ""
                filters["department"] = dept_override or ""
            results = search(
                query="",
                documents=documents,
                index=text_index,
                model_name=payload.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                name=filters.get("name") or None,
                designation=filters.get("designation") or None,
                department=filters.get("department") or None,
            )
            for item in results:
                filename = _find_image_for_id(item.get("id"))
                if filename:
                    item["image_url"] = f"/images/{filename}"

            university_token = _extract_university_from_prompt(prompt)
            degree_token = _extract_degree_from_prompt(prompt)
            min_years = _extract_min_years(prompt)

            if university_token:
                query_keys = _university_query_keys(university_token)
                if not query_keys or not query_keys.intersection(university_key_index):
                    university_token = ""

            if list_intent or attribute_query:
                filtered_results = results
                if university_token:
                    query_keys = _university_query_keys(university_token)
                    filtered_results = [
                        doc
                        for doc in filtered_results
                        if query_keys
                        and _university_key_set(str(doc.get("university", "")))
                        and _university_key_set(str(doc.get("university", ""))).intersection(query_keys)
                    ]
                if degree_token:
                    deg_norm = _normalize_field_text(degree_token)
                    filtered_results = [
                        doc
                        for doc in filtered_results
                        if deg_norm and deg_norm in _normalize_field_text(str(doc.get("highest_degree", "")))
                    ]
                if min_years is not None:
                    filtered_results = [
                        doc
                        for doc in filtered_results
                        if (_parse_experience_years(str(doc.get("experience", ""))) or 0) >= min_years
                    ]
                results = filtered_results
            if results:
                requested = []
                for keywords, label, key in field_map:
                    if any(keyword in prompt_lower for keyword in keywords):
                        requested.append((label, key))
                if requested and len(results) == 1:
                    doc = results[0]
                    lines = []
                    for label, key in requested:
                        value = doc.get(key) or "Not available"
                        lines.append(f"{label}: {value}")
                    answer = f"{doc.get('name')}: " + "; ".join(lines) + "."
                    return jsonify({"answer": answer, "results": results, "filters": filters})
                if requested and len(results) > 1:
                    return jsonify(
                        {
                            "answer": "Multiple matches found. Please specify the full name.",
                            "results": results,
                            "filters": filters,
                        }
                    )
            if attribute_query and requested and not results:
                return jsonify(
                    {
                        "answer": "Please specify the full name for this query.",
                        "results": [],
                        "filters": filters,
                    }
                )
            info_intent = any(
                keyword in prompt_lower
                for keyword in [
                    "date of joining",
                    "joining date",
                    "experience",
                    "university",
                    "specialization",
                    "degree",
                    "designation",
                    "promotion",
                    "association",
                    "contract",
                ]
            )

            if list_intent and (not info_intent or attribute_query):
                dept_label = filters.get("department") or "all departments"
                desig_label = filters.get("designation") or "faculty"
                count = len(results)
                answer = f"Found {count} {desig_label} in {dept_label}."
                if count == 0:
                    answer = "No matching faculty found."
                return jsonify({"answer": answer, "results": results, "filters": filters})

            context_results = results[:15]
            answer = generate_response(prompt, context_results)
            return jsonify({"answer": answer, "results": results, "filters": filters})
        except GroqError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/search/image")
    def search_image() -> Any:
        if "image" not in request.files:
            return jsonify({"error": "Image file is required"}), 400
        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"error": "Image file is required"}), 400

        try:
            image = Image.open(file.stream).convert("RGB")
        except OSError:
            return jsonify({"error": "Invalid image file"}), 400

        try:
            _ensure_face_resources()
            results = search_face(
                image=image,
                documents=face_documents,
                index=face_index,
                meta=face_meta,
                mtcnn=mtcnn,
                resnet=resnet,
                device=device,
                top_k=1,
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

        for item in results:
            filename = item.get("image_filename")
            if filename:
                item["image_url"] = f"/images/{filename}"
        return jsonify(results)

    @app.post("/api/assistant/image")
    def assistant_image() -> Any:
        prompt = (request.form.get("prompt") or "").strip()
        if "image" not in request.files:
            return jsonify({"error": "Image file is required"}), 400
        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"error": "Image file is required"}), 400

        try:
            image = Image.open(file.stream).convert("RGB")
        except OSError:
            return jsonify({"error": "Invalid image file"}), 400

        try:
            _ensure_face_resources()
            results = search_face(
                image=image,
                documents=face_documents,
                index=face_index,
                meta=face_meta,
                mtcnn=mtcnn,
                resnet=resnet,
                device=device,
                top_k=1,
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

        for item in results:
            filename = item.get("image_filename")
            if filename:
                item["image_url"] = f"/images/{filename}"
        if results:
            best = results[0]
            prompt_lower = (prompt or "").lower()
            field_map = [
                (["department"], "Department", "department"),
                (["highest degree", "degree"], "Highest Degree", "highest_degree"),
                (["university"], "University", "university"),
                (["area of specialization", "specialization"], "Specialization", "specialization"),
                (["date of joining", "joining date"], "Date of Joining", "date_of_joining"),
                (
                    [
                        "experience",
                        "years of experience",
                        "work experience",
                        "years of working",
                        "working years",
                        "years of service",
                        "tenure",
                    ],
                    "Experience",
                    "experience",
                ),
                (["designation at joining", "designation when joined"], "Designation at Joining", "designation_at_joining"),
                (["present designation", "current designation"], "Present Designation", "present_designation"),
                (["date of promotion", "promotion date"], "Date of Promotion", "date_of_promotion"),
                (["association type"], "Association Type", "association_type"),
                (["contract type"], "Contract Type", "contract_type"),
                (["currently associated", "currently associated?"], "Currently Associated", "currently_associated"),
            ]
            requested = []
            for keywords, label, key in field_map:
                if any(keyword in prompt_lower for keyword in keywords):
                    requested.append((label, key))

            if requested:
                lines = []
                for label, key in requested:
                    value = best.get(key) or "Not available"
                    lines.append(f"{label}: {value}")
                answer = f"{best.get('name')}: " + "; ".join(lines) + "."
                return jsonify({"answer": answer, "results": results})
            answer = (
                f"Best match: {best.get('name')} — {best.get('present_designation')} "
                f"({best.get('department')})."
            )
            return jsonify({"answer": answer, "results": results})
        try:
            answer = generate_response(prompt or "Identify the faculty member", results)
        except GroqError as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({"answer": answer, "results": results})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
