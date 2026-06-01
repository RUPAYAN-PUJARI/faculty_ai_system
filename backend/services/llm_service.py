"""Groq LLM service utilities (Llama)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "512"))


class GroqError(RuntimeError):
    pass


def _call_groq(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    if not GROQ_API_KEY:
        raise GroqError("GROQ_API_KEY is not set. Add it to .env and restart the server.")

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or GROQ_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise GroqError(f"Groq error: {response.status_code} {response.text}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise GroqError("Groq response missing content") from exc


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _match_department(prompt: str, departments: List[str]) -> str:
    normalized_prompt = _normalize_text(prompt)
    keyword_map = {
        "computer science & engineering": ["cse", "cs", "computer science"],
        "electronics & communication engineering": ["ece", "electronics"],
        "electrical engineering": ["ee", "electrical"],
        "information technology": ["it", "information technology"],
        "applied science & humanities": ["ash", "applied science", "humanities"],
    }

    for dept in departments:
        dept_norm = _normalize_text(dept)
        for canonical, keywords in keyword_map.items():
            if any(keyword in normalized_prompt for keyword in keywords):
                if canonical in dept_norm:
                    return dept
    for canonical, keywords in keyword_map.items():
        if any(keyword in normalized_prompt for keyword in keywords):
            return canonical.title().replace("&", "&")
    return ""


def parse_prompt_to_filters(prompt: str, departments: List[str], designations: List[str]) -> Dict[str, str]:
    system_prompt = (
        "You are a query parser for a faculty search engine. "
        "Extract name, designation, and department from the user prompt. "
        "Return ONLY valid JSON with keys: name, designation, department. "
        "If a value is unknown, return an empty string. "
        "Use the provided departments/designations when matching. "
        "Map common abbreviations: CSE or CS -> Computer Science & Engineering; "
        "ECE -> Electronics & Communication Engineering; EE -> Electrical Engineering; "
        "IT -> Information Technology; ASH -> Applied Science & Humanities."
    )
    user_prompt = (
        "Prompt: "
        f"{prompt}\n\n"
        f"Departments: {departments}\n"
        f"Designations: {designations}\n"
        "Return JSON only."
    )

    content = _call_groq(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )

    try:
        parsed: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GroqError("Failed to parse LLM response as JSON") from exc

    name = str(parsed.get("name", "")).strip()
    designation = str(parsed.get("designation", "")).strip()
    department = str(parsed.get("department", "")).strip()

    prompt_norm = _normalize_text(prompt)
    designation_norm = _normalize_text(designation)
    if designation_norm in {"faculty", "faculties", "staff", "teacher", "teachers"}:
        designation = ""
    if any(keyword in prompt_norm for keyword in ["faculty", "teachers", "teacher", "staff"]):
        designation = ""

    if not department:
        department = _match_department(prompt, departments)
    if not department and any(token in prompt_norm for token in ["cse", "cs", "computer science"]):
        department = _match_department("cse", departments)

    return {"name": name, "designation": designation, "department": department}


def generate_response(prompt: str, results: List[Dict[str, Any]]) -> str:
    context_lines = []
    for item in results[:15]:
        line = (
            f"Name: {item.get('name')}; Designation: {item.get('present_designation')}; "
            f"Department: {item.get('department')}; Specialization: {item.get('specialization')}; "
            f"Experience: {item.get('experience')}"
        )
        context_lines.append(line[:300])

    fallback = "I couldn't find a match."
    if results:
        top = results[0]
        name = top.get("name") or "This faculty member"
        designation = top.get("present_designation") or ""
        department = top.get("department") or ""
        extra = ""
        if designation and department:
            extra = f" — {designation} in {department}."
        elif designation:
            extra = f" — {designation}."
        elif department:
            extra = f" — {department}."
        fallback = f"Match found: {name}{extra}"

    system_prompt = (
        "You are a helpful faculty assistant. Use only the provided context. "
        "If context is provided, do NOT say you couldn't find a match. "
        "Keep it concise."
    )
    user_prompt = (
        f"User prompt: {prompt}\n"
        f"Top match summary: {fallback}\n"
        "Context:\n" + "\n".join(context_lines)
    )

    response = _call_groq(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    if results and isinstance(response, str):
        lowered = response.lower()
        if "couldn't find" in lowered or "no match" in lowered:
            return fallback
    return response


def generate_placement_response(
    prompt: str,
    results: List[Dict[str, Any]],
    total: int,
    offset: int,
    limit: int,
) -> str:
    if total == 0 or not results:
        return "No matching placement records found."

    start_idx = offset + 1
    end_idx = offset + len(results)
    companies = []
    seen = set()
    for item in results:
        employer = (item.get("employer") or "").strip()
        if employer and employer.lower() not in seen:
            seen.add(employer.lower())
            companies.append(employer)
    company_list = ", ".join(companies) if companies else "Not available"
    required_response = (
        f"Here are the placement details for records {start_idx}-{end_idx} of {total}. "
        f"Companies in this range: {company_list}."
    )

    system_prompt = (
        "You are a placements assistant. Respond with exactly one sentence that starts with "
        "'Here are the placement details' and includes the company list exactly as provided. "
        "Do not add extra details."
    )
    user_prompt = f"Required response: {required_response}"

    response = _call_groq(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=160,
    )
    if not isinstance(response, str) or "Here are the placement details" not in response:
        return required_response
    return response


def generate_scholarship_response(
    prompt: str,
    results: List[Dict[str, Any]],
    total: int,
) -> str:
    context_lines = []
    for item in results[:15]:
        eligibility = item.get("eligibility") or {}
        benefits = item.get("benefits") or {}
        line = (
            f"Title: {item.get('title')}; Short Name: {item.get('short_name')}; "
            f"Type: {item.get('scholarship_type')}; State: {item.get('state')}; "
            f"Provider: {item.get('provider')}; Targets: {', '.join(item.get('target_groups') or [])}; "
            f"Max Income: {eligibility.get('maximum_family_income') or eligibility.get('sc_st_maximum_family_income')}; "
            f"Tuition Limit: {benefits.get('tuition_fee_limit')}; Fee Waiver: {benefits.get('tuition_fee_waiver')}"
        )
        context_lines.append(line[:320])

    summary = f"Summary: total={total}, showing={len(results)}."

    system_prompt = (
        "You are a helpful scholarship assistant. Use only the provided context lines and summary. "
        "If no results, say you couldn't find matching scholarship schemes. Keep it concise and factual."
    )
    user_prompt = (
        f"User prompt: {prompt}\n"
        f"{summary}\n"
        "Context:\n" + "\n".join(context_lines)
    )

    return _call_groq(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=280,
    )


def generate_web_response(prompt: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "I couldn't find relevant information on the BPPIMT website."

    context_lines = []
    for item in results[:8]:
        title = item.get("title") or "BPPIMT Page"
        url = item.get("url") or ""
        snippet = str(item.get("snippet") or item.get("content") or "")
        snippet = snippet[:320]
        pdf_links = item.get("pdf_links") or []
        pdf_text = ""
        if pdf_links:
            pdf_parts = []
            for link in pdf_links[:4]:
                label = link.get("title") or "PDF"
                href = link.get("url") or ""
                if href:
                    pdf_parts.append(f"{label}: {href}")
            if pdf_parts:
                pdf_text = " PDF Links: " + " | ".join(pdf_parts)
        context_lines.append(
            f"Title: {title}; URL: {url}; Snippet: {snippet}.{pdf_text}"
        )

    system_prompt = (
        "You are a helpful assistant for BPPIMT website queries. "
        "Use ONLY the provided context lines. "
        "Answer the user's question concisely and include relevant URLs. "
        "If the user asks about timetable/routine, include the PDF links if present. "
        "If the answer is not in the context, say so explicitly."
    )
    user_prompt = (
        f"User question: {prompt}\n"
        "Context:\n" + "\n".join(context_lines)
    )

    return _call_groq(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=320,
    )
