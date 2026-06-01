"""Generate scholarship text documents from scholarship JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "scholarship_details.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "processed" / "scholarship_documents.json"


def _stringify_value(value: Any) -> str:
	if isinstance(value, list):
		return ", ".join(str(item) for item in value if item)
	if isinstance(value, dict):
		return json.dumps(value, ensure_ascii=False)
	return str(value or "")


def _build_text(record: Dict[str, Any]) -> str:
	parts = [
		f"Title: {record.get('title', '')}",
		f"Short Name: {record.get('short_name', '')}",
		f"Type: {record.get('scholarship_type', '')}",
		f"State: {record.get('state', '')}",
		f"Provider: {record.get('provider', '')}",
		f"Overview: {record.get('overview', '')}",
		f"Categories: {_stringify_value(record.get('categories'))}",
		f"Target Groups: {_stringify_value(record.get('target_groups'))}",
		f"Eligible Communities: {_stringify_value(record.get('eligible_communities'))}",
		f"Eligible Courses: {_stringify_value(record.get('eligible_courses'))}",
		f"Eligibility: {_stringify_value(record.get('eligibility'))}",
		f"Benefits: {_stringify_value(record.get('benefits'))}",
		f"Application Mode: {record.get('application_mode', '')}",
		f"Keywords: {_stringify_value(record.get('keywords'))}",
	]
	return ". ".join([part for part in parts if part and not part.endswith(": ")])


def generate_documents(input_json: Path, output_json: Path) -> List[Dict[str, Any]]:
	with input_json.open("r", encoding="utf-8") as handle:
		records = json.load(handle)

	documents: List[Dict[str, Any]] = []
	for record in records:
		if not isinstance(record, dict):
			continue
		doc = dict(record)
		doc["text"] = _build_text(record)
		documents.append(doc)

	output_json.parent.mkdir(parents=True, exist_ok=True)
	with output_json.open("w", encoding="utf-8") as handle:
		json.dump(documents, handle, indent=2, ensure_ascii=False)

	return documents


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate scholarship documents from JSON")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to input JSON")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output JSON")
	args = parser.parse_args()

	generate_documents(args.input, args.output)
	print(f"Saved scholarship documents to {args.output}")


if __name__ == "__main__":
	main()
