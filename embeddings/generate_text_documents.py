"""Generate faculty text documents from the cleaned CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "faculty_list_cleaned.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "processed" / "faculty_documents.json"


def _build_text(row: Dict[str, str]) -> str:
	parts = [
		f"Name: {row.get('Name of Faculty', '').strip()}",
		f"Department: {row.get('Department', '').strip()}",
		f"Designation: {row.get('Present Designation', '').strip()}",
		f"Specialization: {row.get('Area of Specialization', '').strip()}",
		f"Highest Degree: {row.get('Highest Degree', '').strip()}",
		f"University: {row.get('University', '').strip()}",
		f"Experience: {row.get('Experience', '').strip()}",
		f"Association Type: {row.get('Association Type', '').strip()}",
		f"Contract Type: {row.get('Contract Type', '').strip()}",
		f"Currently Associated: {row.get('Currently Associated', '').strip()}",
	]
	return ". ".join([p for p in parts if p and not p.endswith(': ')])


def generate_documents(input_csv: Path, output_json: Path) -> List[Dict[str, str]]:
	documents: List[Dict[str, str]] = []
	with input_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			doc = {
				"id": row.get("S.No", "").strip(),
				"name": row.get("Name of Faculty", "").strip(),
				"department": row.get("Department", "").strip(),
				"present_designation": row.get("Present Designation", "").strip(),
				"designation_at_joining": row.get("Designation at Joining", "").strip(),
				"highest_degree": row.get("Highest Degree", "").strip(),
				"university": row.get("University", "").strip(),
				"specialization": row.get("Area of Specialization", "").strip(),
				"date_of_joining": row.get("Date of Joining", "").strip(),
				"experience": row.get("Experience", "").strip(),
				"association_type": row.get("Association Type", "").strip(),
				"contract_type": row.get("Contract Type", "").strip(),
				"currently_associated": row.get("Currently Associated", "").strip(),
				"date_of_leaving": row.get("Date of Leaving", "").strip(),
				"photo_path": row.get("photo_path", "").strip(),
			}
			doc["text"] = _build_text(row)
			documents.append(doc)

	output_json.parent.mkdir(parents=True, exist_ok=True)
	with output_json.open("w", encoding="utf-8") as handle:
		json.dump(documents, handle, indent=2, ensure_ascii=False)

	return documents


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate faculty text documents from CSV")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to input CSV")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output JSON")
	args = parser.parse_args()

	generate_documents(args.input, args.output)
	print(f"Saved documents to {args.output}")


if __name__ == "__main__":
	main()
