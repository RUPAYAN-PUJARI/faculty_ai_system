"""Generate placement text documents from placement CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "placement_details.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "processed" / "placement_documents.json"


def _build_text(row: Dict[str, str]) -> str:
	parts = [
		f"Student: {row.get('Student_Name', '').strip()}",
		f"Enrollment: {row.get('Enrollment_Number', '').strip()}",
		f"Discipline: {row.get('Discipline', '').strip()}",
		f"Year of Passing: {row.get('Year_of_Passing', '').strip()}",
		f"Campus: {row.get('On_Off_Campus', '').strip()}",
		f"Employer: {row.get('Employer', '').strip()}",
		f"Academic Year: {row.get('Academic_Year', '').strip()}",
	]
	return ". ".join([p for p in parts if p and not p.endswith(": ")])


def generate_documents(input_csv: Path, output_json: Path) -> List[Dict[str, str]]:
	documents: List[Dict[str, str]] = []
	with input_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			doc = {
				"id": row.get("Sl_No", "").strip(),
				"student_name": row.get("Student_Name", "").strip(),
				"enrollment_number": row.get("Enrollment_Number", "").strip(),
				"discipline": row.get("Discipline", "").strip(),
				"year_of_passing": row.get("Year_of_Passing", "").strip(),
				"on_off_campus": row.get("On_Off_Campus", "").strip(),
				"employer": row.get("Employer", "").strip(),
				"academic_year": row.get("Academic_Year", "").strip(),
			}
			doc["text"] = _build_text(row)
			documents.append(doc)

	output_json.parent.mkdir(parents=True, exist_ok=True)
	with output_json.open("w", encoding="utf-8") as handle:
		json.dump(documents, handle, indent=2, ensure_ascii=False)

	return documents


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate placement documents from CSV")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to input CSV")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output JSON")
	args = parser.parse_args()

	generate_documents(args.input, args.output)
	print(f"Saved placement documents to {args.output}")


if __name__ == "__main__":
	main()
