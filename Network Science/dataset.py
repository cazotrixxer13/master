import os
import json
import requests
from datetime import datetime

BASE_URL = "https://api.openalex.org/works"

OUT_DIR = "openalex_snapshots"
os.makedirs(OUT_DIR, exist_ok=True)

FIELDS = "id,publication_year,primary_topic,referenced_works"

FILTER_TEMPLATE = "primary_topic.field.id:fields/17,publication_year:{year}"

TYPES = (
    "types/article|types/book-chapter|types/dissertation|types/book|"
    "types/review|types/letter|types/report|types/book-section|types/preprint"
)

MAX_PER_FILE = 10000

START_YEAR = 1950
END_YEAR = datetime.now().year

def clean_work(w: dict) -> dict:
    """
    Shrink a raw OpenAlex work object to only what we need for the project.
    """
    pt = w.get("primary_topic") or {}
    sf = pt.get("subfield") or {}
    fld = pt.get("field") or {}

    return {
        "id": w.get("id"),
        "year": w.get("publication_year"),
        "field": fld.get("display_name"),
        "field_id": fld.get("id"),
        "subfield": sf.get("display_name"),
        "subfield_id": sf.get("id"),
        "primary_topic": pt.get("display_name"),
        "primary_topic_id": pt.get("id"),
        "referenced_works": w.get("referenced_works", []) or [],
    }


def fetch_count(year: int) -> int:
    """
    Ask OpenAlex how many CS-primary papers exist for the given year.
    This is cheap because we request only 1 result and read meta.count.
    """
    params = {
        "filter": FILTER_TEMPLATE.format(year=year) + f",type:{TYPES}",
        "select": "id",      # keep it tiny
        "per-page": 1,
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["meta"]["count"]


def fetch_snapshot(year: int, max_per_file: int = MAX_PER_FILE) -> None:
    """
    Download all CS-primary works for a given year, shrink them,
    and save them in JSONL chunks.
    """
    total_expected = fetch_count(year)
    if total_expected == 0:
        print(f"[INFO] Year {year}: 0 CS papers, skipping.")
        return

    print(f"[INFO] Total CS (primary field) papers in {year}: {total_expected}")

    cursor = "*"
    total = 0
    file_index = 1
    items_in_file = 0

    out_path = os.path.join(OUT_DIR, f"cs_{year}_part_{file_index:03d}.jsonl")
    out_f = open(out_path, "w", encoding="utf-8")

    while True:
        params = {
            "filter": FILTER_TEMPLATE.format(year=year) + f",type:{TYPES}",
            "select": FIELDS,
            "per-page": 200,
            "cursor": cursor,
        }

        r = requests.get(BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            cleaned = clean_work(work)
            out_f.write(json.dumps(cleaned) + "\n")
            total += 1
            items_in_file += 1

            if items_in_file >= max_per_file:
                out_f.close()
                file_index += 1
                items_in_file = 0
                out_path = os.path.join(OUT_DIR, f"cs_{year}_part_{file_index:03d}.jsonl")
                out_f = open(out_path, "w", encoding="utf-8")

        print(f"Year {year}: fetched {total}/{total_expected}")

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    out_f.close()
    print(f"Year {year}: DONE (downloaded {total}/{total_expected})")

if __name__ == "__main__":
    print(f"Downloading full CS network by year from {START_YEAR} to {END_YEAR}...")
    for y in range(START_YEAR, END_YEAR + 1):
        try:
            fetch_snapshot(y)
        except Exception as e:
            print(f"[ERROR] Year {y}: {e}")
    print("All done.")
