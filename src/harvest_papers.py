#!/usr/bin/env python3
import re, csv, argparse, pathlib, time, unicodedata, random
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

OPENALEX = "https://api.openalex.org"
HEADERS = {"User-Agent": "tue-llm-shift/1.0 (mailto:julian.abc@student.uni-tuebingen.de)"}

# ---------- HTTP ----------
@retry(reraise=True, stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=1, max=30),
       retry=retry_if_exception_type(requests.RequestException))
def get_json(url, params=None):
    params = dict(params or {})
    params.setdefault("mailto", "julian.abc@student.uni-tuebingen.de")
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    if r.status_code >= 400:
        try:
            print("OpenAlex error:", r.json())
        except Exception:
            print("OpenAlex error (text):", r.text[:500])
    r.raise_for_status()
    return r.json()

# ---------- OpenAlex helpers ----------
def resolve_institution_id(name: str):
    j = get_json(f"{OPENALEX}/institutions", params={"search": name, "per-page": 25})
    cands = j.get("results", [])
    if not cands:
        raise RuntimeError(f"No institution found for '{name}'")
    name_l = name.lower()
    for c in cands:
        if c.get("display_name","").lower() == name_l:
            return c["id"].split("/")[-1]
    for c in cands:
        if c.get("display_name","").lower().startswith(name_l):
            return c["id"].split("/")[-1]
    return cands[0]["id"].split("/")[-1]

def resolve_concepts(concept_names):
    ids = []
    for q in concept_names:
        j = get_json(f"{OPENALEX}/concepts", params={"search": q, "per-page": 5})
        best = None
        for c in j.get("results", []):
            if c.get("display_name","").lower() == q.lower():
                best = c; break
        if not best and j.get("results"):
            best = j["results"][0]
        if best:
            ids.append(best["id"].split("/")[-1])
    return ids

def build_filters(inst_id, y0, y1, concept_ids=None, doc_types=None, lang=None):
    flt = []
    if inst_id:  # optional institution
        flt.append(f"authorships.institutions.id:{inst_id}")
    flt.append(f"publication_year:{y0}-{y1}")
    if concept_ids: flt.append("concepts.id:" + "|".join(concept_ids))
    if doc_types:  flt.append("type:" + "|".join(doc_types))
    if lang:       flt.append(f"language:{lang}")
    return ",".join(flt)

def count_works(inst_id, y0, y1, concept_ids=None, doc_types=None, lang=None):
    params = {"filter": build_filters(inst_id, y0, y1, concept_ids, doc_types, lang),
              "per-page": 1, "select": "id", "cursor": "*"}
    j = get_json(f"{OPENALEX}/works", params=params)
    return j.get("meta", {}).get("count", 0)

SELECT_FIELDS = ",".join([
    "id","title","publication_year","publication_date","type","language",
    "abstract_inverted_index",
    "best_oa_location","primary_location","locations","open_access"
])

def _iter_year(inst_id, year, concept_ids=None, doc_types=None, lang=None):
    params = {
        "filter": build_filters(inst_id, year, year, concept_ids, doc_types, lang),
        "per-page": 200,
        "sort": "publication_date:asc",
        "select": SELECT_FIELDS,
        "cursor": "*"
    }
    while True:
        j = get_json(f"{OPENALEX}/works", params=params)
        results = j.get("results", [])
        if not results: break
        for w in results: yield w
        nxt = j.get("meta", {}).get("next_cursor")
        if not nxt: break
        params["cursor"] = nxt

def iter_works_range(inst_id, y0, y1, concept_ids=None, max_works=100000, doc_types=None, lang=None):
    params = {
        "filter": build_filters(inst_id, y0, y1, concept_ids, doc_types, lang),
        "per-page": 200,
        "sort": "publication_year:asc",
        "select": SELECT_FIELDS,
        "cursor": "*"
    }
    total = 0
    while True:
        j = get_json(f"{OPENALEX}/works", params=params)
        for w in j.get("results", []):
            yield w
            total += 1
            if total >= max_works: return
        nxt = j.get("meta", {}).get("next_cursor")
        if not nxt: return
        params["cursor"] = nxt

# ---------- Abstract & Cleaning ----------
def reconstruct_abstract(inv):
    if not inv: return ""
    positions = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    if not positions: return ""
    positions.sort(key=lambda x: x[0])
    words = [w for _, w in positions]
    text = " ".join(words)
    text = unicodedata.normalize("NFKC", text).replace("\u00AD", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

_MATH_PATTERNS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"(?<!\$)\$[^$]*\$(?!\$)", re.DOTALL),
    re.compile(r"\\\[(.*?)\\\]", re.DOTALL),
    re.compile(r"\\\((.*?)\\\)", re.DOTALL),
    re.compile(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", re.DOTALL),
    re.compile(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", re.DOTALL),
    re.compile(r"\\begin\{gather\*?\}.*?\\end\{gather\*?\}", re.DOTALL),
    re.compile(r"\\begin\{eqnarray\*?\}.*?\\end\{eqnarray\*?\}", re.DOTALL),
]

def clean_text(raw: str) -> str:
    if not raw: return ""
    txt = raw
    for pat in _MATH_PATTERNS:
        txt = pat.sub(" ", txt)
    lines_out = []
    for line in txt.splitlines():
        l = line.strip()
        if not l: continue
        sym_ratio = sum(1 for c in l if re.match(r"[^\w\s.,;:()'%/+-]", c)) / max(1, len(l))
        if sym_ratio > 0.25: continue
        if re.match(r"^(figure|fig\.|table|algorithm)\s+\d+[:.)-]", l, re.I): continue
        lines_out.append(l)
    txt = " ".join(lines_out)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def pick_link(work):
    bol = work.get("best_oa_location") or {}
    for key in ("landing_page_url", "url"):
        if bol.get(key): return bol[key]
    pl = work.get("primary_location") or {}
    if pl.get("landing_page_url"): return pl["landing_page_url"]
    for loc in (work.get("locations") or []):
        if loc.get("landing_page_url"): return loc["landing_page_url"]
    return f"https://openalex.org/{work.get('id','').split('/')[-1]}"

# ---------- Eligible iterator & sampling ----------
def iter_eligible_year(inst_id, year, concept_ids, doc_types, lang, min_len):
    for w in _iter_year(inst_id, year, concept_ids, doc_types, lang):
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        abstract = clean_text(abstract)
        if not abstract or len(abstract) < min_len:
            continue
        date = w.get("publication_date") or (f"{w.get('publication_year','NA')}-01-01")
        headline = (w.get("title") or "").strip()
        link = pick_link(w)
        yield {"date": date, "headline": headline, "article": abstract, "link": link}

def sample_year_reservoir_eligible(inst_id, year, cap, concept_ids, doc_types, lang, min_len):
    res, n = [], 0
    for row in iter_eligible_year(inst_id, year, concept_ids, doc_types, lang, min_len):
        n += 1
        if cap <= 0:
            res.append(row); continue
        if len(res) < cap:
            res.append(row)
        else:
            j = random.randint(1, n)
            if j <= cap:
                res[j - 1] = row
    return res, n

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Harvest Tübingen abstracts (cleaned) into a CSV")
    ap.add_argument("--institution", required=False, help="Optional institution name (global search if omitted)")
    ap.add_argument("--year-start", type=int, default=2010)
    ap.add_argument("--year-end", type=int, default=2025)
    ap.add_argument("--concept", action="append", default=[])
    ap.add_argument("--restrict-cs", type=int, default=0)
    ap.add_argument("--doc-type", action="append", default=[])
    ap.add_argument("--lang", default=None, help="Optional language filter, e.g. en")
    ap.add_argument("--max-works", type=int, default=100000)
    ap.add_argument("--max-per-year", type=int, default=0, help="Random cap per year (0 = no limit)")
    ap.add_argument("--min-text-chars", type=int, default=200)
    ap.add_argument("--csv-path", default="./tue_papers_abstracts.csv")
    args = ap.parse_args()

    # institution optional
    inst_id = None
    if args.institution:
        inst_id = resolve_institution_id(args.institution)

    concept_names = list(args.concept)
    if args.restrict_cs and "computer science" not in [c.lower() for c in concept_names]:
        concept_names.append("Computer Science")
    concept_ids = resolve_concepts(concept_names) if concept_names else []

    # CSV
    pathlib.Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.csv_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\', doublequote=True)
    writer.writerow(["date", "headline", "article", "link"])

    if args.max_per_year > 0:
        yearly_samples = {}
        total_rows = 0
        for year in range(args.year_start, args.year_end + 1):
            sample, _eligible = sample_year_reservoir_eligible(
                inst_id, year, args.max_per_year, concept_ids, args.doc_type, args.lang, args.min_text_chars
            )
            yearly_samples[year] = sample
            total_rows += len(sample)

        print(f"Planned rows to write (sum of per-year samples): {total_rows}")

        with tqdm(total=total_rows, desc="Writing CSV", unit="row") as pbar:
            for year in range(args.year_start, args.year_end + 1):
                for r in yearly_samples.get(year, []):
                    writer.writerow([r["date"], r["headline"], r["article"], r["link"]])
                    pbar.update(1)
                    time.sleep(0.005)
    else:
        total_hits = count_works(inst_id, args.year_start, args.year_end,
                                 concept_ids=concept_ids, doc_types=args.doc_type, lang=args.lang)
        print(f"Total works (progress target): {total_hits}")
        written = 0
        with tqdm(total=total_hits, desc="Harvesting abstracts", unit="work") as pbar:
            for w in iter_works_range(inst_id, args.year_start, args.year_end,
                                      concept_ids=concept_ids, max_works=args.max_works,
                                      doc_types=args.doc_type, lang=args.lang):
                pbar.update(1)
                abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
                abstract = clean_text(abstract)
                if not abstract or len(abstract) < args.min_text_chars:
                    continue
                date = w.get("publication_date") or (f"{w.get('publication_year','NA')}-01-01")
                headline = (w.get("title") or "").strip()
                link = pick_link(w)
                writer.writerow([date, headline, abstract, link])
                written += 1
                time.sleep(0.005)
        print(f"Rows written: {written}")

    fout.close()
    print(f"CSV written to: {args.csv_path}")

if __name__ == "__main__":
    main()
