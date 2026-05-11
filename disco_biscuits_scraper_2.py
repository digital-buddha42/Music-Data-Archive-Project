#!/usr/bin/env python3
"""
Disco Biscuits Full Scraper
============================
Scrapes discobiscuits.net for all shows (1995-2026) and produces
3 Tableau-ready CSV files:

  1. shows.csv       one row per show  (~2,054 rows)
  2. setlists.csv    one row per song played  (~25-30k rows)
  3. songs_dim.csv   one row per unique song (dimension table)

Usage:
  pip install requests beautifulsoup4
  python disco_biscuits_scraper.py

Output files are written to ./output/
Runtime: ~8-15 minutes (rate limiting included)
"""

import csv
import json
import time
import re
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependencies. Run:  pip install requests beautifulsoup4")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://discobiscuits.net"
YEARS       = list(range(1995, 2027))
OUTPUT_DIR  = "./output"
MAX_WORKERS = 10          # concurrent requests
DELAY       = 0.05        # seconds between requests per worker
TIMEOUT     = 15

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project)",
    "Accept": "text/html,application/xhtml+xml",
}

# Confirmed from live DOM inspection:
#   Song <a> tags have NO class themselves — cursor-pointer is on the PARENT <span>
#   Set labels are <span class="...text-content-text-tertiary..."> with text S1/S2/E etc
#   Segues are <span class="...text-content-text-secondary..."> with text ">"
SET_LABEL_RE = re.compile(r"^(S[1-4]|E[0-3]?)$")


# ── Step 1: Discover all show URLs ────────────────────────────────────────────

def get_show_urls_for_year(year: int) -> list:
    url = f"{BASE_URL}/shows/year/{year}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] Year {year} failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    seen = set()
    urls = []
    for a in soup.find_all("a", href=re.compile(r"/shows/\d{4}-\d{2}-\d{2}")):
        href = a["href"].split("#")[0]          # strip #photos fragments
        full = href if href.startswith("http") else BASE_URL + href
        if full not in seen:
            seen.add(full)
            urls.append(full)
    return urls


def discover_all_urls() -> list:
    print("── Step 1: Discovering show URLs ──────────────────────────")
    all_urls = []
    seen = set()
    for year in YEARS:
        urls = get_show_urls_for_year(year)
        added = 0
        for u in urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)
                added += 1
        print(f"  {year}: {added:3d} shows  (total so far: {len(all_urls)})")
        time.sleep(0.1)
    print(f"\nTotal unique show URLs: {len(all_urls)}\n")
    return all_urls


# ── Step 2: Scrape individual show pages ──────────────────────────────────────

def parse_show(url: str) -> dict | None:
    time.sleep(DELAY)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Metadata from JSON-LD schema tag ──────────────────────────────────
    schema = {}
    script_tag = soup.find("script", type="application/ld+json")
    if script_tag:
        try:
            schema = json.loads(script_tag.string or "{}")
        except json.JSONDecodeError:
            pass

    date    = schema.get("startDate", "")
    venue   = schema.get("location", {}).get("name", "")
    addr    = schema.get("location", {}).get("address", {})
    city    = addr.get("addressLocality", "")
    state   = addr.get("addressRegion", "")
    country = addr.get("addressCountry", "US")

    if not date:
        return None

    # ── Setlist parsing ────────────────────────────────────────────────────
    songs       = []
    current_set = "S1"
    position    = 1
    visited     = set()

    main = soup.find("main")
    if not main:
        return _make_result(date, venue, city, state, country, url, songs)

    for el in main.find_all(True):
        txt = el.get_text(strip=True)
        cls = " ".join(el.get("class") or [])

        # ── Set label (S1, S2, E1 …) ──────────────────────────────────────
        if (
            el.name in ("span", "div")
            and SET_LABEL_RE.match(txt)
            and "text-content-text-tertiary" in cls
            and not el.find("a")
        ):
            current_set = txt
            position    = 1
            continue

        # ── Song link ─────────────────────────────────────────────────────
        # The <a> itself has no class; cursor-pointer lives on its parent <span>
        if el.name == "a":
            href = el.get("href", "")
            if "/songs/" not in href:
                continue

            parent_cls = " ".join(el.parent.get("class") or []) if el.parent else ""
            if "cursor-pointer" not in parent_cls:
                continue

            eid = id(el)
            if eid in visited:
                continue
            visited.add(eid)

            song_name = el.get_text(strip=True)
            slug      = href.replace("/songs/", "").strip("/")

            # ── Segue: look at siblings of the song's grandparent ─────────
            # Structure: grandparent > [parent-span > a] [segue-span ">"] [next-song …]
            segue = ""
            try:
                gp = el.parent.parent if el.parent else None
                if gp:
                    for sib in gp.find_next_siblings():
                        sib_txt = sib.get_text(strip=True)
                        sib_cls = " ".join(sib.get("class") or [])
                        if sib_txt == ">" and "text-content-text-secondary" in sib_cls:
                            segue = ">"
                            break
                        if sib.find("a", href=re.compile(r"/songs/")):
                            break
            except Exception:
                pass

            songs.append({
                "date":       date,
                "venue":      venue,
                "city":       city,
                "state":      state,
                "country":    country,
                "set":        current_set,
                "position":   position,
                "song":       song_name,
                "song_slug":  slug,
                "segue_into": segue,
                "show_url":   url,
            })
            position += 1

    return _make_result(date, venue, city, state, country, url, songs)


def _make_result(date, venue, city, state, country, url, songs):
    dt = None
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        pass
    meta = {
        "date":        date,
        "year":        dt.year           if dt else "",
        "month":       dt.month          if dt else "",
        "day":         dt.day            if dt else "",
        "day_of_week": dt.strftime("%A") if dt else "",
        "venue":       venue,
        "city":        city,
        "state":       state,
        "country":     country,
        "show_url":    url,
    }
    return {"meta": meta, "songs": songs}


# ── Step 3: Run all scrapes concurrently ──────────────────────────────────────

def scrape_all(urls: list) -> tuple:
    print(f"── Step 2: Scraping {len(urls)} show pages ({MAX_WORKERS} workers) ──")
    show_rows = []
    song_rows = []
    errors    = 0
    done      = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parse_show, u): u for u in urls}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is None:
                errors += 1
            else:
                show_rows.append(result["meta"])
                song_rows.extend(result["songs"])

            if done % 100 == 0 or done == len(urls):
                pct = done / len(urls) * 100
                print(f"  [{done:4d}/{len(urls)}] {pct:5.1f}%  |  "
                      f"{len(show_rows)} shows  |  {len(song_rows):,} songs  |  {errors} errors")

    show_rows.sort(key=lambda r: r["date"])
    print(f"\nDone: {len(show_rows)} shows, {len(song_rows):,} songs, {errors} errors\n")
    return show_rows, song_rows


# ── Step 4: Songs dimension table ─────────────────────────────────────────────

def build_songs_dim(song_rows: list) -> list:
    data = defaultdict(lambda: {
        "song": "",
        "song_slug": "",
        "total_plays": 0,
        "debut_date": "9999-99-99",
        "last_played": "0000-00-00",
        "set_counts": defaultdict(int),
        "position_sum": 0,
        "segue_count": 0,
    })

    for row in song_rows:
        slug = row["song_slug"]
        d    = data[slug]
        d["song"]       = row["song"]
        d["song_slug"]  = slug
        d["total_plays"] += 1
        if row["date"] < d["debut_date"]:
            d["debut_date"] = row["date"]
        if row["date"] > d["last_played"]:
            d["last_played"] = row["date"]
        d["set_counts"][row["set"]] += 1
        d["position_sum"] += row["position"]
        if row["segue_into"] == ">":
            d["segue_count"] += 1

    dims = []
    for slug, d in sorted(data.items(), key=lambda x: -x[1]["total_plays"]):
        most_common_set = max(d["set_counts"], key=d["set_counts"].get)
        avg_pos = round(d["position_sum"] / d["total_plays"], 1) if d["total_plays"] else ""
        segue_pct = round(d["segue_count"] / d["total_plays"] * 100, 1) if d["total_plays"] else 0
        dims.append({
            "song_slug":        slug,
            "song":             d["song"],
            "total_plays":      d["total_plays"],
            "debut_date":       d["debut_date"],
            "last_played":      d["last_played"],
            "most_common_set":  most_common_set,
            "avg_set_position": avg_pos,
            "segue_pct":        segue_pct,
        })
    return dims


# ── Step 5: Write CSVs ────────────────────────────────────────────────────────

def write_csv(path: str, rows: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    size_kb = os.path.getsize(path) // 1024
    print(f"  ✓ {os.path.basename(path):30s}  {len(rows):6,} rows   {size_kb:,} KB")


def write_all(show_rows, song_rows, songs_dim):
    print("── Step 3: Writing CSVs ────────────────────────────────────")

    song_count_by_url = Counter(r["show_url"] for r in song_rows)
    for s in show_rows:
        s["song_count"] = song_count_by_url.get(s["show_url"], 0)

    write_csv(
        os.path.join(OUTPUT_DIR, "shows.csv"),
        show_rows,
        ["date", "year", "month", "day", "day_of_week",
         "venue", "city", "state", "country", "song_count", "show_url"],
    )

    song_rows_sorted = sorted(song_rows, key=lambda r: (r["date"], r["set"], r["position"]))
    write_csv(
        os.path.join(OUTPUT_DIR, "setlists.csv"),
        song_rows_sorted,
        ["date", "venue", "city", "state", "country",
         "set", "position", "song", "song_slug", "segue_into", "show_url"],
    )

    write_csv(
        os.path.join(OUTPUT_DIR, "songs_dim.csv"),
        songs_dim,
        ["song_slug", "song", "total_plays", "debut_date",
         "last_played", "most_common_set", "avg_set_position", "segue_pct"],
    )

    print(f"\nAll files saved to: {os.path.abspath(OUTPUT_DIR)}/")


# ── Tableau notes ─────────────────────────────────────────────────────────────

TABLEAU_TIPS = """
┌──────────────────────────────────────────────────────────────────┐
│  Tableau setup                                                   │
├──────────────────────────────────────────────────────────────────┤
│  Connect all 3 CSVs, then relate them:                           │
│    setlists ──(show_url)──► shows                               │
│    setlists ──(song_slug)──► songs_dim                          │
│                                                                  │
│  Useful calculated fields:                                       │
│    Is Encore?        LEFT([set],1) = "E"                        │
│    Is Segue?         [segue_into] = ">"                         │
│    Show Year         YEAR(DATE([date]))                         │
│    Days Since Debut  DATEDIFF('day',[debut_date],[date])        │
│                                                                  │
│  Recommended views:                                              │
│    Map          city/state colored by show count                │
│    Bar          top 50 songs by total_plays (songs_dim)         │
│    Heatmap      song x year frequency                           │
│    Timeline     shows/year + avg songs per show                 │
│    Segues       song → next song (filter segue_into = ">")      │
│    Debut track  songs introduced over time by debut_date        │
└──────────────────────────────────────────────────────────────────┘
"""


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("  Disco Biscuits Scraper — discobiscuits.net")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    all_urls             = discover_all_urls()
    show_rows, song_rows = scrape_all(all_urls)
    songs_dim            = build_songs_dim(song_rows)
    write_all(show_rows, song_rows, songs_dim)

    elapsed = int(time.time() - t0)
    print(f"\nTotal time: {elapsed // 60}m {elapsed % 60}s")
    print(TABLEAU_TIPS)
