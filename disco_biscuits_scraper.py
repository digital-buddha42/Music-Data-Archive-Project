#!/usr/bin/env python3
"""
Disco Biscuits Full Scraper
============================
Scrapes discobiscuits.net for all shows (1995–2026) and produces
3 Tableau-ready CSV files:

  1. shows.csv          — one row per show (2,054 rows)
  2. setlists.csv       — one row per song played (~25–30k rows)
  3. songs_dim.csv      — one row per unique song (dimension table)

Usage:
  pip install requests beautifulsoup4
  python disco_biscuits_scraper.py

Output files are written to ./output/
Runtime: ~8–15 minutes (polite rate limiting included)
"""

import csv
import json
import time
import re
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependencies. Run:  pip install requests beautifulsoup4")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://discobiscuits.net"
YEARS      = list(range(1995, 2027))
OUTPUT_DIR = "./output"
MAX_WORKERS = 12        # concurrent requests — polite for the site
DELAY       = 0.05      # seconds between requests per worker
TIMEOUT     = 15        # request timeout in seconds

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research/data project; contact via GitHub)",
    "Accept": "text/html,application/xhtml+xml",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Discover all show URLs ────────────────────────────────────────────

def get_show_urls_for_year(year: int) -> list[str]:
    """Fetch the year listing page and extract individual show URLs."""
    url = f"{BASE_URL}/shows/year/{year}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] Year {year} failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all("a", href=re.compile(r"/shows/\d{4}-\d{2}-\d{2}"))
    urls = []
    for a in links:
        href = a["href"]
        # Strip fragment (#photos etc.)
        href = href.split("#")[0]
        full = href if href.startswith("http") else BASE_URL + href
        if full not in urls:
            urls.append(full)
    return urls


def discover_all_urls() -> list[str]:
    print("── Step 1: Discovering show URLs across all years ──")
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
        print(f"  {year}: {added:3d} shows  (running total: {len(all_urls)})")
        time.sleep(0.1)
    print(f"\nTotal unique show URLs: {len(all_urls)}\n")
    return all_urls


# ── Step 2: Scrape individual show pages ──────────────────────────────────────

def parse_show(url: str) -> dict | None:
    """
    Fetch a single show page and parse:
      - show metadata (date, venue, city, state, country) from JSON-LD schema
      - setlist (songs, sets, segues) from DOM anchor tags
    Returns a dict with keys: meta (dict) + songs (list of dicts).
    Returns None on failure.
    """
    time.sleep(DELAY)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Metadata from JSON-LD ──────────────────────────────────────────────
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
        return None  # skip pages without valid date

    # ── Setlist from song anchor tags ──────────────────────────────────────
    # Song links have href="/songs/<slug>" and class containing "cursor-pointer"
    # Set labels (S1, S2, E1 etc.) appear as standalone text nodes in span/div elements
    songs = []
    current_set = "S1"
    position    = 1
    visited     = set()

    main = soup.find("main")
    if not main:
        return {"meta": _meta(date, venue, city, state, country, url), "songs": songs}

    for el in main.find_all(True):  # all elements
        text = el.get_text(strip=True)

        # Detect set label elements (text is exactly S1/S2/S3/E/E1/E2)
        if re.fullmatch(r"S[1-4]|E[0-3]?", text) and not el.find("a"):
            current_set = text
            position    = 1
            continue

        # Detect song links
        if el.name == "a":
            href = el.get("href", "")
            if "/songs/" not in href:
                continue
            # Only setlist songs have cursor-pointer in class
            cls = " ".join(el.get("class", []))
            if "cursor-pointer" not in cls:
                continue
            if id(el) in visited:
                continue
            visited.add(id(el))

            song_name = el.get_text(strip=True)
            slug      = href.replace("/songs/", "").strip("/")

            # Detect segue: the grandparent's next sibling often contains ">"
            segue = ""
            try:
                parent   = el.parent
                grandpar = parent.parent if parent else None
                if grandpar:
                    nxt = grandpar.find_next_sibling()
                    if nxt and nxt.get_text(strip=True) == ">":
                        segue = ">"
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

    return {"meta": _meta(date, venue, city, state, country, url), "songs": songs}


def _meta(date, venue, city, state, country, url):
    dt = datetime.strptime(date, "%Y-%m-%d") if date else None
    return {
        "date":       date,
        "year":       dt.year  if dt else "",
        "month":      dt.month if dt else "",
        "day":        dt.day   if dt else "",
        "day_of_week": dt.strftime("%A") if dt else "",
        "venue":      venue,
        "city":       city,
        "state":      state,
        "country":    country,
        "show_url":   url,
    }


def scrape_all(urls: list[str]) -> tuple[list[dict], list[dict]]:
    """Scrape all show URLs concurrently. Returns (show_rows, song_rows)."""
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
                print(f"  [{done:4d}/{len(urls)}]  {pct:5.1f}%  |  "
                      f"{len(show_rows)} shows  |  {len(song_rows):,} songs  |  {errors} errors")

    print(f"\nScrape complete: {len(show_rows)} shows, {len(song_rows):,} songs, {errors} errors\n")
    return show_rows, song_rows


# ── Step 3: Build dimension table & derived fields ────────────────────────────

def build_songs_dim(song_rows: list[dict]) -> list[dict]:
    """
    One row per unique song slug. Includes:
      - debut date, last played date, total plays
      - most common set position, most common set (S1/S2/E)
    Tableau can join this to setlists.csv on song_slug.
    """
    from collections import defaultdict

    data = defaultdict(lambda: {
        "song": "",
        "song_slug": "",
        "total_plays": 0,
        "debut_date": "9999-99-99",
        "last_played": "0000-00-00",
        "set_counts": defaultdict(int),
        "position_sum": 0,
    })

    for row in song_rows:
        slug = row["song_slug"]
        d    = data[slug]
        d["song"]      = row["song"]
        d["song_slug"] = slug
        d["total_plays"] += 1
        if row["date"] < d["debut_date"]:
            d["debut_date"] = row["date"]
        if row["date"] > d["last_played"]:
            d["last_played"] = row["date"]
        d["set_counts"][row["set"]] += 1
        d["position_sum"] += row["position"]

    dims = []
    for slug, d in sorted(data.items(), key=lambda x: -x[1]["total_plays"]):
        most_common_set = max(d["set_counts"], key=d["set_counts"].get)
        avg_position    = round(d["position_sum"] / d["total_plays"], 1) if d["total_plays"] else ""
        dims.append({
            "song_slug":        slug,
            "song":             d["song"],
            "total_plays":      d["total_plays"],
            "debut_date":       d["debut_date"],
            "last_played":      d["last_played"],
            "most_common_set":  most_common_set,
            "avg_set_position": avg_position,
        })
    return dims


# ── Step 4: Write CSVs ────────────────────────────────────────────────────────

def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    size_kb = os.path.getsize(path) // 1024
    print(f"  ✓ {os.path.basename(path):30s}  {len(rows):6,} rows   {size_kb:,} KB")


def write_all_csvs(show_rows, song_rows, songs_dim):
    print("── Step 3: Writing CSVs ──")

    # ── shows.csv ────────────────────────────────────────────────────────────
    # Enrich with song count per show
    from collections import Counter
    song_counts = Counter(r["date"] + "|" + r["show_url"] for r in song_rows)

    show_rows_sorted = sorted(show_rows, key=lambda r: r["date"])
    for s in show_rows_sorted:
        key = s["date"] + "|" + s["show_url"]
        s["song_count"] = song_counts.get(key, 0)

    write_csv(
        os.path.join(OUTPUT_DIR, "shows.csv"),
        show_rows_sorted,
        fieldnames=["date", "year", "month", "day", "day_of_week",
                    "venue", "city", "state", "country", "song_count", "show_url"],
    )

    # ── setlists.csv ─────────────────────────────────────────────────────────
    # Sort by date → set → position for clean Tableau ordering
    song_rows_sorted = sorted(
        song_rows,
        key=lambda r: (r["date"], r["set"], r["position"])
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "setlists.csv"),
        song_rows_sorted,
        fieldnames=["date", "venue", "city", "state", "country",
                    "set", "position", "song", "song_slug", "segue_into", "show_url"],
    )

    # ── songs_dim.csv ────────────────────────────────────────────────────────
    write_csv(
        os.path.join(OUTPUT_DIR, "songs_dim.csv"),
        songs_dim,
        fieldnames=["song_slug", "song", "total_plays", "debut_date",
                    "last_played", "most_common_set", "avg_set_position"],
    )

    print(f"\nAll files saved to: {os.path.abspath(OUTPUT_DIR)}/\n")


# ── Tableau tips printed at end ───────────────────────────────────────────────

TABLEAU_TIPS = """
┌─────────────────────────────────────────────────────────────────────┐
│  Tableau setup                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  1. Connect to all 3 CSVs as separate data sources                  │
│                                                                     │
│  2. Join / relate them:                                             │
│     setlists.csv  ──(song_slug)──►  songs_dim.csv                  │
│     setlists.csv  ──(date)──────►  shows.csv                       │
│                                                                     │
│  3. Useful calculated fields:                                       │
│     • Is Segue?    [segue_into] = ">"                               │
│     • Is Encore?   LEFT([set],1) = "E"                             │
│     • Is Original? Use songs_dim to tag covers vs originals         │
│     • Gap (days)   DATEDIFF('day',[debut_date],[date])             │
│                                                                     │
│  4. Recommended viz ideas:                                          │
│     • Map: city/state by show count (shows.csv)                    │
│     • Bar: top 30 songs all time (songs_dim.total_plays)           │
│     • Heatmap: song × year play frequency (setlists + songs_dim)   │
│     • Timeline: shows per year with song_count on tooltip          │
│     • Network: segue pairs (song → next song, filtered by segue)   │
└─────────────────────────────────────────────────────────────────────┘
"""


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("  Disco Biscuits Scraper — discobiscuits.net")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    # 1. Discover
    all_urls = discover_all_urls()

    # 2. Scrape
    show_rows, song_rows = scrape_all(all_urls)

    # 3. Dimension table
    songs_dim = build_songs_dim(song_rows)

    # 4. Write
    write_all_csvs(show_rows, song_rows, songs_dim)

    elapsed = int(time.time() - t0)
    print(f"Total time: {elapsed // 60}m {elapsed % 60}s")
    print(TABLEAU_TIPS)
