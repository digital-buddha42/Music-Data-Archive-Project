# Python 3.10+
# Disco Biscuits — SHOWS-first scraper (by year)
# Outputs:
#   - shows.csv              (show_url, show_date, venue, city, state)
#   - setlist_items.csv      (show_url, set_name, set_index, song_slug, song_title, segue_in, segue_out)
#   - biscuits_shows.sqlite  (tables: shows, setlist_items)
#
# Design notes:
# - Iterates /shows/year/YYYY (clean source of show links)
# - For each show page: parse header "M/D/YY at Venue – City, ST" and all Set blocks
# - Robust venue cleanup (strip leading date + 'at'/'@')
# - Set positions: order within each Set block = 1..N, segues from arrows around anchors
# - Delta mode: skip show_url already present in DB/CSVs

import asyncio, re, os, sqlite3, argparse, pathlib
from dataclasses import dataclass, asdict, fields
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime

import pandas as pd
from dateutil import parser as dtparse
from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeout

BASE = "https://discobiscuits.net"

# ---------------- Models ----------------

@dataclass
class Show:
    show_url: str
    show_date: Optional[str]  # ISO YYYY-MM-DD
    venue: Optional[str]
    city: Optional[str]
    state: Optional[str]

@dataclass
class SetItem:
    show_url: str
    set_name: Optional[str]     # "Set I", "Set II", "Encore"
    set_index: Optional[int]    # 1-based within set
    song_slug: Optional[str]
    song_title: Optional[str]
    segue_in: Optional[bool]
    segue_out: Optional[bool]

# ---------------- Utils ----------------

def clean_text(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def to_iso_date(maybe: Optional[str]) -> Optional[str]:
    if not maybe: return None
    try:
        return dtparse.parse(maybe, dayfirst=False, yearfirst=False).date().isoformat()
    except Exception:
        return None

def add_suffix(path: str, suffix: str) -> str:
    p = pathlib.Path(path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))

def safe_to_csv(df: pd.DataFrame, path: str, label: str):
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        alt = add_suffix(path, "-NEW")
        df.to_csv(alt, index=False)
        print(f"[info] {label}: '{path}' locked by another program; wrote '{alt}' instead.")

def safe_write_sqlite(db_path: str, shows_df: pd.DataFrame, items_df: pd.DataFrame):
    try:
        con = sqlite3.connect(db_path, timeout=5)
    except sqlite3.OperationalError:
        alt = add_suffix(db_path, "-NEW")
        print(f"[info] DB '{db_path}' locked; writing to '{alt}' instead.")
        con = sqlite3.connect(alt)
    shows_df.to_sql("shows", con, if_exists="replace", index=False)
    items_df.to_sql("setlist_items", con, if_exists="replace", index=False)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_show ON shows(show_url)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_show_date ON shows(show_date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_items_show ON setlist_items(show_url)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_items_set ON setlist_items(show_url, set_name, set_index)")
    con.commit(); con.close()

def strip_date_prefix_from_venue(venue: Optional[str]) -> Optional[str]:
    if not venue: return venue
    # Remove leading date and optional "at"/"@" — handles 1/2/03, 01/02/2003, etc.
    return re.sub(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*(?:at|@)?\s*", "", venue).strip()

def parse_header_line(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parse 'M/D/YY at Venue – City, ST' or 'Venue - City, ST' etc."""
    date_iso = None; venue = None; city = None; state = None

    # Try full pattern lines with a dash between venue and location
    m = re.search(r"\n\s*([^\n]+?)\s*[–-]\s*([A-Za-z .'\-]+),\s*([A-Za-z .'\-]+)\s*\n", text)
    if m:
        left = clean_text(m.group(1))
        venue = strip_date_prefix_from_venue(left)
        city  = clean_text(m.group(2))
        state = clean_text(m.group(3).upper())

    # Extract any plausible date
    for pat in (r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", r"\b(\d{4}-\d{2}-\d{2})\b", r"\b([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b"):
        d = re.search(pat, text)
        if d:
            date_iso = to_iso_date(d.group(1)) or d.group(1)
            break

    return date_iso, venue, city, state

# ---------------- Scraping ----------------

async def goto(page: Page, url: str, how: str = "domcontentloaded"):
    await page.goto(url, wait_until=how)

async def year_show_links(page: Page, year: int) -> List[str]:
    await goto(page, f"{BASE}/shows/year/{year}", how="networkidle")
    # Year pages list each show as a link to /shows/YYYY-MM-DD-...
    links = set()
    for a in await page.locator('a[href^="/shows/"]').all():
        href = await a.get_attribute("href")
        if href and re.search(r"^/shows/\d{4}-\d{2}-\d{2}-", href):
            links.add(BASE + href)
    return sorted(links)

async def parse_setlist_items(page: Page) -> List[dict]:
    """
    Walk the DOM: find headings that look like Set/Encore, take all following siblings
    up to the next heading, and extract /songs/ anchors in order.
    Also detect segues by arrows around each anchor.
    """
    return await page.evaluate("""
        () => {
          const SET = /^(Set\\s*(?:[IVX]+|\\d+)\\b|Encore\\b)/i;
          const arrow = />|→|➜|↠|->|›|»|⇢|➔|~>/;

          function isSetHeading(el){
            const t = (el.innerText || '').trim();
            return SET.test(t);
          }

          function nextSiblingsUntil(el, stopper){
            const out = [];
            let cur = el.nextElementSibling;
            while (cur){
              const t = (cur.innerText || '').trim();
              if (stopper(cur, t)) break;
              out.push(cur);
              cur = cur.nextElementSibling;
            }
            return out;
          }

          function segInFor(a){
            const prevTxt = ((a.previousSibling && a.previousSibling.textContent) || '')
                         + ((a.previousElementSibling && a.previousElementSibling.textContent) || '');
            return arrow.test(prevTxt.trim());
          }
          function segOutFor(a){
            const nextTxt = ((a.nextSibling && a.nextSibling.textContent) || '')
                         + ((a.nextElementSibling && a.nextElementSibling.textContent) || '');
            return arrow.test(nextTxt.trim());
          }

          const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4,strong,b,p,div'))
                                .filter(isSetHeading);

          const items = [];
          for (const h of headings){
            const setName = (h.innerText || '').trim().split('\\n')[0];
            const blockNodes = nextSiblingsUntil(h, (node, txt) => isSetHeading(node));
            const anchors = [];
            for (const n of blockNodes){
              anchors.push(...n.querySelectorAll('a[href^="/songs/"]'));
            }
            let idx = 0;
            for (const a of anchors){
              idx += 1;
              const href = a.getAttribute('href') || '';
              const slug = href.split('/').pop();
              const title = (a.innerText || '').trim();
              items.push({
                setName,
                setIndex: idx,
                songSlug: slug || null,
                songTitle: title || null,
                segIn: segInFor(a),
                segOut: segOutFor(a),
              });
            }
          }
          return items;
        }
    """)

async def scrape_show(page: Page, show_url: str, debug=False) -> Tuple[Show, List[SetItem]]:
    try:
        await goto(page, show_url, how="domcontentloaded")
    except PWTimeout:
        await goto(page, show_url, how="load")

    text = await page.locator("body").inner_text()
    date_iso, venue, city, state = parse_header_line(text)

    # Fallback: try to derive from URL if header not parsed
    if not (date_iso and venue and city and state):
        slug = show_url.rsplit("/", 1)[-1]
        parts = slug.split("-")
        if len(parts) >= 4:
            y, m, d = parts[:3]
            if y.isdigit() and m.isdigit() and d.isdigit():
                date_iso = date_iso or f"{y}-{m}-{d}"
            state = state or parts[-1].upper()
            city = city or parts[-2].replace("_"," ").replace("+"," ").title()
            vparts = parts[3:-2]
            vguess = " ".join(vparts).replace("_"," ").replace("+"," ").title() if vparts else None
            venue = venue or vguess
    venue = strip_date_prefix_from_venue(venue)

    items_raw = await parse_setlist_items(page)
    items: List[SetItem] = [SetItem(
        show_url=show_url,
        set_name=r.get("setName"),
        set_index=int(r.get("setIndex")) if r.get("setIndex") else None,
        song_slug=r.get("songSlug"),
        song_title=r.get("songTitle"),
        segue_in=bool(r.get("segIn")),
        segue_out=bool(r.get("segOut")),
    ) for r in (items_raw or [])]

    if debug:
        print(f"  [debug] {show_url}: date={date_iso}, venue={venue}, city={city}, state={state}, items={len(items)}")

    show = Show(show_url=show_url, show_date=date_iso, venue=venue, city=city, state=state)
    return show, items

# ---------------- Persistence ----------------

def dataclass_df(cls, rows) -> pd.DataFrame:
    cols = [f.name for f in fields(cls)]
    if not rows: return pd.DataFrame(columns=cols)
    return pd.DataFrame([asdict(r) for r in rows], columns=cols)

def load_existing(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shows = pd.DataFrame(columns=[f.name for f in fields(Show)])
    items = pd.DataFrame(columns=[f.name for f in fields(SetItem)])
    if os.path.exists(args.sqlite):
        con = sqlite3.connect(args.sqlite)
        try: shows = pd.read_sql("SELECT * FROM shows", con)
        except Exception: pass
        try: items = pd.read_sql("SELECT * FROM setlist_items", con)
        except Exception: pass
        con.close()
    else:
        if os.path.exists(args.out_shows_csv): shows = pd.read_csv(args.out_shows_csv)
        if os.path.exists(args.out_items_csv): items = pd.read_csv(args.out_items_csv)
    return shows, items

# ---------------- Main ----------------

async def main():
    ap = argparse.ArgumentParser(description="Disco Biscuits — build a shows/setlists database from year pages.")
    ap.add_argument("--years", default="1995-{}".format(datetime.now().year),
                    help="Year or range, e.g. 2003 or 1995-2025")
    ap.add_argument("--delta", action="store_true", help="Skip shows already in DB/CSVs")
    ap.add_argument("--out_shows_csv", default="shows.csv")
    ap.add_argument("--out_items_csv", default="setlist_items.csv")
    ap.add_argument("--sqlite", default="biscuits_shows.sqlite")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Parse years
    rng = args.years.strip()
    if "-" in rng:
        a, b = rng.split("-", 1)
        years = list(range(int(a), int(b) + 1))
    else:
        years = [int(rng)]
    years = [y for y in years if 1995 <= y <= datetime.now().year]

    existing_shows_df, existing_items_df = load_existing(args)
    already = set(existing_shows_df["show_url"]) if args.delta and not existing_shows_df.empty else set()

    shows: List[Show] = []
    items: List[SetItem] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()

        # Collect all show URLs from year pages
        all_show_urls: List[str] = []
        for y in years:
            try:
                urls = await year_show_links(page, y)
                print(f"[year {y}] found {len(urls)} shows")
                all_show_urls.extend(urls)
            except Exception as e:
                print(f"[warn] year {y}: {e}")

        # Filter delta
        todo = [u for u in sorted(set(all_show_urls)) if u not in already]
        print(f"Scraping {len(todo)} show pages{ ' (delta)' if args.delta else '' } ...")

        sem = asyncio.Semaphore(args.concurrency)
        async def work(u: str):
            async with sem:
                p = await ctx.new_page()
                try:
                    sh, it = await scrape_show(p, u, debug=args.debug)
                    shows.append(sh)
                    items.extend(it)
                except Exception as e:
                    print(f"[warn] {u}: {e}")
                finally:
                    await p.close()

        await asyncio.gather(*[work(u) for u in todo])

        await ctx.close(); await browser.close()

    # Build DataFrames
    shows_df = dataclass_df(Show, shows)
    items_df = dataclass_df(SetItem, items)

    # Normalize date
    if not shows_df.empty:
        shows_df["show_date"] = shows_df["show_date"].map(lambda x: to_iso_date(x) or x)

    # Write
    safe_to_csv(shows_df, args.out_shows_csv, "shows.csv")
    safe_to_csv(items_df, args.out_items_csv, "setlist_items.csv")
    safe_write_sqlite(args.sqlite, shows_df, items_df)
    print(f"Done. Rows: shows={len(shows_df)}, setlist_items={len(items_df)}")

if __name__ == "__main__":
    asyncio.run(main())
