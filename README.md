# Jamband Data Archive

A personal project to build a clean, unified dataset for jam band concerts — with the long-term goal of enabling setlist analysis, song prediction, and fan-facing tools.

## Overview

This repo contains scrapers and data pipelines targeting two sources:

- **Bonnaroo/Setlist scraping** (`scrape_bip.py`, `BIP Scraping.ipynb`) — Async scraper using Playwright to extract show metadata and full setlists from discobiscuits.net. Outputs structured CSVs and a SQLite database with delta-update support (skips already-scraped shows).
- **Internet Archive API** (`Internet Archive API.ipynb`) — Queries the Internet Archive for live recordings linked to shows, enabling correlation between setlist data and available recordings.

## Data Schema

**`shows.csv`** — one row per show
| Column | Description |
|--------|-------------|
| `show_url` | Canonical URL (primary key) |
| `show_date` | ISO date (YYYY-MM-DD) |
| `venue` | Venue name |
| `city` / `state` | Location |

**`setlist_items.csv`** — one row per song played
| Column | Description |
|--------|-------------|
| `show_url` | Foreign key to shows |
| `set_name` | "Set I", "Set II", "Encore" |
| `set_index` | Position within the set |
| `song_slug` / `song_title` | Song identifier and display name |
| `segue_in` / `segue_out` | Boolean — whether the song segued |

## Tech Stack

- Python 3.10+, Pandas, Playwright (async)
- SQLite for local storage
- Internet Archive API

## Running the Scraper

```bash
pip install playwright pandas python-dateutil
playwright install chromium
python scrape_bip.py
```

The scraper supports delta mode — re-running it will only fetch new shows not already in the database.
