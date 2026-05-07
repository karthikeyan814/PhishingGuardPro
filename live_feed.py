"""
PhishGuard Pro — live_feed.py
==============================
Fetches REAL phishing URLs from 3 live threat intelligence feeds:
  1. OpenPhish  — openphish.com/feed.txt         (updates every 12h)
  2. URLhaus    — urlhaus.abuse.ch/downloads/...  (updates every 5min)
  3. PhishTank  — checkphishing.org mirror        (updates every hour)

Usage:
    python live_feed.py           # fetch once now
    python live_feed.py --loop    # fetch every 6 hours forever (for server)
"""

import os, sys, time, csv, json, sqlite3, argparse, io
import urllib.request, urllib.error

BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "phishguard.db")

# ── Feed URLs ──────────────────────────────────────────────────────
FEEDS = {
    "openphish": "https://openphish.com/feed.txt",
    "urlhaus":   "https://urlhaus.abuse.ch/downloads/text/",
    "phishtank": "http://data.phishtank.com/data/online-valid.csv",
}

HEADERS = {
    "User-Agent": "PhishGuardPro/1.0 (academic research; contact: phishguardpro@example.com)"
}

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables():
    """Make sure live_feed table exists."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blacklist(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            source TEXT DEFAULT 'system',
            added_at TEXT DEFAULT(datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS live_feed_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed TEXT NOT NULL,
            urls_fetched INTEGER DEFAULT 0,
            urls_new INTEGER DEFAULT 0,
            fetched_at TEXT DEFAULT(datetime('now')),
            status TEXT DEFAULT 'ok',
            error TEXT
        );
    """)
    conn.commit()
    conn.close()

def bulk_insert(urls, source):
    """Insert list of URLs into blacklist, skip duplicates. Returns count of new ones."""
    if not urls:
        return 0
    conn  = get_conn()
    new   = 0
    batch = []
    for url in urls:
        url = url.strip()
        if url and url.startswith("http"):
            batch.append((url, source))
    # Use INSERT OR IGNORE for speed
    cursor = conn.executemany(
        "INSERT OR IGNORE INTO blacklist(url, source) VALUES(?, ?)", batch
    )
    new = cursor.rowcount
    conn.commit()
    conn.close()
    return new

def log_fetch(feed, fetched, new, status="ok", error=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO live_feed_log(feed,urls_fetched,urls_new,status,error) VALUES(?,?,?,?,?)",
        (feed, fetched, new, status, error)
    )
    conn.commit()
    conn.close()

def fetch_url(url, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

# ── Feed parsers ───────────────────────────────────────────────────

def fetch_openphish():
    """OpenPhish: one URL per line."""
    print("  🌐 Fetching OpenPhish feed...")
    try:
        text = fetch_url(FEEDS["openphish"])
        urls = [line.strip() for line in text.splitlines()
                if line.strip().startswith("http")]
        new  = bulk_insert(urls, "openphish")
        log_fetch("openphish", len(urls), new)
        print(f"     ✅ OpenPhish: {len(urls)} fetched, {new} new added to blacklist")
        return len(urls), new
    except Exception as e:
        log_fetch("openphish", 0, 0, "error", str(e))
        print(f"     ⚠️  OpenPhish failed: {e}")
        return 0, 0

def fetch_urlhaus():
    """URLhaus: lines starting with http, lines with # are comments."""
    print("  🌐 Fetching URLhaus feed...")
    try:
        text = fetch_url(FEEDS["urlhaus"])
        urls = [line.strip() for line in text.splitlines()
                if line.strip().startswith("http")]
        new  = bulk_insert(urls, "urlhaus")
        log_fetch("urlhaus", len(urls), new)
        print(f"     ✅ URLhaus:   {len(urls)} fetched, {new} new added to blacklist")
        return len(urls), new
    except Exception as e:
        log_fetch("urlhaus", 0, 0, "error", str(e))
        print(f"     ⚠️  URLhaus failed: {e}")
        return 0, 0

def fetch_phishtank():
    """PhishTank CSV: columns include url, verified, valid."""
    print("  🌐 Fetching PhishTank feed...")
    try:
        text  = fetch_url(FEEDS["phishtank"], timeout=30)
        reader = csv.DictReader(io.StringIO(text))
        urls  = []
        for row in reader:
            url = row.get("url", "").strip()
            if url.startswith("http"):
                urls.append(url)
        new = bulk_insert(urls, "phishtank")
        log_fetch("phishtank", len(urls), new)
        print(f"     ✅ PhishTank: {len(urls)} fetched, {new} new added to blacklist")
        return len(urls), new
    except Exception as e:
        log_fetch("phishtank", 0, 0, "error", str(e))
        print(f"     ⚠️  PhishTank failed: {e}")
        return 0, 0

def fetch_all():
    """Fetch all 3 feeds and return totals."""
    print("\n" + "═"*50)
    print("  PhishGuard Pro — Live Feed Sync")
    print("═"*50)
    ensure_tables()

    total_fetched, total_new = 0, 0
    for fn in [fetch_openphish, fetch_urlhaus, fetch_phishtank]:
        f, n = fn()
        total_fetched += f
        total_new     += n

    # Show blacklist size
    conn = get_conn()
    bl_count = conn.execute("SELECT COUNT(*) FROM blacklist").fetchone()[0]
    conn.close()

    print(f"\n  📊 Summary:")
    print(f"     Total fetched : {total_fetched:,}")
    print(f"     New added     : {total_new:,}")
    print(f"     Blacklist size: {bl_count:,} URLs")
    print("═"*50 + "\n")
    return total_fetched, total_new

def get_real_training_data(limit=5000):
    """
    Pull real phishing URLs from blacklist for training.
    Returns list of (url, label=1) tuples.
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT url FROM blacklist ORDER BY RANDOM() LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [(r["url"], 1) for r in rows]

def get_blacklist_count():
    try:
        conn = get_conn()
        c = conn.execute("SELECT COUNT(*) FROM blacklist").fetchone()[0]
        conn.close()
        return c
    except:
        return 0

# ── Main loop (for server / Render deployment) ─────────────────────
def loop_forever(interval_hours=6):
    print(f"🔄 Live feed loop started — syncing every {interval_hours}h")
    while True:
        fetch_all()
        print(f"  ⏳ Next sync in {interval_hours} hours...")
        time.sleep(interval_hours * 3600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run forever every 6 hours")
    parser.add_argument("--hours", type=int, default=6, help="Loop interval in hours")
    args = parser.parse_args()

    if args.loop:
        loop_forever(args.hours)
    else:
        fetch_all()
