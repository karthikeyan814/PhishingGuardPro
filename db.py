import os, sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "phishguard.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blacklist(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            source TEXT DEFAULT 'system',
            added_at TEXT DEFAULT(datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS reports(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            reporter_ip TEXT,
            description TEXT,
            vote_count INTEGER DEFAULT 1,
            verified INTEGER DEFAULT 0,
            submitted_at TEXT DEFAULT(datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS training_samples(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            features_json TEXT NOT NULL,
            label INTEGER NOT NULL,
            verified INTEGER DEFAULT 0,
            added_at TEXT DEFAULT(datetime('now'))
        );
    """)
    seeds = [
        "http://paypal-secure-login.tk/verify",
        "http://amazon-update-account.ml/signin",
        "http://secure-apple-id.gq/update/password",
        "http://hdfc-netbanking-secure.tk/signin",
        "http://sbi-online-banking.ml/account/verify",
        "http://chase-bank-alert.gq/update/credentials",
    ]
    conn.executemany("INSERT OR IGNORE INTO blacklist(url,source) VALUES(?,?)",
                     [(u,"seed") for u in seeds])
    conn.commit()
    conn.close()

def is_blacklisted(url):
    conn = get_conn()
    row = conn.execute("SELECT id FROM blacklist WHERE url=?", (url.strip(),)).fetchone()
    conn.close()
    return row is not None

def add_to_blacklist(url, source="community"):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO blacklist(url,source) VALUES(?,?)", (url.strip(), source))
    conn.commit()
    conn.close()

def submit_report(url, reporter_ip, description, features_json, label):
    conn = get_conn()
    ex = conn.execute("SELECT id,vote_count FROM reports WHERE url=?", (url,)).fetchone()
    if ex:
        nc = ex["vote_count"] + 1
        conn.execute("UPDATE reports SET vote_count=?,verified=? WHERE id=?",
                     (nc, 1 if nc>=3 else 0, ex["id"]))
    else:
        conn.execute("INSERT INTO reports(url,reporter_ip,description) VALUES(?,?,?)",
                     (url, reporter_ip, description))
    conn.execute("INSERT INTO training_samples(url,features_json,label) VALUES(?,?,?)",
                 (url, features_json, label))
    conn.commit()
    row = conn.execute("SELECT vote_count FROM reports WHERE url=?", (url,)).fetchone()
    if row and row["vote_count"] >= 3:
        conn.execute("INSERT OR IGNORE INTO blacklist(url,source) VALUES(?,?)", (url,"community"))
        conn.commit()
    conn.close()

def get_stats():
    conn = get_conn()
    bl   = conn.execute("SELECT COUNT(*) FROM blacklist").fetchone()[0]
    rep  = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    recent = [dict(r) for r in conn.execute(
        "SELECT url,vote_count,verified,submitted_at FROM reports ORDER BY submitted_at DESC LIMIT 10"
    ).fetchall()]
    conn.close()
    return {"blacklist_count": bl, "total_reports": rep, "recent_reports": recent}
