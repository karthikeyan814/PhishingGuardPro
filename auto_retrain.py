"""
PhishGuard Pro — auto_retrain.py
==================================
Retrains the ML model by combining:
  1. Real phishing URLs from live_feed blacklist
  2. Real legitimate URLs (curated list)
  3. Community-reported URLs from db.py
  4. Synthetic generated URLs (fallback if real data is low)

Run:
    python auto_retrain.py           # retrain once now
    python auto_retrain.py --loop    # retrain every 24 hours
"""

import os, sys, time, random, string, argparse
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from features import extract_features, to_vector, FEATURE_NAMES
from live_feed import get_real_training_data, get_blacklist_count, fetch_all

MODEL_PATH   = os.path.join(BASE, "model.pkl")
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Real legitimate URLs (top global websites) ─────────────────────
LEGIT_URLS = [
    "https://www.google.com", "https://www.youtube.com", "https://www.facebook.com",
    "https://www.amazon.com", "https://www.wikipedia.org", "https://www.twitter.com",
    "https://www.instagram.com", "https://www.linkedin.com", "https://www.reddit.com",
    "https://www.github.com", "https://www.microsoft.com", "https://www.apple.com",
    "https://www.netflix.com", "https://www.stackoverflow.com", "https://www.paypal.com",
    "https://www.ebay.com", "https://www.walmart.com", "https://www.chase.com",
    "https://www.bankofamerica.com", "https://www.wellsfargo.com",
    "https://www.dropbox.com", "https://www.adobe.com", "https://www.zoom.us",
    "https://www.slack.com", "https://www.spotify.com", "https://www.twitch.tv",
    "https://www.pinterest.com", "https://www.quora.com", "https://www.medium.com",
    "https://www.nytimes.com", "https://www.bbc.com", "https://www.cnn.com",
    "https://www.hdfc.com", "https://www.sbi.co.in", "https://www.icicibank.com",
    "https://mail.google.com/mail", "https://drive.google.com",
    "https://docs.google.com", "https://www.office.com", "https://outlook.live.com",
    "https://www.amazon.com/products/laptop", "https://www.google.com/search?q=ai",
    "https://github.com/topics/machine-learning", "https://stackoverflow.com/questions",
    "https://www.wikipedia.org/wiki/Phishing", "https://www.youtube.com/watch?v=abc",
    "https://www.linkedin.com/in/profile", "https://www.reddit.com/r/python",
    "https://www.apple.com/iphone", "https://www.microsoft.com/windows",
]

# ── Synthetic phishing generator (fallback) ────────────────────────
PHISHING_TRICKS = [
    "paypa1","g00gle","arnazon","rnicrosott","app1e","facebok","netfl1x",
    "lnstagram","twltter","linkedln","yah00","ebay-secure","amazon-verify",
    "paypal-update","apple-id-locked","microsoft-alert","hdfc-netbanking",
    "sbi-online","icici-bank-secure","chase-bank-verify",
]
PHISHING_TLDS  = [".tk",".ml",".ga",".cf",".gq",".xyz",".top",".club",".online",".site"]
PHISHING_PATHS = [
    "/verify-account","/confirm-identity","/secure-login","/update-payment",
    "/account-suspended","/urgent-action","/free-prize","/your-account-locked",
    "/reset-password-now","/verify-now/step1","/invoice.php","/signin.asp",
]

def _rstr(n): return ''.join(random.choices(string.ascii_lowercase+string.digits, k=n))
def _rip():   return ".".join(str(random.randint(1,254)) for _ in range(4))

def _gen_phishing():
    s = random.randint(1, 6)
    if s == 1:
        return f"http://{random.choice(PHISHING_TRICKS)}{random.choice(PHISHING_TLDS)}{random.choice(PHISHING_PATHS)}"
    elif s == 2:
        brand = random.choice(["paypal","apple","amazon","microsoft","hdfc","sbi"])
        sub   = random.choice(["secure","verify","update","login"])
        return f"http://{brand}.{sub}{random.choice(PHISHING_TLDS)}{random.choice(PHISHING_PATHS)}"
    elif s == 3:
        return f"http://{_rip()}:{random.choice([8080,8443,9090])}{random.choice(PHISHING_PATHS)}"
    elif s == 4:
        brand = random.choice(["google.com","paypal.com","apple.com"])
        return f"http://{_rstr(12)}.{brand}.{_rstr(5)}.com{random.choice(PHISHING_PATHS)}"
    elif s == 5:
        return f"http://{random.choice(['google.com','paypal.com'])}@{_rstr(12)}.com{random.choice(PHISHING_PATHS)}"
    else:
        return f"http://{_rstr(14)}{random.choice(PHISHING_TLDS)}{random.choice(PHISHING_PATHS)}"

def _gen_legit():
    import random
    base = random.choice(LEGIT_URLS)
    return base

# ── Community reports from DB ──────────────────────────────────────
def load_community():
    try:
        import sqlite3
        db = os.path.join(BASE, "phishguard.db")
        if not os.path.exists(db):
            return []
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT url, label FROM training_samples WHERE verified=1"
        ).fetchall()
        conn.close()
        return [(r["url"], r["label"]) for r in rows]
    except:
        return []

# ── Build full training dataset ────────────────────────────────────
def build_dataset():
    samples = []

    # 1. Real phishing from live feeds
    bl_count = get_blacklist_count()
    if bl_count > 100:
        print(f"  📥 Loading real phishing URLs from blacklist ({bl_count:,} available)...")
        real_phish = get_real_training_data(limit=min(bl_count, 5000))
        samples += real_phish
        print(f"     ✅ {len(real_phish):,} real phishing URLs loaded")
    else:
        print(f"  ⚠️  Blacklist has only {bl_count} URLs — run live_feed.py first")
        print("     Falling back to synthetic phishing data...")

    # 2. Synthetic phishing as top-up (always add some for variety)
    n_synth = max(0, 3000 - len([s for s in samples if s[1]==1]))
    synth_phish = [(_gen_phishing(), 1) for _ in range(n_synth)]
    samples += synth_phish
    print(f"  🔧 {n_synth:,} synthetic phishing URLs added for variety")

    # 3. Real legitimate URLs (expanded)
    legit_base = [(url, 0) for url in LEGIT_URLS]
    # Expand by adding path variations
    legit_expanded = []
    paths = ["/","/about","/contact","/products","/login","/signup",
             "/account","/dashboard","/help","/search","/blog","/news"]
    for url, _ in legit_base:
        for path in random.sample(paths, 4):
            legit_expanded.append((url.rstrip("/")+path, 0))
    samples += legit_expanded
    print(f"  ✅ {len(legit_expanded):,} legitimate URLs loaded")

    # 4. Community reports
    community = load_community()
    if community:
        samples += community
        print(f"  👥 {len(community):,} verified community reports added")

    random.shuffle(samples)
    print(f"\n  📊 Total dataset: {len(samples):,} samples")
    print(f"     Phishing : {sum(1 for _,l in samples if l==1):,}")
    print(f"     Legit    : {sum(1 for _,l in samples if l==0):,}")
    return samples

# ── Feature extraction ─────────────────────────────────────────────
def build_matrix(samples):
    X, y, bad = [], [], 0
    for url, label in samples:
        try:
            vec = to_vector(extract_features(url))
            X.append(vec)
            y.append(label)
        except:
            bad += 1
    if bad:
        print(f"  ⚠️  Skipped {bad} malformed URLs")
    return np.array(X), np.array(y)

# ── Train & evaluate ───────────────────────────────────────────────
def train(X_train, X_test, y_train, y_test):
    models = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08,
            max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
    }

    print("\n" + "═"*50)
    print("  TRAINING MODELS")
    print("═"*50)
    best_clf, best_acc, best_name = None, 0, ""
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        acc  = accuracy_score(y_test, clf.predict(X_test)) * 100
        prec = precision_score(y_test, clf.predict(X_test)) * 100
        rec  = recall_score(y_test, clf.predict(X_test)) * 100
        f1   = f1_score(y_test, clf.predict(X_test)) * 100
        flag = " 🏆" if acc > best_acc else ""
        print(f"  {name:<22} Acc={acc:.2f}%  Prec={prec:.2f}%  Rec={rec:.2f}%  F1={f1:.2f}%{flag}")
        if acc > best_acc:
            best_acc, best_clf, best_name = acc, clf, name
    return best_clf, best_name, best_acc

# ── Save ───────────────────────────────────────────────────────────
def save(clf, name, acc):
    joblib.dump(clf, MODEL_PATH)
    print(f"\n  ✅ Model saved → {MODEL_PATH}")
    print(f"  Best: {name}  ({acc:.2f}% accuracy)")

# ── Main ───────────────────────────────────────────────────────────
def retrain():
    print("\n" + "█"*50)
    print("█  PhishGuard Pro — Auto Retrain Pipeline        █")
    print("█"*50)

    # Step 1: Fetch fresh live data first
    print("\n🔄 Step 1: Syncing live phishing feeds...")
    fetch_all()

    # Step 2: Build dataset
    print("📦 Step 2: Building dataset...")
    samples = build_dataset()

    # Step 3: Feature matrix
    print("\n🔬 Step 3: Extracting features...")
    X, y = build_matrix(samples)
    print(f"  Matrix: {X.shape}")

    # Step 4: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Step 5: Train
    clf, name, acc = train(X_train, X_test, y_train, y_test)

    # Step 6: Save
    save(clf, name, acc)

    print("\n" + "█"*50)
    print("█  Retrain Complete! Model is now up to date.    █")
    print("█"*50 + "\n")
    return acc

def loop_forever(interval_hours=24):
    print(f"🔄 Auto-retrain loop started — retraining every {interval_hours}h")
    while True:
        retrain()
        print(f"  ⏳ Next retrain in {interval_hours} hours...")
        time.sleep(interval_hours * 3600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop",  action="store_true", help="Retrain every 24 hours")
    parser.add_argument("--hours", type=int, default=24, help="Interval in hours")
    args = parser.parse_args()
    if args.loop:
        loop_forever(args.hours)
    else:
        retrain()
