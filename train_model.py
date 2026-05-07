"""
PhishGuard Pro — train_model.py
================================
Generates 10,000 labelled samples and trains a Gradient Boosting
classifier that is saved as  model.pkl  (the exact path app.py expects).

Run:
    python train_model.py

The script can also be triggered automatically by app.py every 500
community reports via subprocess.Popen().
"""

import os, sys, json, sqlite3, random, re, string
import urllib.parse, ipaddress
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)

# ── Import your own feature extractor ────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from features import extract_features, to_vector, FEATURE_NAMES
from db import get_conn, DB_PATH

MODEL_PATH = os.path.join(BASE, "model.pkl")
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — SYNTHETIC DATA GENERATOR  (5 000 legit + 5 000 phish)
# ═══════════════════════════════════════════════════════════════════

LEGIT_DOMAINS = [
    "google.com","facebook.com","amazon.com","microsoft.com","apple.com",
    "paypal.com","netflix.com","instagram.com","twitter.com","linkedin.com",
    "youtube.com","github.com","stackoverflow.com","wikipedia.org","reddit.com",
    "yahoo.com","ebay.com","walmart.com","chase.com","bankofamerica.com",
    "wellsfargo.com","citibank.com","hdfc.com","sbi.co.in","icicibank.com",
    "dropbox.com","adobe.com","salesforce.com","zoom.us","spotify.com",
]

LEGIT_PATHS = [
    "/","/home","/about","/contact","/products","/services",
    "/login","/signup","/account/dashboard","/help","/search",
    "/blog/security-tips","/support","/news","/offers",
]

PHISHING_TRICKS = [
    "paypa1","g00gle","arnazon","rnicrosott","app1e","facebok",
    "netfl1x","lnstagram","twltter","linkedln","yah00","ebay-secure",
    "amazon-verify","paypal-update","apple-id-locked","microsoft-alert",
    "facebook-security","chase-bank-verify","wellsfargo-secure","citibank-login",
    "hdfc-netbanking","sbi-online","icici-bank-secure","paypal-resolution",
]

PHISHING_TLDS = [
    ".tk",".ml",".ga",".cf",".gq",".xyz",".top",
    ".club",".online",".site",".space",".zip",".work",".click",
]

PHISHING_PATHS = [
    "/verify-account","/confirm-identity","/secure-login","/update-payment",
    "/account-suspended","/urgent-action","/click-here-now","/free-prize",
    "/your-account-locked","/reset-password-now","/verify-now/step1",
    "/limited-offer/claim","/invoice.php","/document.asp","/signin.jsp",
    "/banking/update/credentials","/account/verify/billing",
]

def _rstr(n): return ''.join(random.choices(string.ascii_lowercase+string.digits, k=n))
def _rip():   return ".".join(str(random.randint(1,254)) for _ in range(4))

def _gen_legit():
    domain    = random.choice(LEGIT_DOMAINS)
    path      = random.choice(LEGIT_PATHS)
    scheme    = "https" if random.random() > 0.04 else "http"
    subdomain = random.choice(["","www.","m.","shop.","mail.","secure."])
    return f"{scheme}://{subdomain}{domain}{path}"

def _gen_phishing():
    strategy = random.randint(1, 7)

    if strategy == 1:                               # typosquatting
        fake = random.choice(PHISHING_TRICKS)
        tld  = random.choice(PHISHING_TLDS)
        return f"http://{fake}{tld}{random.choice(PHISHING_PATHS)}"

    elif strategy == 2:                             # brand in subdomain (your feature!)
        brand  = random.choice(LEGIT_DOMAINS).split(".")[0]
        sub    = random.choice(["secure","verify","update","login","account"])
        tld    = random.choice(PHISHING_TLDS)
        return f"http://{brand}.{sub}{tld}{random.choice(PHISHING_PATHS)}"

    elif strategy == 3:                             # raw IP address
        port = random.choice(["",":8080",":8443",":9090"])
        return f"http://{_rip()}{port}{random.choice(PHISHING_PATHS)}"

    elif strategy == 4:                             # many subdomains
        brand  = random.choice(LEGIT_DOMAINS)
        subs   = ".".join(_rstr(6) for _ in range(random.randint(3,5)))
        return f"http://{subs}.{brand}{random.choice(PHISHING_PATHS)}"

    elif strategy == 5:                             # @ symbol trick
        brand = random.choice(LEGIT_DOMAINS)
        fake  = _rstr(12)
        return f"http://{brand}@{fake}.com{random.choice(PHISHING_PATHS)}"

    elif strategy == 6:                             # URL shortener abuse
        shorteners = ["bit.ly","tinyurl.com","t.co","ow.ly","rb.gy"]
        path = "/" + _rstr(7)
        return f"http://{random.choice(shorteners)}{path}"

    else:                                           # exe/php payload path
        domain = _rstr(random.randint(8,18))
        tld    = random.choice(PHISHING_TLDS)
        payload= random.choice([".exe",".php",".asp",".jsp"])
        return f"http://{domain}{tld}/download/file{payload}"


def generate_synthetic(n_each=5000):
    print(f"  🌐 Generating {n_each:,} legitimate URLs ...")
    legit = [(_gen_legit(), 0) for _ in range(n_each)]
    print(f"  🚨 Generating {n_each:,} phishing  URLs ...")
    phish = [(_gen_phishing(), 1) for _ in range(n_each)]
    return legit + phish


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 — PULL COMMUNITY REPORTS FROM db.py's SQLite DB
# ═══════════════════════════════════════════════════════════════════

def load_community_samples():
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT url, label FROM training_samples WHERE verified=1"
        ).fetchall()
        conn.close()
        return [(r["url"], r["label"]) for r in rows]
    except Exception as e:
        print(f"  ⚠️  Could not load community samples: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 — FEATURE EXTRACTION (uses YOUR features.py)
# ═══════════════════════════════════════════════════════════════════

def build_matrix(samples):
    X, y = [], []
    bad  = 0
    for url, label in samples:
        try:
            feats = extract_features(url)
            vec   = to_vector(feats)
            X.append(vec)
            y.append(label)
        except Exception:
            bad += 1
    if bad:
        print(f"  ⚠️  Skipped {bad} malformed URLs.")
    return np.array(X), np.array(y)


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 — TRAIN + COMPARE 5 MODELS
# ═══════════════════════════════════════════════════════════════════

def train_and_compare(X_train, X_test, X_train_sc, X_test_sc, y_train, y_test):
    models = {
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                       max_depth=5, random_state=RANDOM_STATE), False),
        "Random Forest":     (
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False),
        "Decision Tree":     (
            DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE), False),
        "SVM":               (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), True),
        "Logistic Regression":(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
    }

    results, trained = [], {}
    print("\n" + "═"*56)
    print("  TRAINING ALL 5 MODELS")
    print("═"*56)

    for name, (clf, scaled) in models.items():
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc  if scaled else X_test
        print(f"\n  🔄 {name} ...")
        clf.fit(Xtr, y_train)
        yp  = clf.predict(Xte)
        acc = accuracy_score(y_test, yp)  * 100
        prec= precision_score(y_test, yp) * 100
        rec = recall_score(y_test, yp)    * 100
        f1  = f1_score(y_test, yp)        * 100
        print(f"     Accuracy={acc:.2f}%  Precision={prec:.2f}%  Recall={rec:.2f}%  F1={f1:.2f}%")
        results.append({"name":name,"acc":acc,"prec":prec,"rec":rec,"f1":f1,
                        "clf":clf,"scaled":scaled})
        trained[name] = (clf, scaled)

    return results


def print_comparison(results):
    print("\n" + "═"*56)
    print("  MODEL COMPARISON")
    print("═"*56)
    print(f"  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("  " + "-"*50)
    for r in sorted(results, key=lambda x: x["acc"], reverse=True):
        crown = " 🏆" if r == max(results, key=lambda x: x["acc"]) else ""
        print(f"  {r['name']:<22} {r['acc']:>6.2f}% {r['prec']:>6.2f}% "
              f"{r['rec']:>6.2f}% {r['f1']:>6.2f}%{crown}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 — SAVE BEST MODEL AS  model.pkl  (what app.py loads)
# ═══════════════════════════════════════════════════════════════════

def save_best(results):
    best = max(results, key=lambda x: x["acc"])
    clf  = best["clf"]
    joblib.dump(clf, MODEL_PATH)
    print(f"\n  ✅ Saved → {MODEL_PATH}")
    print(f"  Best model : {best['name']}  ({best['acc']:.2f}% accuracy)")
    return best


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6 — FEATURE IMPORTANCE (top 10)
# ═══════════════════════════════════════════════════════════════════

def show_importance(clf):
    if not hasattr(clf, "feature_importances_"):
        return
    print("\n" + "═"*56)
    print("  TOP 10 FEATURES")
    print("═"*56)
    fi = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in fi:
        bar = "█" * int(imp * 300)
        print(f"  {feat:<28} {imp:.4f}  {bar}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7 — QUICK DEMO using YOUR extract_features()
# ═══════════════════════════════════════════════════════════════════

DEMO_URLS = [
    ("https://www.google.com/search?q=phishing",           "SAFE"),
    ("https://amazon.com/products/laptop",                 "SAFE"),
    ("http://paypa1.tk/verify-account",                    "PHISHING"),
    ("http://secure-apple-id.gq/update/password",          "PHISHING"),
    ("http://192.168.1.1:8080/account-suspended",          "PHISHING"),
    ("http://hdfc.secure-login.ml/banking/credentials",    "PHISHING"),
    ("https://github.com/topics/machine-learning",         "SAFE"),
    ("http://bit.ly/xK9pQ2",                               "PHISHING"),
]

def run_demo(clf):
    print("\n" + "═"*56)
    print("  LIVE PREDICTIONS (using your features.py)")
    print("═"*56)
    labels_map = {0:"✅ SAFE    ", 1:"🚨 PHISHING"}
    for url, expected in DEMO_URLS:
        feats = extract_features(url)
        vec   = np.array([to_vector(feats)])
        prob  = float(clf.predict_proba(vec)[0][1]) * 100
        pred  = 1 if prob >= 70 else 0
        match = "✓" if labels_map[pred].strip().startswith(expected[:4]) else "✗"
        print(f"  {match} {labels_map[pred]}  {prob:5.1f}%  {url[:52]}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*56)
    print("█  PhishGuard Pro — Training Pipeline                 █")
    print("█"*56)

    # 1. Gather data
    print("\n📦 Building dataset ...")
    samples  = generate_synthetic(n_each=5000)
    community = load_community_samples()
    if community:
        print(f"  ➕ Adding {len(community):,} verified community reports.")
        samples += community
    random.shuffle(samples)
    print(f"  Total samples: {len(samples):,}")

    # 2. Feature matrix
    print("\n🔬 Extracting features via features.py ...")
    X, y = build_matrix(samples)
    print(f"  Matrix shape : {X.shape}  |  Phishing: {y.sum():,}  |  Legit: {(y==0).sum():,}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)
    X_te_sc   = scaler.transform(X_test)

    # 4. Train
    results = train_and_compare(X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test)

    # 5. Compare
    print_comparison(results)

    # 6. Save best → model.pkl
    print("\n" + "═"*56)
    print("  SAVING MODEL")
    print("═"*56)
    best = save_best(results)

    # 7. Feature importance
    show_importance(best["clf"])

    # 8. Demo
    run_demo(best["clf"])

    print("\n" + "█"*56)
    print("█  Done!  app.py will auto-load model.pkl on startup.  █")
    print("█"*56 + "\n")
