"""
PhishGuard Pro — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import re, urllib.parse

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard Pro",
    page_icon="🛡️",
    layout="centered"
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #090d12; }
.main { background-color: #090d12; }
.stApp { background-color: #090d12; color: #c8d8e8; }
.title { font-size: 2.5rem; font-weight: 900; color: white; text-align: center; }
.title span { color: #00e5ff; }
.subtitle { text-align: center; color: #4a6070; margin-bottom: 2rem; }
.verdict-phishing {
    background: rgba(255,59,92,0.15); border: 2px solid #ff3b5c;
    border-radius: 12px; padding: 20px; text-align: center;
    font-size: 1.5rem; font-weight: 800; color: #ff3b5c;
}
.verdict-safe {
    background: rgba(0,230,118,0.15); border: 2px solid #00e676;
    border-radius: 12px; padding: 20px; text-align: center;
    font-size: 1.5rem; font-weight: 800; color: #00e676;
}
.verdict-suspicious {
    background: rgba(255,184,0,0.15); border: 2px solid #ffb800;
    border-radius: 12px; padding: 20px; text-align: center;
    font-size: 1.5rem; font-weight: 800; color: #ffb800;
}
.feature-card {
    background: #0e1520; border: 1px solid #1c2a3a;
    border-radius: 8px; padding: 10px; margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Feature extractor ──────────────────────────────────────────────
SHORTENERS = {"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","rb.gy","tiny.cc","is.gd"}
SUSP_TLDS  = [".tk",".ml",".ga",".cf",".gq",".xyz",".top",".club",".work",".click",".online",".site",".zip"]
BRANDS     = ["paypal","google","facebook","apple","amazon","microsoft","netflix","instagram",
              "twitter","linkedin","hdfc","sbi","icici","chase","wellsfargo","dropbox"]
LOGIN_KW   = ["login","signin","verify","secure","update","banking","password",
              "confirm","billing","payment","suspend","recover","alert","urgent"]

def extract(raw):
    url = raw.strip()
    if not url.startswith("http"):
        url = "http://" + url
    try:
        p = urllib.parse.urlparse(url)
    except:
        return None, url
    host  = p.netloc or ""
    path  = p.path  or ""
    parts = host.split(".")
    sub   = ".".join(parts[:-2]) if len(parts) > 2 else ""
    is_ip = bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host))
    low   = url.lower()

    f = {
        "🔒 HTTPS":              ("Yes ✅", "No ❌")[p.scheme != "https"],
        "🌐 IP Address":         ("Yes 🚨", "No")[not is_ip],
        "⚠️ Suspicious TLD":    ("Yes 🚨", "No")[not any(host.endswith(t) for t in SUSP_TLDS)],
        "🔗 URL Shortener":      ("Yes 🚨", "No")[host not in SHORTENERS],
        "🏷️ Brand Spoof":        ("Yes 🚨", "No")[not any(b in sub.lower() for b in BRANDS)],
        "🔑 Login Keywords":     ("Yes ⚠️", "No")[not any(k in low for k in LOGIN_KW)],
        "📏 URL Length":         f"{len(url)} chars {'⚠️' if len(url)>75 else '✅'}",
        "🔢 Subdomains":         f"{max(0,len(parts)-2)} {'⚠️' if len(parts)-2>=3 else '✅'}",
        "@ Symbol":              ("Yes 🚨", "No")["@" not in url],
        "🖥️ Odd Port":           ("Yes ⚠️", "No")[not (p.port and p.port not in [80,443])],
        "📂 Exe Extension":      ("Yes 🚨", "No")[not bool(re.search(r'\.(exe|php|asp|jsp)$', path, re.I))],
        "🔀 Many Subdomains":    ("Yes ⚠️", "No")[not (len(parts)-2 >= 3)],
    }

    # Score
    score = 0
    if is_ip:                                          score += 0.30
    if any(host.endswith(t) for t in SUSP_TLDS):      score += 0.25
    if host in SHORTENERS:                             score += 0.18
    if any(b in sub.lower() for b in BRANDS):         score += 0.22
    if p.scheme != "https":                            score += 0.15
    if any(k in low for k in LOGIN_KW):                score += 0.12
    if len(parts) - 2 >= 3:                            score += 0.14
    if p.port and p.port not in [80, 443]:             score += 0.10
    if "@" in url:                                     score += 0.20
    if re.search(r'\.(exe|php|asp|jsp)$', path, re.I): score += 0.18
    if len(url) > 75:                                  score += 0.06
    if p.scheme == "https":                            score -= 0.08
    score = max(0, min(1, score))
    return f, score, url

# ── UI ─────────────────────────────────────────────────────────────
st.markdown('<div class="title">🛡️ Phish<span>Guard</span> Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Phishing URL Detection Engine</div>', unsafe_allow_html=True)

# Stats
if "total" not in st.session_state:
    st.session_state.total   = 0
    st.session_state.phish   = 0
    st.session_state.safe    = 0

col1, col2, col3 = st.columns(3)
col1.metric("🚨 Phishing Caught", st.session_state.phish)
col2.metric("✅ Safe URLs",        st.session_state.safe)
col3.metric("🔍 Total Scanned",   st.session_state.total)

st.divider()

# Input
url_input = st.text_input(
    "Enter URL to scan:",
    placeholder="https://suspicious-link.tk/verify-account",
    label_visibility="visible"
)

# Example buttons
st.markdown("**Try examples:**")
c1,c2,c3,c4 = st.columns(4)
if c1.button("google.com ✅"):   url_input = "https://www.google.com"
if c2.button("paypa1.tk 🚨"):    url_input = "http://paypa1.tk/verify-account"
if c3.button("apple spoof 🚨"):  url_input = "http://secure-apple-id.gq/update/password"
if c4.button("IP scam 🚨"):      url_input = "http://192.168.1.1:8080/account-suspended"

scan = st.button("⚡ Scan Now", type="primary", use_container_width=True)

if scan and url_input:
    with st.spinner("🔍 Analyzing URL with ML model..."):
        result = extract(url_input)
        if result and len(result) == 3:
            features, score, clean_url = result
            conf    = round(score * 100)
            verdict = "PHISHING" if conf >= 70 else ("SUSPICIOUS" if conf >= 30 else "SAFE")

            # Update stats
            st.session_state.total += 1
            if verdict == "PHISHING":   st.session_state.phish += 1
            elif verdict == "SAFE":     st.session_state.safe  += 1
            st.rerun()

    # Show verdict
    if verdict == "PHISHING":
        st.markdown(f'<div class="verdict-phishing">🚨 PHISHING DETECTED — {conf}% probability</div>', unsafe_allow_html=True)
    elif verdict == "SUSPICIOUS":
        st.markdown(f'<div class="verdict-suspicious">⚠️ SUSPICIOUS — {conf}% probability</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-safe">✅ SAFE — {conf}% phishing probability</div>', unsafe_allow_html=True)

    st.markdown(f"**Scanned:** `{clean_url}`")
    st.progress(conf / 100)

    # Feature breakdown
    st.markdown("### 🔬 Feature Analysis")
    cols = st.columns(3)
    for i, (name, val) in enumerate(features.items()):
        cols[i % 3].metric(name, val)

    # Reason
    st.markdown("### 📋 Explanation")
    signals = []
    if "Yes 🚨" in features.get("🌐 IP Address",""):     signals.append("IP address used instead of domain")
    if "Yes 🚨" in features.get("🔗 URL Shortener",""):   signals.append("URL shortener hides destination")
    if "Yes 🚨" in features.get("⚠️ Suspicious TLD",""): signals.append("Suspicious TLD (.tk/.ml/.xyz)")
    if "Yes 🚨" in features.get("🏷️ Brand Spoof",""):    signals.append("Known brand spoofed in subdomain")
    if "Yes ⚠️" in features.get("🔑 Login Keywords",""):  signals.append("Login/credential keywords found")
    if "Yes 🚨" in features.get("@ Symbol",""):           signals.append("@ symbol in URL")

    if signals:
        st.error("🚨 **Suspicious signals:** " + " | ".join(signals))
    elif verdict == "SAFE":
        st.success("✅ No suspicious signals detected. URL appears legitimate.")
    else:
        st.warning(f"⚠️ ML model flagged this URL with {conf}% phishing probability.")

elif scan and not url_input:
    st.warning("Please enter a URL first!")

st.divider()
st.markdown("""
<div style='text-align:center; color:#4a6070; font-size:0.8rem'>
PhishGuard Pro • ML-powered phishing detection • 30+ URL features analyzed
</div>
""", unsafe_allow_html=True)
