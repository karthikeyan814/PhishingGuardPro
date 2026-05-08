
import os, sys, json, subprocess, time
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from features import extract_features, to_vector, FEATURE_NAMES
from db import init_db, is_blacklisted, submit_report, add_to_blacklist, get_stats

MODEL_PATH    = os.path.join(BASE, "model.pkl")
RETRAIN_EVERY = 500
app   = Flask(__name__)
CORS(app)
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded!")
    else:
        print("model.pkl not found — run train_model.py first")

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PhishGuard Pro</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#090d12;--surface:#0e1520;--border:#1c2a3a;--accent:#00e5ff;--danger:#ff3b5c;--warn:#ffb800;--safe:#00e676;--text:#c8d8e8;--muted:#4a6070;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;}
.wrap{max-width:920px;margin:0 auto;padding:0 24px 80px}
header{padding:40px 0 28px;display:flex;align-items:center;gap:16px;border-bottom:1px solid var(--border);margin-bottom:44px;flex-wrap:wrap}
.logo-icon{width:50px;height:50px;background:linear-gradient(135deg,var(--accent),#0077ff);border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0}
.logo-text h1{font-size:1.7rem;font-weight:800;color:#fff}
.logo-text h1 span{color:var(--accent)}
.logo-text p{font-size:.72rem;color:var(--muted);margin-top:2px;font-family:'Space Mono',monospace}
.badge-online{margin-left:auto;display:flex;align-items:center;gap:7px;font-size:.7rem;font-family:'Space Mono',monospace;color:var(--safe)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--safe);}
.hero{text-align:center;margin-bottom:36px}
.hero h2{font-size:clamp(1.7rem,5vw,2.6rem);font-weight:800;color:#fff;line-height:1.15;margin-bottom:10px}
.hero h2 em{font-style:normal;color:var(--accent)}
.hero p{color:var(--muted);font-size:.95rem;max-width:480px;margin:0 auto}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:28px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 20px;text-align:center}
.stat-n{font-size:1.5rem;font-weight:800;font-family:'Space Mono',monospace;color:#fff}
.stat-n.d{color:var(--danger)}.stat-n.s{color:var(--safe)}
.stat-l{font-size:.65rem;color:var(--muted);margin-top:3px;letter-spacing:1px;text-transform:uppercase}
.scan-box{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:28px;margin-bottom:28px;}
.lbl{font-size:.68rem;font-family:'Space Mono',monospace;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}
.url-row{display:flex;gap:10px;align-items:stretch}
.url-input{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:12px;padding:13px 16px;color:#fff;font-family:'Space Mono',monospace;font-size:.88rem;outline:none;}
.url-input::placeholder{color:var(--muted)}
.scan-btn{background:linear-gradient(135deg,var(--accent),#0099dd);border:none;border-radius:12px;padding:13px 26px;color:#000;font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;cursor:pointer;white-space:nowrap}
.scan-btn:disabled{opacity:.45;cursor:not-allowed}
.examples{margin-top:14px;display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.examples span{font-size:.68rem;color:var(--muted);font-family:'Space Mono',monospace}
.ex-btn{background:rgba(0,229,255,.05);border:1px solid var(--border);border-radius:7px;padding:3px 11px;font-size:.68rem;color:var(--accent);font-family:'Space Mono',monospace;cursor:pointer}
.result{display:none;background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:28px;margin-bottom:28px}
.result.on{display:block}
.vrow{display:flex;align-items:center;gap:18px;margin-bottom:24px;flex-wrap:wrap}
.vbadge{padding:9px 22px;border-radius:50px;font-weight:800;font-size:1rem;letter-spacing:1px;text-transform:uppercase;display:flex;align-items:center;gap:7px}
.vbadge.phishing{background:rgba(255,59,92,.15);color:var(--danger);border:1px solid var(--danger)}
.vbadge.safe{background:rgba(0,230,118,.15);color:var(--safe);border:1px solid var(--safe)}
.vbadge.suspicious{background:rgba(255,184,0,.15);color:var(--warn);border:1px solid var(--warn)}
.conf-blk{flex:1;min-width:160px}
.conf-lbl{font-size:.67rem;font-family:'Space Mono',monospace;color:var(--muted);margin-bottom:5px}
.conf-wrap{height:7px;background:var(--border);border-radius:4px;overflow:hidden;margin-bottom:4px}
.conf-bar{height:100%;border-radius:4px;transition:width 1s ease;}
.conf-bar.phishing{background:linear-gradient(90deg,var(--warn),var(--danger))}
.conf-bar.safe{background:linear-gradient(90deg,#00b09b,var(--safe))}
.conf-bar.suspicious{background:linear-gradient(90deg,#ff8c00,var(--warn))}
.conf-pct{font-size:.82rem;font-family:'Space Mono',monospace;color:#fff;font-weight:700}
.scanned-url{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:11px 15px;font-family:'Space Mono',monospace;font-size:.78rem;color:var(--text);word-break:break-all;margin-bottom:18px}
.reason{padding:13px 15px;border-radius:10px;margin-bottom:18px;font-size:.86rem;line-height:1.65}
.reason.phishing{background:rgba(255,59,92,.08);border-left:3px solid var(--danger)}
.reason.safe{background:rgba(0,230,118,.08);border-left:3px solid var(--safe)}
.reason.suspicious{background:rgba(255,184,0,.08);border-left:3px solid var(--warn)}
.feat-ttl{font-size:.67rem;font-family:'Space Mono',monospace;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}
.feat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:9px}
.fc{background:var(--bg);border:1px solid var(--border);border-radius:9px;padding:11px 13px}
.fc.bad{border-color:var(--danger);background:rgba(255,59,92,.05)}
.fn{font-size:.65rem;color:var(--muted);font-family:'Space Mono',monospace;margin-bottom:3px}
.fv{font-size:.92rem;font-weight:700;color:#fff}
.fv.bad{color:var(--danger)}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="logo-icon">🛡️</div>
  <div class="logo-text">
    <h1>Phish<span>Guard</span> Pro</h1>
    <p>REAL-TIME PHISHING DETECTION ENGINE</p>
  </div>
  <div class="badge-online"><div class="dot"></div><span>ML ENGINE ONLINE</span></div>
</header>
<div class="hero">
  <h2>Detect <em>Phishing URLs</em><br/>Before They Catch You</h2>
  <p>Paste any suspicious URL — our ML model analyzes 30+ signals in milliseconds.</p>
</div>
<div class="stats">
  <div class="stat"><div class="stat-n d" id="sPhish">0</div><div class="stat-l">Phishing Caught</div></div>
  <div class="stat"><div class="stat-n s" id="sSafe">0</div><div class="stat-l">Safe URLs</div></div>
  <div class="stat"><div class="stat-n" id="sTotal">0</div><div class="stat-l">Total Scanned</div></div>
</div>
<div class="scan-box">
  <div class="lbl">// ENTER URL TO SCAN</div>
  <div class="url-row">
    <input class="url-input" id="urlInput" type="text" placeholder="https://suspicious-link.tk/verify-account" autocomplete="off" spellcheck="false"/>
    <button class="scan-btn" id="scanBtn" onclick="runScan()">⚡ Scan Now</button>
  </div>
  <div class="examples">
    <span>Try:</span>
    <button class="ex-btn" onclick="tryEx('https://www.google.com')">google.com ✅</button>
    <button class="ex-btn" onclick="tryEx('http://paypa1.tk/verify-account')">paypa1.tk 🚨</button>
    <button class="ex-btn" onclick="tryEx('http://secure-apple-id.gq/update/password')">apple spoof 🚨</button>
    <button class="ex-btn" onclick="tryEx('http://hdfc-netbanking-secure.tk/signin')">bank spoof 🚨</button>
  </div>
</div>
<div class="result" id="result">
  <div class="vrow">
    <div class="vbadge" id="vbadge"></div>
    <div class="conf-blk">
      <div class="conf-lbl">PHISHING PROBABILITY</div>
      <div class="conf-wrap"><div class="conf-bar" id="confBar"></div></div>
      <div class="conf-pct" id="confPct"></div>
    </div>
  </div>
  <div class="scanned-url" id="scannedUrl"></div>
  <div class="reason" id="reason"></div>
  <div class="feat-ttl">// FEATURE ANALYSIS</div>
  <div class="feat-grid" id="featGrid"></div>
</div>
</div>
<script>
let stats={phish:0,safe:0,total:0};
async function runScan(){
  const url=document.getElementById("urlInput").value.trim();
  if(!url){alert("Please enter a URL!");return;}
  const btn=document.getElementById("scanBtn");
  btn.disabled=true;btn.textContent="Scanning...";
  try{
    const res=await fetch("/api/scan",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url})});
    const d=await res.json();
    if(d.error){alert(d.error);return;}
    showResult(url,d.features,d.confidence,d.verdict,d.verdict.toLowerCase(),d.reason);
    if(d.verdict==="PHISHING")stats.phish++;else if(d.verdict==="SAFE")stats.safe++;stats.total++;
    document.getElementById("sPhish").textContent=stats.phish;
    document.getElementById("sSafe").textContent=stats.safe;
    document.getElementById("sTotal").textContent=stats.total;
  }catch(e){alert("Cannot reach backend!");}
  btn.disabled=false;btn.textContent="⚡ Scan Now";
}
function showResult(url,f,conf,verdict,cls,reason){
  const icons={phishing:"🚨",suspicious:"⚠️",safe:"✅"};
  document.getElementById("vbadge").className=`vbadge ${cls}`;
  document.getElementById("vbadge").innerHTML=`${icons[cls]||"❓"} ${verdict}`;
  const cb=document.getElementById("confBar");cb.className=`conf-bar ${cls}`;
  setTimeout(()=>{cb.style.width=conf+"%";},60);
  document.getElementById("confPct").textContent=`${conf}% phishing probability`;
  document.getElementById("scannedUrl").textContent=url;
  const rb=document.getElementById("reason");rb.className=`reason ${cls}`;rb.textContent=reason;
  if(f){
    const SHOW=[
      {k:"uses_https",l:"HTTPS",fmt:v=>v?"Yes ✅":"No ❌",bad:v=>!v},
      {k:"has_ip_address",l:"IP Address",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"suspicious_tld",l:"Suspicious TLD",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"is_shortener",l:"URL Shortener",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"brand_in_subdomain",l:"Brand Spoof",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"has_login_keyword",l:"Login Keywords",fmt:v=>v?"Yes ⚠️":"No",bad:v=>v},
      {k:"url_length",l:"URL Length",fmt:v=>`${v} chars`,bad:v=>v>75},
      {k:"num_subdomains",l:"Subdomains",fmt:v=>v,bad:v=>v>=3},
    ];
    document.getElementById("featGrid").innerHTML=SHOW.map(s=>{
      const v=f[s.k]??0,isBad=s.bad(v);
      return`<div class="fc ${isBad?"bad":""}"><div class="fn">${s.l}</div><div class="fv ${isBad?"bad":""}">${s.fmt(v)}</div></div>`;
    }).join("");
  }
  document.getElementById("result").classList.add("on");
}
function tryEx(url){document.getElementById("urlInput").value=url;runScan();}
document.getElementById("urlInput").addEventListener("keydown",e=>{if(e.key==="Enter")runScan();});
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": model is not None})

@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if is_blacklisted(url):
        return jsonify({"url":url,"verdict":"PHISHING","confidence":100,
                        "reason":"⛔ Known phishing URL — blacklist match","features":{}})
    features = extract_features(url)
    vec = np.array([to_vector(features)])
    if model is None:
        return jsonify({"error": "Model not ready. Run train_model.py first."}), 503
    prob    = float(model.predict_proba(vec)[0][1])
    conf    = round(prob * 100, 1)
    verdict = "PHISHING" if conf >= 70 else ("SUSPICIOUS" if conf >= 30 else "SAFE")
    if conf >= 90:
        add_to_blacklist(url, "auto")
    return jsonify({"url":url,"verdict":verdict,"confidence":conf,
                    "reason":_reason(features,verdict,conf),"features":features})

@app.route("/api/report", methods=["POST"])
def report():
    data  = request.get_json(silent=True) or {}
    url   = (data.get("url") or "").strip()
    label = int(data.get("label", 1))
    desc  = data.get("description", "")
    if not url:
        return jsonify({"error": "No URL"}), 400
    features = extract_features(url)
    submit_report(url, request.remote_addr, desc, json.dumps(features), label)
    st = get_stats()
    if st["total_reports"] % RETRAIN_EVERY == 0:
        subprocess.Popen([sys.executable, os.path.join(BASE, "train_model.py")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify({"message": f"Report saved! Total: {st['total_reports']}", **st})

@app.route("/api/stats")
def stats_route():
    st = get_stats()
    st["model_loaded"] = model is not None
    return jsonify(st)

def _reason(f, verdict, conf):
    clues = []
    if f.get("has_ip_address"):     clues.append("IP address instead of domain")
    if f.get("is_shortener"):       clues.append("URL shortener hides destination")
    if f.get("suspicious_tld"):     clues.append("Suspicious TLD (.tk/.ml/.xyz/.gq)")
    if f.get("brand_in_subdomain"): clues.append("Brand spoofed in subdomain")
    if f.get("has_login_keyword"):  clues.append("Login/credential keywords in URL")
    if f.get("url_is_very_long"):   clues.append("Extremely long URL")
    if f.get("many_subdomains"):    clues.append("Excessive subdomains (3+)")
    if not f.get("uses_https"):     clues.append("No HTTPS encryption")
    if f.get("num_at", 0) > 0:     clues.append("@ symbol in URL")
    if not clues:
        return ("✅ No suspicious signals detected." if verdict == "SAFE"
                else f"⚠️ ML model flagged this URL ({conf:.0f}% confidence).")
    return "🚨 Signals: " + "; ".join(clues) + f". ({conf:.0f}% phishing probability)"

# Initialize for gunicorn (runs at import time)
init_db()
load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)