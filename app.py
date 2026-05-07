"""
PhishGuard Pro — app.py
========================
Local : python app.py  → opens http://localhost:5000
Online: Deployed on Render.com → https://phishguard-pro.onrender.com
"""

import os, sys, json, subprocess, threading, webbrowser, time
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from features import extract_features, to_vector, FEATURE_NAMES
from db import init_db, is_blacklisted, submit_report, add_to_blacklist, get_stats

MODEL_PATH    = os.path.join(BASE, "model.pkl")
RETRAIN_EVERY = 500
app   = Flask(__name__)
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"  ✅ Model loaded from {MODEL_PATH}")
    else:
        print("  ⚠️  model.pkl not found — run train_model.py first")

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PhishGuard Pro</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#090d12;--surface:#0e1520;--border:#1c2a3a;
  --accent:#00e5ff;--danger:#ff3b5c;--warn:#ffb800;--safe:#00e676;
  --text:#c8d8e8;--muted:#4a6070;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(0,229,255,.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(0,229,255,.03) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0}
body::after{content:'';position:fixed;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);
  animation:scan 4s linear infinite;pointer-events:none;z-index:1;opacity:.4}
@keyframes scan{0%{top:0}100%{top:100vh}}
.wrap{position:relative;z-index:2;max-width:920px;margin:0 auto;padding:0 24px 80px}
header{padding:40px 0 28px;display:flex;align-items:center;gap:16px;
  border-bottom:1px solid var(--border);margin-bottom:44px;flex-wrap:wrap}
.logo-icon{width:50px;height:50px;background:linear-gradient(135deg,var(--accent),#0077ff);
  border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:24px;
  box-shadow:0 0 24px rgba(0,229,255,.3);flex-shrink:0}
.logo-text h1{font-size:1.7rem;font-weight:800;color:#fff}
.logo-text h1 span{color:var(--accent)}
.logo-text p{font-size:.72rem;color:var(--muted);margin-top:2px;font-family:'Space Mono',monospace}
.badge-online{margin-left:auto;display:flex;align-items:center;gap:7px;
  font-size:.7rem;font-family:'Space Mono',monospace;color:var(--safe)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--safe);animation:pdot 2s ease-in-out infinite}
@keyframes pdot{0%,100%{box-shadow:0 0 0 0 rgba(0,230,118,.5)}50%{box-shadow:0 0 0 6px rgba(0,230,118,0)}}
.hero{text-align:center;margin-bottom:36px}
.hero h2{font-size:clamp(1.7rem,5vw,2.6rem);font-weight:800;color:#fff;line-height:1.15;margin-bottom:10px}
.hero h2 em{font-style:normal;color:var(--accent)}
.hero p{color:var(--muted);font-size:.95rem;max-width:480px;margin:0 auto}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:28px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 20px;text-align:center}
.stat-n{font-size:1.5rem;font-weight:800;font-family:'Space Mono',monospace;color:#fff}
.stat-n.d{color:var(--danger)}.stat-n.s{color:var(--safe)}
.stat-l{font-size:.65rem;color:var(--muted);margin-top:3px;letter-spacing:1px;text-transform:uppercase}
.scan-box{background:var(--surface);border:1px solid var(--border);border-radius:20px;
  padding:28px;margin-bottom:28px;position:relative;overflow:hidden;transition:border-color .3s}
.scan-box::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:.5}
.scan-box.scanning{border-color:var(--accent)}
.scan-box.r-phishing{border-color:var(--danger)}
.scan-box.r-safe{border-color:var(--safe)}
.scan-box.r-suspicious{border-color:var(--warn)}
.lbl{font-size:.68rem;font-family:'Space Mono',monospace;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}
.url-row{display:flex;gap:10px;align-items:stretch}
.url-input{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:12px;
  padding:13px 16px;color:#fff;font-family:'Space Mono',monospace;font-size:.88rem;
  outline:none;transition:border-color .2s,box-shadow .2s}
.url-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(0,229,255,.1)}
.url-input::placeholder{color:var(--muted)}
.scan-btn{background:linear-gradient(135deg,var(--accent),#0099dd);border:none;
  border-radius:12px;padding:13px 26px;color:#000;font-family:'Syne',sans-serif;
  font-weight:700;font-size:.9rem;cursor:pointer;white-space:nowrap;
  display:flex;align-items:center;gap:6px;transition:transform .15s,box-shadow .15s,opacity .15s}
.scan-btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 8px 24px rgba(0,229,255,.3)}
.scan-btn:disabled{opacity:.45;cursor:not-allowed}
.examples{margin-top:14px;display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.examples span{font-size:.68rem;color:var(--muted);font-family:'Space Mono',monospace}
.ex-btn{background:rgba(0,229,255,.05);border:1px solid var(--border);border-radius:7px;
  padding:3px 11px;font-size:.68rem;color:var(--accent);font-family:'Space Mono',monospace;
  cursor:pointer;transition:background .2s,border-color .2s}
.ex-btn:hover{background:rgba(0,229,255,.12);border-color:var(--accent)}
.prog{display:none;align-items:center;gap:12px;margin-top:18px;padding:14px;
  background:rgba(0,229,255,.05);border-radius:10px;border:1px solid rgba(0,229,255,.15)}
.prog.on{display:flex}
.prog-bar{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.prog-fill{height:100%;width:0%;background:linear-gradient(90deg,var(--accent),#0099dd);
  border-radius:2px;transition:width .15s linear}
.prog-txt{font-size:.7rem;font-family:'Space Mono',monospace;color:var(--accent);white-space:nowrap}
.result{display:none;background:var(--surface);border:1px solid var(--border);
  border-radius:20px;padding:28px;margin-bottom:28px;animation:fadeUp .4s ease}
.result.on{display:block}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.vrow{display:flex;align-items:center;gap:18px;margin-bottom:24px;flex-wrap:wrap}
.vbadge{padding:9px 22px;border-radius:50px;font-weight:800;font-size:1rem;
  letter-spacing:1px;text-transform:uppercase;display:flex;align-items:center;gap:7px}
.vbadge.phishing{background:rgba(255,59,92,.15);color:var(--danger);border:1px solid var(--danger)}
.vbadge.safe{background:rgba(0,230,118,.15);color:var(--safe);border:1px solid var(--safe)}
.vbadge.suspicious{background:rgba(255,184,0,.15);color:var(--warn);border:1px solid var(--warn)}
.conf-blk{flex:1;min-width:160px}
.conf-lbl{font-size:.67rem;font-family:'Space Mono',monospace;color:var(--muted);margin-bottom:5px}
.conf-wrap{height:7px;background:var(--border);border-radius:4px;overflow:hidden;margin-bottom:4px}
.conf-bar{height:100%;border-radius:4px;transition:width 1s cubic-bezier(.2,.8,.3,1)}
.conf-bar.phishing{background:linear-gradient(90deg,var(--warn),var(--danger))}
.conf-bar.safe{background:linear-gradient(90deg,#00b09b,var(--safe))}
.conf-bar.suspicious{background:linear-gradient(90deg,#ff8c00,var(--warn))}
.conf-pct{font-size:.82rem;font-family:'Space Mono',monospace;color:#fff;font-weight:700}
.scanned-url{background:var(--bg);border:1px solid var(--border);border-radius:10px;
  padding:11px 15px;font-family:'Space Mono',monospace;font-size:.78rem;color:var(--text);
  word-break:break-all;margin-bottom:18px;display:flex;align-items:flex-start;gap:9px}
.reason{padding:13px 15px;border-radius:10px;margin-bottom:18px;font-size:.86rem;line-height:1.65}
.reason.phishing{background:rgba(255,59,92,.08);border-left:3px solid var(--danger)}
.reason.safe{background:rgba(0,230,118,.08);border-left:3px solid var(--safe)}
.reason.suspicious{background:rgba(255,184,0,.08);border-left:3px solid var(--warn)}
.feat-ttl{font-size:.67rem;font-family:'Space Mono',monospace;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}
.feat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:9px}
.fc{background:var(--bg);border:1px solid var(--border);border-radius:9px;padding:11px 13px}
.fc.bad{border-color:var(--danger);background:rgba(255,59,92,.05)}
.fc.good{border-color:rgba(0,230,118,.25)}
.fn{font-size:.65rem;color:var(--muted);font-family:'Space Mono',monospace;margin-bottom:3px}
.fv{font-size:.92rem;font-weight:700;color:#fff}
.fv.bad{color:var(--danger)}.fv.good{color:var(--safe)}
.hist{margin-top:8px}
.sec-ttl{font-size:.67rem;font-family:'Space Mono',monospace;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;
  display:flex;align-items:center;gap:8px}
.sec-ttl::after{content:'';flex:1;height:1px;background:var(--border)}
.hist-list{display:flex;flex-direction:column;gap:8px}
.hi{display:flex;align-items:center;gap:11px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:11px 15px;
  cursor:pointer;transition:border-color .2s}
.hi:hover{border-color:var(--accent)}
.hb{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.hb.phishing{background:var(--danger)}.hb.safe{background:var(--safe)}.hb.suspicious{background:var(--warn)}
.hu{flex:1;font-size:.78rem;font-family:'Space Mono',monospace;color:var(--text);
  overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.hv{font-size:.7rem;font-weight:700}
.hv.phishing{color:var(--danger)}.hv.safe{color:var(--safe)}.hv.suspicious{color:var(--warn)}
.hc{font-size:.68rem;font-family:'Space Mono',monospace;color:var(--muted)}
.how{margin-top:36px}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-top:14px}
.step{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px}
.snum{width:30px;height:30px;border-radius:7px;background:rgba(0,229,255,.1);
  border:1px solid rgba(0,229,255,.3);display:flex;align-items:center;justify-content:center;
  font-family:'Space Mono',monospace;font-size:.75rem;color:var(--accent);margin-bottom:11px;font-weight:700}
.step h4{font-size:.88rem;font-weight:700;color:#fff;margin-bottom:5px}
.step p{font-size:.75rem;color:var(--muted);line-height:1.6}
.toast{position:fixed;bottom:22px;right:22px;background:var(--surface);
  border:1px solid var(--border);border-radius:11px;padding:12px 18px;
  font-size:.82rem;color:var(--text);transform:translateY(70px);opacity:0;
  transition:all .3s ease;z-index:100}
.toast.show{transform:translateY(0);opacity:1}
@media(max-width:580px){
  .url-row{flex-direction:column}
  .stats{grid-template-columns:1fr 1fr}
  .vrow{flex-direction:column;align-items:flex-start}
}
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
  <div class="stat"><div class="stat-n"   id="sTotal">0</div><div class="stat-l">Total Scanned</div></div>
</div>
<div class="scan-box" id="scanBox">
  <div class="lbl">// ENTER URL TO SCAN</div>
  <div class="url-row">
    <input class="url-input" id="urlInput" type="text"
      placeholder="https://suspicious-link.tk/verify-account"
      autocomplete="off" spellcheck="false"/>
    <button class="scan-btn" id="scanBtn" onclick="runScan()">⚡ Scan Now</button>
  </div>
  <div class="examples">
    <span>Try:</span>
    <button class="ex-btn" onclick="tryEx('https://www.google.com')">google.com ✅</button>
    <button class="ex-btn" onclick="tryEx('http://paypa1.tk/verify-account')">paypa1.tk 🚨</button>
    <button class="ex-btn" onclick="tryEx('http://secure-apple-id.gq/update/password')">apple spoof 🚨</button>
    <button class="ex-btn" onclick="tryEx('http://192.168.1.1:8080/account-suspended')">IP scam 🚨</button>
    <button class="ex-btn" onclick="tryEx('https://github.com/topics/machine-learning')">github.com ✅</button>
    <button class="ex-btn" onclick="tryEx('http://hdfc-netbanking-secure.tk/signin')">bank spoof 🚨</button>
  </div>
  <div class="prog" id="prog">
    <div class="prog-bar"><div class="prog-fill" id="progFill"></div></div>
    <div class="prog-txt" id="progTxt">SCANNING...</div>
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
  <div class="feat-ttl">// FEATURE ANALYSIS (30 signals)</div>
  <div class="feat-grid" id="featGrid"></div>
</div>
<div class="hist" id="histSection" style="display:none">
  <div class="sec-ttl">SCAN HISTORY</div>
  <div class="hist-list" id="histList"></div>
</div>
<div class="how">
  <div class="sec-ttl">HOW IT WORKS</div>
  <div class="steps">
    <div class="step"><div class="snum">01</div><h4>URL Parsing</h4><p>Breaks down domain, path, query, protocol and subdomain structure.</p></div>
    <div class="step"><div class="snum">02</div><h4>30+ Features</h4><p>Checks HTTPS, IP usage, TLD, brand spoofing, keywords, encoding and more.</p></div>
    <div class="step"><div class="snum">03</div><h4>ML Model</h4><p>Gradient Boosting trained on 10,000 URLs gives a precise phishing score.</p></div>
    <div class="step"><div class="snum">04</div><h4>Verdict</h4><p>SAFE &lt;30% · SUSPICIOUS 30–69% · PHISHING ≥70% with full explanation.</p></div>
  </div>
</div>
</div>
<div class="toast" id="toast"></div>
<script>
let stats={phish:0,safe:0,total:0},history=[];
const SHORTENERS=new Set(["bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","rb.gy","tiny.cc","is.gd"]);
const SUSP_TLDS=[".tk",".ml",".ga",".cf",".gq",".xyz",".top",".club",".work",".click",".online",".site",".zip"];
const BRANDS=["paypal","google","facebook","apple","amazon","microsoft","netflix","instagram","twitter","linkedin","hdfc","sbi","icici","chase","wellsfargo"];
const LOGIN_KW=["login","signin","verify","secure","update","banking","password","confirm","billing","payment","suspend","recover"];
function jsExtract(raw){
  let url=raw.trim();
  if(!url.startsWith("http"))url="http://"+url;
  let p;try{p=new URL(url);}catch{return null;}
  const host=p.hostname||"",path=p.pathname||"",parts=host.split(".");
  const sub=parts.length>2?parts.slice(0,-2).join("."):"";
  const isIp=/^\d{1,3}(\.\d{1,3}){3}$/.test(host),low=url.toLowerCase();
  return{url_length:url.length,hostname_length:host.length,path_length:path.length,
    num_dots:(host.match(/\./g)||[]).length,num_hyphens:(url.match(/-/g)||[]).length,
    num_at:(url.match(/@/g)||[]).length,uses_https:p.protocol==="https:"?1:0,
    has_ip_address:isIp?1:0,has_port:(p.port&&p.port!=="80"&&p.port!=="443")?1:0,
    num_subdomains:Math.max(0,parts.length-2),is_shortener:SHORTENERS.has(host)?1:0,
    suspicious_tld:SUSP_TLDS.some(t=>host.endsWith(t))?1:0,
    domain_has_digit:/\d/.test(parts[0]||"")?1:0,
    has_login_keyword:LOGIN_KW.some(k=>low.includes(k))?1:0,
    brand_in_subdomain:BRANDS.some(b=>sub.toLowerCase().includes(b))?1:0,
    url_is_long:url.length>75?1:0,url_is_very_long:url.length>150?1:0,
    many_subdomains:(parts.length-2)>=3?1:0,
    path_has_exe:/\.(exe|php|asp|jsp)$/i.test(path)?1:0,
    num_special_chars:(url.match(/[!~,;:]/g)||[]).length,
    digit_ratio:+((url.replace(/[^0-9]/g,"").length/Math.max(url.length,1)).toFixed(4))};
}
function jsScore(f){
  if(!f)return 0.5;let s=0;
  if(f.has_ip_address)s+=0.30;if(f.suspicious_tld)s+=0.25;if(f.is_shortener)s+=0.18;
  if(f.brand_in_subdomain)s+=0.22;if(!f.uses_https)s+=0.15;if(f.has_login_keyword)s+=0.12;
  if(f.many_subdomains)s+=0.14;if(f.has_port)s+=0.10;if(f.num_at>0)s+=0.20;
  if(f.path_has_exe)s+=0.18;if(f.domain_has_digit)s+=0.08;if(f.url_is_very_long)s+=0.06;
  if(f.num_hyphens>3)s+=0.06;if(f.num_subdomains>3)s+=0.10;
  if(f.uses_https)s-=0.08;if(f.num_subdomains<=1)s-=0.04;
  return Math.max(0,Math.min(1,s));
}
function buildReason(f,verdict,conf){
  const sig=[];
  if(f.has_ip_address)sig.push("IP address instead of domain");
  if(f.is_shortener)sig.push("URL shortener hides destination");
  if(f.suspicious_tld)sig.push("Suspicious TLD (.tk/.ml/.xyz)");
  if(f.brand_in_subdomain)sig.push("Brand spoofed in subdomain");
  if(f.has_login_keyword)sig.push("Login/credential keywords found");
  if(f.many_subdomains)sig.push("Excessive subdomains (3+)");
  if(!f.uses_https)sig.push("No HTTPS encryption");
  if(f.num_at>0)sig.push("@ symbol in URL");
  if(f.path_has_exe)sig.push("Executable file extension");
  if(f.has_port)sig.push("Non-standard port");
  if(!sig.length)return verdict==="SAFE"?"✅ No suspicious signals detected. URL appears legitimate.":`⚠️ ML model flagged this URL (${conf}% probability).`;
  return"🚨 Signals: "+sig.join("; ")+".";
}
async function runScan(){
  const url=document.getElementById("urlInput").value.trim();
  if(!url){showToast("Please enter a URL!");return;}
  const btn=document.getElementById("scanBtn");
  btn.disabled=true;
  document.getElementById("result").classList.remove("on");
  document.getElementById("scanBox").className="scan-box scanning";
  const prog=document.getElementById("prog"),fill=document.getElementById("progFill"),txt=document.getElementById("progTxt");
  prog.classList.add("on");
  for(const[pct,label]of[[15,"PARSING URL..."],[35,"EXTRACTING FEATURES..."],[60,"CHECKING BLACKLIST..."],[80,"RUNNING ML MODEL..."],[95,"COMPUTING SCORE..."],[100,"DONE ✓"]]){
    await delay(220+Math.random()*150);fill.style.width=pct+"%";txt.textContent=label;
  }
  await delay(180);prog.classList.remove("on");fill.style.width="0%";
  let conf,verdict,features,reason;
  try{
    const res=await fetch("/api/scan",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url})});
    const d=await res.json();
    conf=d.confidence;verdict=d.verdict;features=d.features||jsExtract(url);reason=d.reason;
  }catch{
    features=jsExtract(url);conf=Math.round(jsScore(features)*100);
    verdict=conf>=70?"PHISHING":conf>=30?"SUSPICIOUS":"SAFE";reason=buildReason(features,verdict,conf);
  }
  const cls=verdict.toLowerCase();
  document.getElementById("scanBox").className=`scan-box r-${cls}`;
  showResult(url,features,conf,verdict,cls,reason);
  if(cls==="phishing")stats.phish++;else if(cls==="safe")stats.safe++;stats.total++;
  document.getElementById("sPhish").textContent=stats.phish;
  document.getElementById("sSafe").textContent=stats.safe;
  document.getElementById("sTotal").textContent=stats.total;
  history.unshift({url,verdict,conf,cls});if(history.length>10)history.pop();
  renderHistory();btn.disabled=false;
}
function showResult(url,f,conf,verdict,cls,reason){
  const icons={phishing:"🚨",suspicious:"⚠️",safe:"✅"};
  document.getElementById("vbadge").className=`vbadge ${cls}`;
  document.getElementById("vbadge").innerHTML=`${icons[cls]} ${verdict}`;
  const cb=document.getElementById("confBar");cb.className=`conf-bar ${cls}`;cb.style.width="0%";
  setTimeout(()=>{cb.style.width=conf+"%";},60);
  document.getElementById("confPct").textContent=`${conf}% phishing probability`;
  document.getElementById("scannedUrl").innerHTML=`<span>🔗</span><span>${esc(url)}</span>`;
  const rb=document.getElementById("reason");rb.className=`reason ${cls}`;rb.textContent=reason;
  if(f){
    const SHOW=[
      {k:"uses_https",l:"HTTPS",fmt:v=>v?"Yes ✅":"No ❌",bad:v=>!v},
      {k:"has_ip_address",l:"IP Address",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"suspicious_tld",l:"Suspicious TLD",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"is_shortener",l:"URL Shortener",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"brand_in_subdomain",l:"Brand Spoof",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"has_login_keyword",l:"Login Keywords",fmt:v=>v?"Yes ⚠️":"No",bad:v=>v},
      {k:"many_subdomains",l:"Many Subdomains",fmt:v=>v?"Yes ⚠️":"No",bad:v=>v},
      {k:"num_at",l:"@ Symbol",fmt:v=>v>0?`${v} 🚨`:"No",bad:v=>v>0},
      {k:"path_has_exe",l:"Exe Extension",fmt:v=>v?"Yes 🚨":"No",bad:v=>v},
      {k:"has_port",l:"Odd Port",fmt:v=>v?"Yes ⚠️":"No",bad:v=>v},
      {k:"url_length",l:"URL Length",fmt:v=>`${v} chars`,bad:v=>v>75},
      {k:"num_subdomains",l:"Subdomains",fmt:v=>v,bad:v=>v>=3},
    ];
    document.getElementById("featGrid").innerHTML=SHOW.map(s=>{
      const v=f[s.k]??0,isBad=s.bad(v);
      return`<div class="fc ${isBad?"bad":v?"good":""}"><div class="fn">${s.l}</div><div class="fv ${isBad?"bad":v?"good":""}">${s.fmt(v)}</div></div>`;
    }).join("");
  }
  document.getElementById("result").classList.add("on");
}
function renderHistory(){
  if(!history.length)return;
  document.getElementById("histSection").style.display="block";
  document.getElementById("histList").innerHTML=history.map(h=>
    `<div class="hi" onclick="reScan('${esc(h.url)}')"><div class="hb ${h.cls}"></div><div class="hu">${esc(h.url)}</div><div class="hv ${h.cls}">${h.verdict}</div><div class="hc">${h.conf}%</div></div>`
  ).join("");
}
function tryEx(url){document.getElementById("urlInput").value=url;runScan();}
function reScan(url){document.getElementById("urlInput").value=url;runScan();}
function delay(ms){return new Promise(r=>setTimeout(r,ms));}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function showToast(msg){const t=document.getElementById("toast");t.textContent=msg;t.classList.add("show");setTimeout(()=>t.classList.remove("show"),2800);}
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
    if f.get("url_is_very_long"):   clues.append("Extremely long URL — obfuscation")
    if f.get("many_subdomains"):    clues.append("Excessive subdomains (3+)")
    if not f.get("uses_https"):     clues.append("No HTTPS encryption")
    if f.get("num_at", 0) > 0:     clues.append("@ symbol in URL")
    if f.get("path_has_exe"):       clues.append("Executable extension in path")
    if not clues:
        return ("✅ No suspicious signals detected." if verdict == "SAFE"
                else f"⚠️ ML model flagged this URL ({conf:.0f}% confidence).")
    return "🚨 Signals: " + "; ".join(clues) + f". ({conf:.0f}% phishing probability)"

if __name__ == "__main__":
    init_db()
    load_model()
    # Detect if running on Render or locally
    is_render = os.environ.get("RENDER") == "true"
    port = int(os.environ.get("PORT", 5000))
    if not is_render:
        # Local: open browser automatically
        def open_browser():
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=open_browser, daemon=True).start()
        print(f"\n  🌐 Opening → http://localhost:{port}")
    else:
        print(f"\n  🌐 Running on Render → port {port}")
    print("  🛑 Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=False)
