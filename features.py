import re, urllib.parse, ipaddress

FEATURE_NAMES = [
    "url_length","hostname_length","path_length","query_length",
    "num_dots","num_hyphens","num_underscores","num_slashes",
    "num_question_marks","num_equals","num_at","num_ampersand",
    "num_percent","uses_https","has_ip_address","has_port",
    "num_subdomains","double_slash_in_path","is_shortener",
    "suspicious_tld","domain_has_digit","has_login_keyword",
    "brand_in_subdomain","has_encoding","url_is_long",
    "url_is_very_long","many_subdomains","digit_ratio",
    "num_special_chars","path_has_exe",
]

SHORTENERS = {"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","rb.gy","tiny.cc","is.gd","buff.ly","adf.ly"}

SUSPICIOUS_TLDS = {".tk",".ml",".ga",".cf",".gq",".xyz",".top",".club",".work",".click",".link",".online",".site",".space",".zip"}

BRANDS = ["paypal","google","facebook","apple","amazon","microsoft","netflix",
          "instagram","twitter","linkedin","dropbox","ebay","bankofamerica",
          "chase","wellsfargo","hdfc","sbi","icici"]

LOGIN_KW = ["login","signin","sign-in","logon","account","verify","secure",
            "update","banking","password","credential","submit","confirm",
            "billing","payment","unlock","recover","alert","suspend"]

def extract_features(url):
    url = url.strip()
    if not url.startswith(("http://","https://")): url = "http://"+url
    try:
        p = urllib.parse.urlparse(url)
        host = p.hostname or ""
        path = p.path or ""
        query = p.query or ""
    except:
        return {k:0 for k in FEATURE_NAMES}
    f = {}
    f["url_length"]          = len(url)
    f["hostname_length"]     = len(host)
    f["path_length"]         = len(path)
    f["query_length"]        = len(query)
    f["num_dots"]            = host.count(".")
    f["num_hyphens"]         = url.count("-")
    f["num_underscores"]     = url.count("_")
    f["num_slashes"]         = url.count("/")
    f["num_question_marks"]  = url.count("?")
    f["num_equals"]          = url.count("=")
    f["num_at"]              = url.count("@")
    f["num_ampersand"]       = url.count("&")
    f["num_percent"]         = url.count("%")
    f["uses_https"]          = 1 if p.scheme=="https" else 0
    f["has_ip_address"]      = _has_ip(host)
    f["has_port"]            = 1 if (p.port and p.port not in (80,443)) else 0
    f["num_subdomains"]      = max(0,len(host.split("."))-2)
    f["double_slash_in_path"]= 1 if "//" in path else 0
    f["is_shortener"]        = 1 if host in SHORTENERS else 0
    f["suspicious_tld"]      = int(any(host.endswith(t) for t in SUSPICIOUS_TLDS))
    f["domain_has_digit"]    = int(bool(re.search(r"\d", host.split(".")[0] if host else "")))
    f["has_login_keyword"]   = int(any(kw in url.lower() for kw in LOGIN_KW))
    sub = ".".join(host.split(".")[:-2]) if len(host.split("."))>2 else ""
    f["brand_in_subdomain"]  = int(any(b in sub.lower() for b in BRANDS))
    f["has_encoding"]        = 1 if "%" in url else 0
    f["url_is_long"]         = 1 if len(url)>75 else 0
    f["url_is_very_long"]    = 1 if len(url)>150 else 0
    f["many_subdomains"]     = 1 if f["num_subdomains"]>=3 else 0
    digits = sum(c.isdigit() for c in url)
    f["digit_ratio"]         = round(digits/max(len(url),1),4)
    f["num_special_chars"]   = sum(url.count(c) for c in "!~,;:")
    f["path_has_exe"]        = int(any(path.lower().endswith(e) for e in [".exe",".php",".asp",".jsp"]))
    return f

def to_vector(f): return [f.get(k,0) for k in FEATURE_NAMES]

def _has_ip(host):
    try: ipaddress.ip_address(host); return 1
    except: return 0
