"""Download diverse + heavy/plus-size full-body standing photos (free-licensed, Wikimedia
Commons) into datasets/diverse_test_photos/ to stress-test SHAPE fidelity across BMI uniformly.

No API key: uses the public Commons API (generator=categorymembers) to list files in
full-body / plus-size categories, then downloads the original to datasets/diverse_test_photos/.
"""
import os
import time
import json
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
OUT = os.path.join(REPO, "datasets/diverse_test_photos")
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "meshmap-research/1.0 (anthropometry testing)"}

# categories rich in single-person, full-body standing photos across body types
CATS = [
    "Category:Full-length_portrait_photographs_of_women",
    "Category:Full-length_portrait_photographs_of_men",
    "Category:Plus-size_models",
    "Category:Standing_people",
]
PER_CAT = 6
MAXW = 1200


def api(params):
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=30))


def list_files(cat):
    r = api({"action": "query", "generator": "categorymembers", "gcmtitle": cat,
             "gcmtype": "file", "gcmlimit": 40, "prop": "imageinfo",
             "iiprop": "url|mime|size", "iiurlwidth": MAXW})
    out = []
    for p in r.get("query", {}).get("pages", {}).values():
        ii = p.get("imageinfo", [{}])[0]
        mime = ii.get("mime", "")
        if mime in ("image/jpeg", "image/png") and ii.get("thumburl"):
            out.append((p["title"], ii["thumburl"]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for cat in CATS:
        try:
            files = list_files(cat)[:PER_CAT]
        except Exception as e:
            print("cat fail", cat, e); continue
        for title, url in files:
            name = "".join(c if c.isalnum() else "_" for c in title.split(":")[-1])[:40]
            ext = ".png" if url.lower().endswith(".png") else ".jpg"
            dst = os.path.join(OUT, f"{n:02d}_{name}{ext}")
            data = None
            for attempt in range(4):                       # throttle + back off on 429
                try:
                    req = urllib.request.Request(url, headers=UA)
                    data = urllib.request.urlopen(req, timeout=60).read()
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        time.sleep(3 * (attempt + 1)); continue
                    print("dl fail", url, e); break
                except Exception as e:
                    print("dl fail", url, e); break
            if data:
                open(dst, "wb").write(data)
                print(f"got {dst} ({len(data)//1024}KB)")
                n += 1
            time.sleep(1.5)                                 # be polite to Wikimedia
    print(f"FETCH_DONE {n} images -> {OUT}")


if __name__ == "__main__":
    main()
