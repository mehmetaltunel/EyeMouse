import requests
import re
from packaging import version
import sys
import webbrowser

GITHUB_REPO = "mehmetaltunel/EyeMouse"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

def check_for_updates(current_version_str):
    """
    GitHub'dan son surumu kontrol eder.
    (yeni_surum_var_mi, yeni_versiyon_str, indirme_linki) doner.
    """
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code != 200:
            print(f"Update check failed: {response.status_code}")
            return False, None, None
            
        data = response.json()
        
        # Release isminden versiyonu ayikla: "Latest Build (v1.0.0.42)" -> "1.0.0.42"
        release_name = data.get("name", "")
        # Regex: v?(\d+(\.\d+)+) -> v ile baslayabilir, rakam.rakam...
        match = re.search(r'v?(\d+(\.\d+)+)', release_name)
        
        if not match:
            print(f"Could not parse version from: {release_name}")
            return False, None, None
            
        latest_version_str = match.group(1)
        
        # Versiyon karsilastir
        # packaging.version kullanmak en sagliklisi ama dependency eklemek istemezsek
        # basit split mantigi da calisir. packaging zaten standart degilse...
        # requirements'a packaging eklemedik. Basit string karsilastirmasi yapalim
        # ya da split integer.
        
        if _is_newer(latest_version_str, current_version_str):
            # Asset linkini bul (Platforma gore)
            download_url = data.get("html_url") # Fallback to release page
            
            # Daha spesifik asset linki bulmaya calis
            assets = data.get("assets", [])
            target_ext = ".exe" if sys.platform == "win32" else ".dmg"
            
            for asset in assets:
                if asset["name"].endswith(target_ext):
                    download_url = asset["browser_download_url"]
                    break
            
            return True, latest_version_str, download_url
            
        return False, latest_version_str, None
        
    except Exception as e:
        print(f"Update check error: {e}")
        return False, None, None

def _is_newer(latest, current):
    """
    latest > current ise True doner.
    Ornek: 1.0.0.42 > 1.0.0.41 -> True
    """
    try:
        l_parts = [int(x) for x in latest.split('.')]
        c_parts = [int(x) for x in current.split('.')]
        
        # Uzunluklari esitle (dolgu sifir)
        while len(l_parts) < len(c_parts): l_parts.append(0)
        while len(c_parts) < len(l_parts): c_parts.append(0)
        
        return l_parts > c_parts
    except:
        return False

def open_url(url):
    webbrowser.open(url)
