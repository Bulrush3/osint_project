import requests
import xml.etree.ElementTree as ET

YANDEX_FOLDER_ID = ''
YANDEX_API_KEY   = ''

def yandex_search_vk_groups(query, page=0):
    params = {
        "folderid": YANDEX_FOLDER_ID,
        "apikey":   YANDEX_API_KEY,
        "query":    f"{query} site:vk.com/club",
        "filter":   "none",
        "page":     page,
        "groupby":  "attr=d.mode=deep.groups-on-page=5.docs-in-group=3",
    }
    r = requests.get("https://yandex.ru/search/xml", params=params)
    # print("Ответ Яндекса:", r.text[:500]) 
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return [u.text for u in root.findall(".//doc/url")]