import requests
import time
import pandas as pd

API_URL = "https://az.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "MuradNLPBot/1.0 (murad.valiyev@student.edu)"
}


def get_all_titles(limit=300):
    titles = []
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "aplimit": "max",
        "format": "json"
    }

    while True:
        response = requests.get(API_URL, params=params, headers=HEADERS)

        if response.status_code != 200:
            print("HTTP error:", response.status_code)
            break

        try:
            data = response.json()
        except ValueError:
            print("Non-JSON response")
            break

        pages = data["query"]["allpages"]
        titles.extend([p["title"] for p in pages])

        if "continue" not in data:
            break

        params.update(data["continue"])

        if limit and len(titles) >= limit:
            return titles[:limit]

    return titles


def get_random_titles(n=3000):
    titles = []
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 10,
        "format": "json"
    }

    while len(titles) < n:
        r = requests.get(API_URL, params=params, headers=HEADERS).json()
        for p in r["query"]["random"]:
            print("Extracting title:", p["title"])
            titles.append(p["title"])



    return titles[:n]



def get_page_data(title):
    params = {
        "action": "query",
        "prop": "extracts|revisions",
        "rvprop": "timestamp|user",
        "explaintext": True,
        "titles": title,
        "format": "json"
    }

    response = requests.get(API_URL, params=params, headers=HEADERS)

    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except ValueError:
        return None

    page = next(iter(data["query"]["pages"].values()))

    text = page.get("extract", "")
    revisions = page.get("revisions", [])

    last_edit = None
    last_editor = None

    if revisions:
        last_edit = revisions[0].get("timestamp")
        last_editor = revisions[0].get("user")

    return {
        "title": title,
        "text": text,
        "last_edit": last_edit,
        "last_editor": last_editor
    }


def main():
    titles = get_random_titles(3000)
    print(f"Collected {len(titles)} titles")

    corpus = []

    for i, title in enumerate(titles, start=1):
        print(f"[{i}/{len(titles)}] {title}")

        result = get_page_data(title)
        if not result:
            continue

        if result["text"].strip():
            corpus.append(result)


    df = pd.DataFrame(corpus)
    df.to_csv("az_wikipedia_plaintext_with3.csv", index=False)

    print("Saved az_wikipedia_plaintext_with3.csv")
    print(df.head())


if __name__ == "__main__":
    main()
