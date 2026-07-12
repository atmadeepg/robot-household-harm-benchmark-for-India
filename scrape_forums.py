import argparse
import json
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
THREAD_URL_PATTERN = re.compile(r"threads/[^/]+\.\d+/?$")
# confirmed: real element is <blockquote class="messageText ..."> — div.message-body .bbWrapper
# is the XenForo 2.x default but doesn't match Indusladies' theme
POST_SELECTOR = ".messageText"
NEXT_PAGE_SELECTOR = "a.pageNav-jump--next"
REQUEST_DELAY_SECONDS = 1.5


def get_resolution_base(soup, fallback_url):
    # XenForo sets an HTML <base> tag that overrides relative-link resolution;
    # ignoring it caused wrong thread URLs when the listing was paginated
    base_tag = soup.find("base", href=True)
    if base_tag:
        return base_tag["href"]
    return fallback_url


def fetch(url, verbose=False):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if verbose:
        print(f"    [fetch] {url} -> status {resp.status_code}, {len(resp.text)} chars")
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(resp.text, "html.parser")


def find_thread_urls(listing_url, max_listing_pages=5):
    thread_urls = set()
    url = listing_url

    for page_num in range(max_listing_pages):
        soup = fetch(url, verbose=True)
        base = get_resolution_base(soup, url)
        page_links_found = 0
        for a in soup.find_all("a", href=True):
            if THREAD_URL_PATTERN.search(a["href"]):
                thread_urls.add(urljoin(base, a["href"]))
                page_links_found += 1
        print(f"    page {page_num + 1}: {page_links_found} thread links found on this page")

        next_link = soup.select_one(NEXT_PAGE_SELECTOR)
        if not next_link or not next_link.get("href"):
            break
        url = urljoin(base, next_link["href"])

    return list(thread_urls)


def scrape_thread(thread_url, max_thread_pages=10):
    posts = []
    url = thread_url

    for _ in range(max_thread_pages):
        soup = fetch(url)
        base = get_resolution_base(soup, url)
        for el in soup.select(POST_SELECTOR):
            text = el.get_text(separator=" ", strip=True)
            if len(text) >= 40:
                posts.append(text)

        next_link = soup.select_one(NEXT_PAGE_SELECTOR)
        if not next_link or not next_link.get("href"):
            break
        url = urljoin(base, next_link["href"])

    return posts


def scrape_forums(listing_urls, max_threads_per_listing=100):
    # "subreddit" field kept for compatibility with cluster_personas.py which
    # uses it as a generic source label — these aren't subreddits
    documents = []

    for listing_url in listing_urls:
        print(f"Finding threads in {listing_url} ...")
        thread_urls = find_thread_urls(listing_url)[:max_threads_per_listing]
        print(f"  found {len(thread_urls)} threads")

        for i, thread_url in enumerate(thread_urls):
            try:
                posts = scrape_thread(thread_url)
                for post_text in posts:
                    documents.append({"text": post_text, "subreddit": listing_url})
                if (i + 1) % 10 == 0:
                    print(f"  scraped {i + 1}/{len(thread_urls)} threads, "
                          f"{len(documents)} posts so far")
            except requests.RequestException as e:
                print(f"  WARNING: failed to scrape {thread_url}: {e}")
                continue

    return documents


def main():
    parser = argparse.ArgumentParser(description="Scrape XenForo-based forums (e.g. Indusladies)")
    parser.add_argument("--listing-urls", nargs="+", required=True,
                        help="Subforum listing page URLs")
    parser.add_argument("--max-threads-per-listing", type=int, default=100)
    parser.add_argument("--out", default="forum_posts.json")
    args = parser.parse_args()

    documents = scrape_forums(args.listing_urls, args.max_threads_per_listing)
    print(f"\nTotal posts scraped: {len(documents)}")

    if len(documents) == 0:
        print("WARNING: zero posts — POST_SELECTOR likely doesn't match this site's "
              "theme; inspect a thread page and check the real class names.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
