import json
import logging
import os
from collections import deque
import pandas as pd
import requests
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from consts import AI_DICT_DETECTION, MIN_SCORE, SAAS_DICT_DETECTION
import re


def extract_json_block(text: str) -> str:
    """
    Extract JSON from a string wrapped in a Markdown code block.
    """
    # Remove triple backticks and any optional "json" language marker
    return re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)


def get_page_info_from_perplexity(prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 400
    }

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        cleaned = extract_json_block(content)
        result = json.loads(cleaned)
        return result

    except Exception as e:
        print(f"Error getting Perplexity response: {e}")
        return {
            "is_ai": False,
            "vendor_name": "",
            "description": "",
            "vendor_website_url": "",
            "system_name": "",
            "favicon_url": ""
        }


def is_contain_ai_words(web_text: str, web_url: str) -> tuple[bool, int]:
    """
    Check if the text likely describes an AI-related product.

    Returns True and the score if AI-related keywords exceed MIN_SCORE.

    Args:
        web_url (str)
        web_text (str): Text content from a web page.

    Returns:
        tuple[bool, int]: (is_ai, score)
    """
    score = 0
    web_text = web_text.lower().replace("-", "").replace("\n", " ").split()

    for ai_word in AI_DICT_DETECTION:
        if ai_word in web_url:
            score += AI_DICT_DETECTION[ai_word]
        if score > MIN_SCORE:
            return True, score

    for word in web_text:
        if word in AI_DICT_DETECTION:
            score += AI_DICT_DETECTION[word]
        if score > MIN_SCORE:
            return True, score
    return False, 0


def is_contain_saas_words(web_text: str, url: str) -> bool:
    score = 0
    if '/signin' in url:  # google auth
        return True
    text = web_text.lower().replace("-", "").replace("\n", " ")
    for word in SAAS_DICT_DETECTION:
        if word in text:
            score += SAAS_DICT_DETECTION[word]
        if score > MIN_SCORE:
            # print(f'saas word has been found in main page: {word}')
            return True
    return False


def is_same_domain(url: str, base: str) -> bool:
    """Check if `url` belongs to the same hostname/domain as `base`."""
    return urlparse(url).netloc == urlparse(base).netloc


def get_all_sub_pages_data(root_url: str, max_pages: int = 30) -> list[dict]:
    """
    Crawl subpages from a root URL (using BFS), collect text, and detect AI-related content.

    Args:
        root_url (str): The full starting URL (e.g., "https://example.com").
        max_pages (int): Maximum number of pages to visit during the crawl.

    Returns:
        List[dict]: A list of dictionaries for each page containing:
            - "url" (str): The page URL.
            - "web_text" (str): The visible text content of the page.
            - "ai_score" (float): The AI content score if detected.
    """
    visited = set()
    to_visit = deque([root_url])
    results = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.popleft()

        if current_url in visited:  # should never get in but for extra safety...
            continue

        web_text, _, _, links, _ = scrape_page(current_url)

        visited.add(current_url)

        to_visit += [link for link in links if link not in visited and is_same_domain(link, root_url)]

        if not web_text:
            continue

        is_ai_and_score = is_contain_ai_words(web_text, current_url)
        if is_ai_and_score[0]:
            results.append(
                {
                    "web_text": web_text,
                    "url": current_url,
                    "ai_score": is_ai_and_score[1]
                }
            )

    return results


def fix_url(hostname: str) -> str:
    """
    Apply any necessary URL corrections before scraping.
    Extend this function as needed.
    """
    url = f'https://{hostname}/'
    url = url.replace('.net', '.com')
    # Add more rules here if needed
    return url


def scrape_page(url: str) -> tuple[str, str, str, list[str], bool]:
    """
    Scrape the given URL using a headless browser.

    Returns:
        - The full visible text content of the page.
        - A list of all absolute HTTP/HTTPS links found in <a> tags.
    """

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            response = page.goto(url, wait_until="domcontentloaded")
            if not response or response.status >= 400:
                raise Exception(f"HTTP error {response.status if response else 'unknown'}")

            final_url = page.url
            logging.info(f"Final URL after redirect: {final_url}")
            favicon_href = page.evaluate("""
                            () => {
                                const rels = ["icon", "shortcut icon", "apple-touch-icon"];
                                for (const rel of rels) {
                                    const link = document.querySelector(`link[rel='${rel}']`);
                                    if (link && link.href) return link.href;
                                }
                                return "/favicon.ico";  // fallback
                            }
                        """)

            # Get all hrefs from <a> tags
            hrefs = page.eval_on_selector_all("a", "els => els.map(el => el.href)")
            domain = urlparse(final_url).netloc
            links = [href for href in hrefs if href and href.startswith("http") and domain in href]
            # print(f"Found {len(links)} links")

            # Get all visible text content
            web_text = page.evaluate("() => document.body.innerText")

            browser.close()
            return web_text, urljoin(url, favicon_href), final_url, links, True

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return "", "", "", [], False


def get_hostname_dict_data(hostname: str) -> dict:
    homepage_url = fix_url(hostname)

    result = {
        "hostname": hostname,
        "hostname_url": homepage_url,
        "is_valid_url": False,
        "vendor_website_url": "",
        "is_ai": False,
        "description": "",
        "system_name": "",
        "vendor_name": "",  # Company name
        "favicon_url": "",
        "has_ai_probability": False,
        "has_saas_probability": False,
        "rating": 5
    }

    web_text, favicon_url, redirect_url, _, is_valid = scrape_page(homepage_url)
    result['is_valid_url'] = is_valid

    if not is_valid:
        prompt = f"""
        You are an AI classifier. The given website could not be scraped (possibly 404 or restricted).
        Based only on the domain name, infer the following:

        1. Is this an AI SaaS product?
        2. Who is the vendor?
        3. What is the system name? (2-3 words)
        4. What does the system do? Write a short description (max 160 characters).
        5. What is the likely homepage URL of the product or company?
        6. What is the likely favicon URL?

        DOMAIN: {hostname}

        Respond in JSON:
        {{
          "is_ai": bool,
          "vendor_name": "str",
          "system_name": "str",
          "description": "str",
          "vendor_website_url": "str",
          "favicon_url": "str"
        }}
        """
        info = get_page_info_from_perplexity(prompt)

        result["vendor_website_url"] = info["vendor_website_url"]
        result["is_ai"] = info.get("is_ai", False)
        result["vendor_name"] = info.get("vendor_name", "")
        result["system_name"] = info.get("system_name")
        result["description"] = info.get("description")
        result["favicon_url"] = info.get("favicon_url")

        return result

    has_saas_probability = is_contain_saas_words(web_text, redirect_url)
    has_ai_probability = is_contain_ai_words(web_text, homepage_url)[0]

    if not has_ai_probability and get_all_sub_pages_data(homepage_url, 10):  # has sub-pages with ai probability
        has_ai_probability = True

    result['favicon_url'] = favicon_url
    result['vendor_website_url'] = redirect_url
    result['has_ai_probability'] = has_ai_probability
    result['has_saas_probability'] = has_saas_probability

    if has_ai_probability or has_saas_probability:
        prompt = f"""
        You are an AI classifier. Check the web page at the URL below and answer:

        1. Is this an AI SaaS product?
        2. Who is the vendor?
        3. What is the system name? (2-3 words)
        4. What does the system do? Write a short description (~160 characters).

        WEB URL: {homepage_url}

        Respond in JSON:
        {{
          "is_ai": bool,
          "vendor_name": "str",
          "system_name": "str",
          "description": "str"
        }}
        """
        info = get_page_info_from_perplexity(prompt)

        result["is_ai"] = info.get("is_ai", False)
        result["vendor_name"] = info.get("vendor_name", "")
        result["system_name"] = info.get("system_name", "")
        result["description"] = info.get("description", "")

    return result


def main():
    load_dotenv()
    input_file = "test.json"
    output_file = "enriched_data_final.xlsx"

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        hostnames = json.load(f)

    hostnames = [url for url in hostnames if url]

    results = []
    for i, hostname in enumerate(hostnames):
        # print(f"{i + 1}. Processing: {hostname}")
        result = get_hostname_dict_data(hostname)
        results.append(result)

    # Save to Excel (creates if not exists, overwrites if exists)
    enriched_df = pd.DataFrame(results)
    enriched_df.to_excel(output_file, index=False)
    print(f"✅ Saved: {output_file}")


if __name__ == "__main__":
    main()
