import json
import logging
import os
from collections import deque
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import tldextract
import openai
from time import sleep


openai.api_key = "YOUR_OPENAI_API_KEY"  # FROM .ENV!

MIN_SCORE = 12

HIGH_SCORE = 15
MEDIUM_SCORE = 5
LOW_SCORE = 2

HIGH_AI_WORDS = {
    "ai", "ml", "intelligence", "deeplearning",
    "artificialintelligence",
    "deeplearning", "neuralnetworks",
    "machinelearning",
    "naturallanguageprocessing", "nlp",
    "largelanguagemodel", "llm",

    # 🔷 OpenAI
    "chatgpt",
    "gpt4", "gpt.4",
    "gpt3.5", "gpt.3.5",
    "openai", "dalle", "dall.e",
    "sora", "whisper",

    # 🔷 Anthropic
    "claude", "claude2", "claude.2",
    "claude3", "claude.3",
    "anthropic",

    # 🔷 Google / DeepMind
    "gemini", "gemini1.5", "gemini.1.5",
    "bard", "palm", "palm2", "palm.2",
    "deepmind", "gemma",

    # 🔷 Meta
    "llama", "llama2", "llama.2",
    "llama3", "llama.3",

    # 🔷 Mistral
    "mistral", "mixtral",

    # 🔷 Cohere
    "commandr", "commandr+",

    # 🔷 xAI
    "grok",

    # 🔷 China AI (Baidu, Alibaba, etc.)
    "ernie", "wenxin", "qwen", "tongyi",
    "pangu", "pan.gu",
    "yayi", "hua", "skywork",

    "deepseek", "deep.seek",
    "deepseekvl", "deepseek.vl",
    "deepseekcoder", "deepseek.coder",
    "deepseekmoe", "deepseek.moe"

    # 🔷 Microsoft
                   "copilot", "phi", "phi2", "phi.2",

    # 🔷 Open Source / Community
    "vicuna", "alpaca", "guanaco",
    "falcon", "stablelm", "replit",
    "notus", "orca", "zephyr", "mpt", "codellama", "code.llama"
}
MEDIUM_AI_WORDS = {
    "data", "algorithm", "crawler", "generation", "automated", "automation", "analysis",
    "sentiment", "processing", "artificial", "predictive"
}
LOW_AI_WORDS = {
    "chatbot", "bot", "machine", "assistant", "prompt"
}

AI_DICT_DETECTION = {word.lower(): HIGH_SCORE for word in HIGH_AI_WORDS}
AI_DICT_DETECTION.update({word.lower(): MEDIUM_SCORE for word in MEDIUM_AI_WORDS})
AI_DICT_DETECTION.update({word.lower(): LOW_SCORE for word in LOW_AI_WORDS})

HIGH_SAAS_WORDS = {
    "dashboard", "subscription", "cloud-based", "platform", "software as a service", "web app", "multi-tenant", "saas",
    "login", "sign up", "pricing", "api", "register", "account", "demo", "online software", "free trial",
}
MEDIUM_SAAS_WORDS = {
     "users", "plans", "admin panel",
}
LOW_SAAS_WORDS = {
     "get started", "access anywhere", "no downloads", "scale",
}

SAAS_DICT_DETECTION = {word.lower(): HIGH_SCORE for word in HIGH_SAAS_WORDS}
SAAS_DICT_DETECTION.update({word.lower(): MEDIUM_SCORE for word in MEDIUM_SAAS_WORDS})
SAAS_DICT_DETECTION.update({word.lower(): LOW_SCORE for word in LOW_SAAS_WORDS})

EXTERNAL_SOURCES = [
    "https://www.linkedin.com/search/results/companies/?keywords={query}",
    "https://en.wikipedia.org/wiki/{query}",
    "https://twitter.com/search?q={query}&src=typed_query"
]


def clean_hostname_for_query(hostname: str) -> str:
    """
    Extracts a clean query string from a hostname.
    Removes subdomains like 'www' or 'accounts' and TLDs like '.com'.

    Example: 'www.openai.com' -> 'openai'
    """
    ext = tldextract.extract(hostname)
    return ext.domain  # e.g., 'openai' from 'www.openai.com'


def get_best_external_source_text(hostname: str) -> str:
    """
    Searches external sources (LinkedIn, Wikipedia, Twitter) using Playwright
    and extracts visible text that can help determine if the product uses AI.
    """
    query = clean_hostname_for_query(hostname)
    for template in EXTERNAL_SOURCES:
         url = template.format(query=query)
         web_text, _ = scrape_page(url)
    # todo need to complete logic of determine best sourc with ai data, best will be linked in later wiki... todo
    # todo check if working
    return ""

def find_about_page_url(soup: BeautifulSoup, base_url: str) -> str | None:
    for a in soup.find_all("a", href=True):
        href = a['href'].lower()
        if any(keyword in href for keyword in ["about", "company", "who-we-are", "about-us", "about us"]):
            full_url = urljoin(base_url, a['href'])
            return full_url
    return None



def get_favicon_url(url: str) -> str:
    """
    Given a URL, return the full favicon URL from the site.
    Supports JavaScript-rendered pages (uses Playwright).
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(2000)  # Let JS settle

            # Get favicon via JS DOM access
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
            browser.close()
            return urljoin(url, favicon_href)

    except Exception as e:
        print(f"Error extracting favicon from {url}: {e}")
        return ""



def get_page_info_from_gpt(web_text: str) -> tuple[bool, str, str, str]:
    """
    Use GPT to analyze scraped text and extract:
    - system name (product)
    - vendor name (company)
    - short description
    Also confirm if it's actually a SaaS product (vs. blog/docs).
    """
    prompt = f"""
You are an AI assistant helping to classify if website give ai usage to consumer. Based on the following web page text, answer the following questions clearly and briefly.

TEXT:
--
{web_text} 
--

Questions:
1. Is this page describing a real Ai SaaS product?
2. What is the name of the product or system?
3. Who is the vendor or company behind it?
4. Write a concise description of what the system does.

Respond in the following JSON format:
{{
  "is_ai": bool,
  "system_name": "string",
  "vendor_name": "string",
  "description": "string"
}}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )

        content = response["choices"][0]["message"]["content"]

        import json
        result = json.loads(content)

        return (
            result.get("is_ai", "").strip(),
            result.get("system_name", "").strip(),
            result.get("vendor_name", "").strip(),
            result.get("description", "").strip()
        )

    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return False, "", "", ""


def get_ai_score(web_text) -> tuple[bool, int]:
    """
    Check if the text likely describes an AI-related product.

    Returns True and the score if AI-related keywords exceed MIN_SCORE.

    Args:
        web_text (str): Text content from a web page.

    Returns:
        tuple[bool, int]: (is_ai, score)
    """
    score = 0
    web_text = web_text.lower().replace("-", "").replace("\n", " ").split()

    for word in web_text:
        if word in AI_DICT_DETECTION:
            score += AI_DICT_DETECTION[word]
        if score > MIN_SCORE:
            return True, score
    return False, 0


def is_contain_saas_words(web_text) -> bool:
    score = 0
    text = web_text.lower().replace("-", "").replace("\n", " ").split()
    for word in SAAS_DICT_DETECTION:
        if word in text:
            score += SAAS_DICT_DETECTION[word]
        if score > MIN_SCORE:
            print(f'saas ford has been found in main page: {word}')
            return True
    return False


def is_web_saas(web_text) -> bool:
    """
    Determine if a website is a SaaS product based on keyword scoring.
    Can be extended later with:
    1. External DB checks (Clearbit, Crunchbase)
    2. GPT-based analysis
    """
    return is_contain_saas_words(web_text)


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

        logging.info(f"Crawling: {current_url}")
        web_text, links = scrape_page(current_url)

        visited.add(current_url)

        to_visit += [link for link in links if link not in visited and is_same_domain(link, root_url)]

        if not web_text:
            continue

        is_ai_and_score = get_ai_score(web_text)
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
    url = f'https://{tldextract.extract(hostname).registered_domain}/'
    url = url.replace('.net', '.com')
    # Add more rules here if needed
    return url


def scrape_page(hostname: str) -> tuple[str, list[str]]:
    """
    Scrape the given URL using a headless browser.

    Returns:
        - The full visible text content of the page.
        - A list of all absolute HTTP/HTTPS links found in <a> tags.
    """
    logging.info(f"Starting scrape: {hostname}")
    if not hostname.startswith('http'): # in case valid url entered (case scrape subpages or external sorces)
        url = fix_url(hostname)
        logging.debug(f"Fixed URL: {url}")
    else:
        url = hostname

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(url, wait_until="networkidle")
            logging.info(f"Page loaded: {url}")

            # Get all hrefs from <a> tags
            hrefs = page.eval_on_selector_all("a", "els => els.map(el => el.href)")
            links = [href for href in hrefs if href and href.startswith("http") and hostname in href]
            logging.info(f"Found {len(links)} links")

            # Get all visible text content
            web_text = page.evaluate("() => document.body.innerText")

            browser.close()
            return web_text, links

    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
        return "", []


def get_hostname_dict_data(hostname: str) -> dict:
    """
    """
    homepage_url = fix_url(hostname)

    result = {
        "hostname": hostname,
        "vendor_website_url": homepage_url, # fixed homepage url
        "is_ai": False,
        "description": "",
        "system_name": "", # Product name
        "vendor_name": "",  # Company name
        "favicon_url": "",
    }

    web_text, _ = scrape_page(homepage_url)

    is_saas = is_web_saas(web_text)
    has_ai_probability = get_ai_score(web_text)[0]

    if not has_ai_probability:
        sub_pages_data = get_all_sub_pages_data(homepage_url, 10)
        if sub_pages_data:
            has_ai_probability = True
            #  search for about page, else for highest ai score
            about_page = next((p for p in sub_pages_data if "about" in p.get("url", "").lower()), None)
            if about_page:
                web_text = about_page.get("web_text", "")
            else:
                sub_pages_data.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
                web_text = sub_pages_data[0].get('web_text')  # gets the web with the highest ai score and will ask for him in prompt

    if has_ai_probability:
        is_ai, system_name, vendor_name, description = get_page_info_from_gpt(web_text)
        result["is_ai"] = is_ai
        result['favicon_url'] = get_favicon_url(homepage_url)
        result['system_name'] = system_name
        result['vendor_name'] = vendor_name
        result['description'] = description
        return result

    if is_saas:
        best_external_web_text = get_best_external_source_text(hostname)
        if best_external_web_text:
            system_name, vendor_name, description, is_ai = get_page_info_from_gpt(best_external_web_text)
            result["is_ai"] = is_ai
            result['favicon_url'] = get_favicon_url(homepage_url)
            result["system_name"] = system_name
            result["vendor_name"] = vendor_name
            result["description"] = description

    return result


def main():
    input_file = "corona_pending_hostnames.json"
    output_file = "enriched_data.xlsx"

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        hostnames = json.load(f)

    hostnames = [url for url in hostnames if url][:30]

    results = []
    for i, hostname in enumerate(hostnames):
        print(f"{i + 1}. Processing: {hostname}")
        result = get_hostname_dict_data(hostname)
        results.append(result)

    # Save to Excel (creates if not exists, overwrites if exists)
    enriched_df = pd.DataFrame(results)
    enriched_df.to_excel(output_file, index=False)
    print(f"✅ Saved: {output_file}")


if __name__ == "__main__":
    main()  # can add async version if test lot of hostnames
