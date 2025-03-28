import asyncio
import json
import os
import re
import whois
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import tldextract
import requests


def format_web_to_open_ai(web_scrape_content: str) -> dict:
    """ Gets the web content and return dict with:
    1. icon_url
    2. matadata
    3. formatted content (no html tags, no css)
    4.
    """
    pass


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


def get_page_info_from_gpt(web_text, metadata) -> tuple[str]:
    """
    - System name
    - Vendor name
    - System Description
    """
    pass


def get_description(soup) -> str:
    desc_tag = (
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", attrs={"property": "og:description"})
    )
    return desc_tag["content"].strip() if desc_tag and "content" in desc_tag.attrs else ""

def extract_vendor_from_whois(domain: str) -> str | None:
    try:
        w = whois.whois(domain)
        return w.get("org") or w.get("name")
    except Exception:
        return None


def extract_vendor_from_footer(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    footers = soup.find_all("footer")

    extra_candidates = soup.find_all(string=re.compile(r"©|\(c\)|Copyright", re.I))
    all_candidates = footers + [tag.parent for tag in extra_candidates if tag.parent]

    for tag in all_candidates:
        text = tag.get_text(" ", strip=True)
        match = re.search(r"(?:©|\(c\)|Copyright)\s*\d{0,4}\s*(.*?)\s*(All rights reserved|\.|$)", text, re.I)
        if match:
            return match.group(1).strip()

    return None


def find_about_page_url(soup: BeautifulSoup, base_url: str) -> str | None:
    for a in soup.find_all("a", href=True):
        href = a['href'].lower()
        if any(keyword in href for keyword in ["about", "company", "who-we-are", "about-us", "about us"]):
            full_url = urljoin(base_url, a['href'])
            return full_url
    return None


def extract_vendor_from_about_page(about_url: str) -> str | None:
    try:
        res = requests.get(about_url, timeout=10)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")

        # Try first heading or first paragraph
        heading = soup.find(["h1", "h2", "h3"])
        if heading:
            return heading.get_text(strip=True)

        paragraph = soup.find("p")
        if paragraph:
            return paragraph.get_text(strip=True)

    except Exception:
        return None


def get_vendor_name(html: str, base_url: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")

    # 1. Try footer
    name = extract_vendor_from_footer(html)
    if name:
        return name

    # 2. Try WHOIS org name
    domain = tldextract.extract(base_url).registered_domain
    if domain:
        name = extract_vendor_from_whois(domain)
        if name:
            return name

    # 3. Try "About" page
    about_url = find_about_page_url(soup, base_url)
    if about_url:
        name = extract_vendor_from_about_page(about_url)
        if name:
            return name


    return ""

def get_favicon_url(soup, base_url):
    # Priority: SVG > PNG > ICO
    rels = ["apple-touch-icon", "icon", "shortcut icon"]

    for rel in rels:
        icon_tag = soup.find("link", rel=rel)
        if icon_tag and "href" in icon_tag.attrs:
            href = icon_tag["href"]
            if any(href.endswith(ext) for ext in [".svg", ".png", ".ico"]):
                return urljoin(base_url, href)
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/favicon.ico"


async def get_internal_subpages(soup, base_url, hostname, max_pages=5) -> list[str]:
    subpages = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Ignore external links and fragments
        if href.startswith("#") or "mailto:" in href or "tel:" in href:
            continue
        if href.startswith("/") and hostname in href:
            subpages.add(urljoin(base_url, href))
        # Limit the number of pages to check - Optional
        # if len(subpages) >= max_pages:
        #     break
    return list(subpages)

def is_contain_ai_words(web_text) -> bool:
    score = 0
    print("start ai check")
    web_text = web_text.lower().replace("-", "").replace("\n", " ").split()

    for word in web_text:
        if word in AI_DICT_DETECTION:
            score += AI_DICT_DETECTION[word]
            print(f'{score} ai points for word {word}')
        if score > MIN_SCORE:
            return True
    return False


def is_page_contain_ai(web_text: str) -> bool:
    """
    1. search by words
    2. send to open ai just web's text (short prompt)  # todo
    """
    return is_contain_ai_words(web_text)  # temp


def is_contain_saas_words(web_text) -> bool:
    score = 0
    print("start saas check")
    text = web_text.lower().replace("-", "").replace("\n", " ").split()
    for word in SAAS_DICT_DETECTION:
        if word in text:
            score += SAAS_DICT_DETECTION[word]
            print(f'{score} saas points for word {word}')
        if score > MIN_SCORE:
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


def extract_metadata(soup: BeautifulSoup) -> dict:
    metadata = {}
    # Page title
    if soup.title:
        metadata["title"] = soup.title.string.strip()
    # Meta description
    description_tag = soup.find("meta", attrs={"name": "description"})
    if description_tag and description_tag.get("content"):
        metadata["description"] = description_tag["content"].strip()
    # Open Graph tags (og:title, og:description, etc.)
    og_tags = ["og:title", "og:description", "og:site_name"]
    for tag in og_tags:
        og_tag = soup.find("meta", property=tag)
        if og_tag and og_tag.get("content"):
            metadata[tag] = og_tag["content"].strip()

    return metadata


async def scrape_url_info(hostname: str) -> dict:
    """
    1. Check if web is SaaS
    2. Find if there is AI usage on the homepage
    3. If not, check a few internal subpages
    """
    result = {
        "hostname": hostname,
        "is_SaaS": False,
        "is_ai": False,
        "vendor_website_url": "",
        "system_name": "", # Product name
        "description": "",
        "vendor_name": "",  # Company name
        "favicon_url": "",
    }

    url = f'https://{tldextract.extract(hostname).registered_domain}/'
    url = url.replace('.net', '.com')

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(url, timeout=60000)
            await page.wait_for_timeout(5000)

            text = await page.evaluate("() => document.body.innerText")
            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Check SaaS
            if not is_web_saas(text):
                result['is_SaaS'] = False
                return result
            result['is_SaaS'] = True

            # Check first for homepage. if not found - scrape all sub-pages
            is_current_page_have_ai = is_page_contain_ai(text)
            if is_current_page_have_ai:
                result['vendor_website_url'] = url
            else:
                subpage_urls = await get_internal_subpages(soup, url, hostname)
                for subpage_url in subpage_urls:
                    try:
                        subpage = await context.new_page() # open new page for not loose the first page content
                        await subpage.goto(subpage_url, timeout=30000)
                        await subpage.wait_for_timeout(2000)
                        sub_text = await subpage.evaluate("() => document.body.innerText")
                        await subpage.close()

                        if is_page_contain_ai(sub_text):
                            is_current_page_have_ai = True
                            sub_html = await subpage.content()
                            sub_text = await subpage.evaluate("() => document.body.innerText")
                            sub_soup = BeautifulSoup(sub_html, "html.parser")
                            result['vendor_website_url'] = subpage_url
                            break
                            # TODO take care if there is more than 1 ai subpage for this hostname
                    except Exception as e:
                        print(f"Failed to scrape subpage {subpage_url}: {e}")
                        return result

                if not is_current_page_have_ai:
                    return result

            result['is_ai'] = True
            # todo need to decide it to search on sub page or homepage

            # option 1: all once from openai and if needed external sources

            metadata = extract_metadata(soup)
            # system_name, vendor_name, description = get_page_info_from_gpt(text, metadata)

            # option 2: all free, no prompt - danger might not have the true details or missing some

            result['system_name'] = soup.title.string.strip() if soup.title else ""
            # result['vendor_name'] = get_vendor_name(html, url) # fixme
            result['description'] = get_description(soup) # or sub_soup?
            result['favicon_url'] = get_favicon_url(soup, url)

            return result

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return result
        finally:
            await browser.close()


async def main():
    input_file = "corona_pending_hostnames.json"
    output_file = "enriched_data.xlsx"  # Changed to .xlsx

    # Ensure input JSON exists
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    # Load hostnames
    with open(input_file, "r", encoding="utf-8") as f:
        hostnames = json.load(f)

    # Filter non-empty and limit for testing
    hostnames = [url for url in hostnames if url][:2]

    results = []
    for i, hostname in enumerate(hostnames):
        print(f"{i + 1}. Processing: {hostname}")
        result = await scrape_url_info(hostname)
        results.append(result)

    # Save to Excel (creates if not exists, overwrites if exists)
    enriched_df = pd.DataFrame(results)
    enriched_df.to_excel(output_file, index=False)
    print(f"✅ Saved: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
