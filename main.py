import asyncio
import json
from typing import Union

import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import tldextract


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


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "").split(".")[0]


def detect_saas(content: str) -> bool:
    keywords = ["sign up", "log in", "dashboard", "platform", "subscription", "cloud", "get started"]
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in keywords)


def get_favicon_url(soup, base_url):
    favicon_tag = (
            soup.find("link", rel="icon") or
            soup.find("link", rel="shortcut icon") or
            soup.find("link", rel="apple-touch-icon")
    )

    if favicon_tag and "href" in favicon_tag.attrs:
        favicon_href = favicon_tag["href"]
        return urljoin(base_url, favicon_href)
    else:
        # Fallback to default /favicon.ico
        parsed_url = urlparse(base_url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}/favicon.ico"


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
    web_text = web_text.lower().replace("-", "").replace("\n", " ").split()

    for word in web_text:
        if word in AI_DICT_DETECTION:
            score += AI_DICT_DETECTION[word]
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
    text = web_text.lower()
    for word in SAAS_DICT_DETECTION:
        if word in text:
            score += SAAS_DICT_DETECTION[word]
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


async def scrape_url_info(hostname: str) -> dict:
    """
    1. Check if web is SaaS
    2. Find if there is AI usage on the homepage
    3. If not, check a few internal subpages
    """
    result = {
        "hostname": hostname,
        "is_SaaS": False,
        "is_ai": False
    }

    url = f'https://{tldextract.extract(hostname).registered_domain}/'

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(url, timeout=60000)
            await page.wait_for_timeout(5000)

            html = await page.content()
            text = await page.evaluate("() => document.body.innerText")
            soup = BeautifulSoup(html, "html.parser")

            # Check SaaS
            if not is_web_saas(text):
                result['is_SaaS'] = False
                return result
            result['is_SaaS'] = True

            # Check first for homepage. if not found - scrape all sub-pages
            is_current_page_have_ai = is_page_contain_ai(text)
            if not is_current_page_have_ai:
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
                            break
                    except Exception as e:
                        print(f"Failed to scrape subpage {subpage_url}: {e}")

                if not is_current_page_have_ai:
                    return result

            result['is_ai'] = True



            title = soup.title.string.strip() if soup.title else ""
            system_name = title
            # todo send title text to NLP for get system name
            # Description
            desc_tag = (
                    soup.find("meta", attrs={"name": "description"}) or
                    soup.find("meta", attrs={"property": "og:description"})
            )
            description = desc_tag["content"].strip() if desc_tag and "content" in desc_tag.attrs else ""
            # todo NLP for description if empty
            # Canonical URL
            canonical_tag = soup.find("link", rel="canonical")
            canonical_url = canonical_tag["href"] if canonical_tag and "href" in canonical_tag.attrs else url

            # Favicon
            favicon_url = get_favicon_url(soup, url)

            return {
                "input_url": url,
                "system_name": system_name,
                "vendor_name": extract_domain(url),
                "is_saas": detect_saas(text),
                "system_description": description,
                "vendor_url": canonical_url,
                "favicon_url": favicon_url,
            }

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {
                "input_url": url,
                "system_name": "",
                "vendor_name": "",
                "is_saas": "",
                "system_description": "",
                "vendor_url": "",
                "favicon_url": "",
                "is_potentially_ai": False,
                "ai_points_breakdown": "",
                "ai_score": 0
            }
        finally:
            await browser.close()


async def main():
    with open("data.json", "r") as f:
        hostnames = json.load(f)
    hostnames = [url for url in hostnames if url][:20]

    results = []
    for i, hostname in enumerate(hostnames):
        print(f"{i + 1}. Processing: {hostname}")
        result = await scrape_url_info(hostname)
        results.append(result)

    enriched_df = pd.DataFrame(results)
    enriched_df.to_csv("enriched_data.csv", index=False)
    print("Saved enriched_data.csv")


if __name__ == "__main__":
    asyncio.run(main())
