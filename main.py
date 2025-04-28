import json
import os
from collections import deque
import pandas as pd
from playwright.async_api import async_playwright
import aiohttp
import asyncio
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from consts import AI_DICT_DETECTION, MIN_SCORE, SAAS_DICT_DETECTION
import re


def extract_json_block(text: str) -> str:
    match = re.search(r'\{[\s\S]+?\}', text)
    if match:
        return match.group()
    raise ValueError("No JSON object found")


async def get_page_info_from_perplexity(prompt: str) -> dict:
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
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                cleaned = extract_json_block(content)
                return json.loads(cleaned)

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
            # print(f'SaaS word has been found in main page: {word}')
            return True
    return False


def is_same_domain(url: str, base: str) -> bool:
    """Check if `url` belongs to the same hostname/domain as `base`."""
    return urlparse(url).netloc == urlparse(base).netloc


async def get_all_sub_pages_data(root_url: str, max_pages: int = 30) -> list[dict]:
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

        web_text, _, _, links, _ = await scrape_page(current_url)

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


async def scrape_page(url: str) -> tuple[str, str, str, list[str], bool]:
    try:
        async with async_playwright() as p:
            # don't remove the headless tag - this opens the web in window and responsible for the redirect
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()

            response = await page.goto(url, wait_until="domcontentloaded")
            if not response or response.status >= 400:
                raise Exception(f"HTTP error {response.status if response else 'unknown'}")

            final_url = page.url
            favicon_href = await page.evaluate("""
                () => {
                    const rels = ["icon", "shortcut icon", "apple-touch-icon"];
                    for (const rel of rels) {
                        const link = document.querySelector(`link[rel='${rel}']`);
                        if (link && link.href) return link.href;
                    }
                    return "/favicon.ico";
                }
            """)

            hrefs = await page.eval_on_selector_all("a", "els => els.map(el => el.href)")
            domain = urlparse(final_url).netloc
            links = [href for href in hrefs if href and href.startswith("http") and domain in href]
            web_text = await page.evaluate("() => document.body.innerText")

            await browser.close()
            return web_text, urljoin(url, favicon_href), final_url, links, True

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return "", "", "", [], False


async def get_hostname_dict_data(hostname: str) -> dict:
    homepage_url = f'https://{hostname}/'

    result = {
        "hostname": hostname,
        "hostname_url": homepage_url,
        "is_valid_url": False,
        "vendor_website_url": "",
        "is_ai": False,
        "is_pure_ai_saas": False,
        "description": "",
        "system_name": "",
        "vendor_name": "",  # Company name
        "favicon_url": "",
        "has_ai_probability": False,
        "has_saas_probability": False,
        "rating": 5
    }

    web_text, favicon_url, redirect_url, _, is_valid = await scrape_page(homepage_url)
    result['is_valid_url'] = is_valid

    if not is_valid:
        prompt = f"""
        You are an AI classifier. The given website could not be scraped (possibly 404 or restricted).
        Based only on the domain name, infer the following:

        1. Is this an AI SaaS product? 
        2. Who is the vendor? (1-3 words)
        3. What is the system name? (1-3 words)
        4. What does the system do? Write a short description (max 160 characters).
        5. What is the likely homepage URL of the product or company?
        6. What is the likely favicon URL?
        7. Is this a *pure* AI SaaS product where AI is the core offering?
        
        HARD RULE:
        If your `description` includes terms like "AI", **"LLM", "machine learning", "generative", or any clear reference to artificial intelligence:
        Then you MUST set `"is_ai": true`.
        
        DOMAIN: {hostname}

        Respond ONLY with a valid JSON object inside a code block like this:
        {{
          "is_ai": bool,
          "vendor_name": "str",
          "system_name": "str",
          "description": "str",
          "vendor_website_url": "str",
          "favicon_url": "str",
          "is_pure_ai_saas": bool
        }}
        """
        info = await get_page_info_from_perplexity(prompt)

        result["vendor_website_url"] = info["vendor_website_url"]
        result["is_ai"] = info.get("is_ai", False)
        result["vendor_name"] = info.get("vendor_name", "")
        result["system_name"] = info.get("system_name")
        result["description"] = info.get("description")
        result["favicon_url"] = info.get("favicon_url")
        result["is_pure_ai_saas"] = info.get("is_pure_ai_saas", False)

        return result

    has_saas_probability = is_contain_saas_words(web_text, redirect_url)
    has_ai_probability = is_contain_ai_words(web_text, homepage_url)[0]

    if not has_ai_probability and await get_all_sub_pages_data(homepage_url, 10):  # has sub-pages with Ai probability
        has_ai_probability = True

    result['favicon_url'] = favicon_url
    result['vendor_website_url'] = redirect_url
    result['has_ai_probability'] = has_ai_probability
    result['has_saas_probability'] = has_saas_probability

    if not with_words_detection or (has_ai_probability or has_saas_probability):
        prompt = f"""
        You are an AI classifier. Check the web page at the URL below and answer:

        1. Is this an AI SaaS product?
        2. Who is the vendor? (1-3 words)
        3. What is the system name? (1-3 words)
        4. What does the system do? Write a short description (~160 characters).
        5. Is this a *pure* AI SaaS product where AI is the core offering?

        HARD RULE:
        If your `description` includes terms like "AI", **"LLM", "machine learning", "generative", or any clear reference to artificial intelligence:
        Then you MUST set `"is_ai": true`.

        WEB URL: {homepage_url}

        Respond ONLY with a valid JSON object inside a code block like this:
        {{
          "is_ai": bool,
          "vendor_name": "str",
          "system_name": "str",
          "description": "str",
          "is_pure_ai_saas": bool
        }}
        """
        info = await get_page_info_from_perplexity(prompt)

        result["is_ai"] = info.get("is_ai", False)
        result["vendor_name"] = info.get("vendor_name", "")
        result["system_name"] = info.get("system_name", "")
        result["description"] = info.get("description", "")
        result["is_pure_ai_saas"] = info.get("is_pure_ai_saas", False)

    return result


async def main():
    load_dotenv()
    input_file = "test.json"
    output_file = "more.xlsx"

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        hostnames = json.load(f)

    hostnames = [url for url in hostnames if url]
    sem = asyncio.Semaphore(scrape_parallel_amount)  # limit to 5 concurrent tasks

    async def get_hostname_with_limit(hostname):
        async with sem:
            return await get_hostname_dict_data(hostname)

    tasks = [get_hostname_with_limit(h) for h in hostnames]
    results = await asyncio.gather(*tasks)

    # Save to Excel (creates if not exists, overwrites if exists)
    enriched_df = pd.DataFrame(results)
    enriched_df.to_excel(output_file, index=False)
    print(f"✅ Saved: {output_file}")


if __name__ == "__main__":
    with_words_detection = False
    scrape_parallel_amount = 15
    asyncio.run(main())


tags = [
  "Generative AI", "Text Generation", "Image Generation", "Video Generation",
  "Chatbot", "Conversational AI", "Creative Tools", "Content Creation",
  "Advertising", "Marketing", "Sales", "CRM", "Automation", "Workflow Automation",
  "No-code", "Low-code", "Web Development", "Website Builder", "Design Tools",
  "Analytics", "Data Intelligence", "Open Source", "NLP", "Transformers",
  "Productivity", "Assistant", "Enterprise AI"
]

verticals = [
  "Marketing", "Sales", "Finance", "Healthcare", "Education",
  "Developer Tools", "Automation", "Design & Creative", "Analytics",
  "Web & CMS", "General", "Legal", "HR & Recruiting", "Productivity"
]
