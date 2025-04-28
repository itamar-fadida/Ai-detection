import json
import os
import requests
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from dotenv import load_dotenv
import re


TAGS_BANK = [
    "Generative AI", "Text Generation", "Image Generation", "Video Generation",
    "Chatbot", "Conversational AI", "Creative Tools", "Content Creation",
    "Advertising", "Marketing", "Sales", "CRM", "Automation", "Workflow Automation",
    "No-code", "Low-code", "Web Development", "Website Builder", "Design Tools",
    "Analytics", "Data Intelligence", "Open Source", "NLP", "Transformers",
    "Productivity", "Assistant", "Enterprise AI"
]

VERTICALS_BANK = [
    "Marketing", "Sales", "Finance", "Healthcare", "Education",
    "Developer Tools", "Automation", "Design & Creative", "Analytics",
    "Web & CMS", "General", "Legal", "HR & Recruiting", "Productivity"
]

RISK_BANK = ['Low', 'Medium', 'High', 'Critical']


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
        "max_tokens": 900
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
            "favicon_url": "",
            "is_pure_ai_saas": False,
            "tags": [],
            "verticals": [],
            "risk": "Low"
        }


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
        return "", "", False


async def get_hostname_dict_data(hostname: str) -> dict:
    load_dotenv()
    homepage_url = f'https://{hostname.strip()}'


result = {
        "hostname": hostname,
        "hostname_url": homepage_url,
        "is_valid_url": False,
        "vendor_website_url": "",
        "is_ai": False,
        "is_pure_ai_saas": False,
        "description": "",
        "system_name": "",
        "vendor_name": "",
        "favicon_url": "",
        "tags": [],
        "verticals": [],
        "risk": "Low",
        "rating": 5
    }

    favicon_url, redirect_url, is_valid = await scrape_page(homepage_url)
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
        8. Select the 2–5 most relevant Tags that best describe the system. Choose ONLY from this list: {TAGS_BANK}.
        9. Select the 1–5 most appropriate Verticals that represent the primary industries or sectors the system serves. Choose ONLY from this list: {VERTICALS_BANK}.
        10. Assign a Risk Level from {RISK_BANK}, based on the following strict evaluation:
        - AI Model Type: If the system is a large language model (LLM), generative AI, or autonomous agent, the minimum risk level is "High".
        - Data Sensitivity: If the system handles personal, financial, healthcare, or confidential data without strong compliance signals, assign "High" or "Critical" risk.
        - Autonomy: Systems that can make decisions without human review should be rated "High" or "Critical" risk.
        - Vendor Trustworthiness: If the vendor is unknown, unverified, or based outside the U.S. or Europe (e.g., China, Russia), assign "Critical" risk unless strong compliance evidence (e.g., GDPR, SOC2) is shown.


        
        HARD RULES:
        - NEVER return empty Tags or Verticals: choose at least 2 Tags and 1 Vertical.
        - If your `description` includes terms like "AI", "LLM", "machine learning", "generative", or any clear reference to artificial intelligence:
          Then you MUST set `"is_ai": true`.
        - ONLY choose Tags and Verticals from the provided lists.
        
        DOMAIN: {hostname}

        Respond ONLY with a valid JSON object inside a code block like this:
        {{
          "is_ai": bool,
          "vendor_name": "str",
          "system_name": "str",
          "description": "str",
          "vendor_website_url": "str",
          "favicon_url": "str",
          "is_pure_ai_saas": bool,
          "tags": [ "str", ... ],
          "verticals": [ "str", ... ],
          "risk": "str"
        }}
        """
        info = await get_page_info_from_perplexity(prompt)

        result["vendor_website_url"] = info.get("vendor_website_url")
        result["is_ai"] = info.get("is_ai", False)
        result["vendor_name"] = info.get("vendor_name", "")
        result["system_name"] = info.get("system_name")
        result["description"] = info.get("description")
        result["favicon_url"] = info.get("favicon_url")
        result["is_pure_ai_saas"] = info.get("is_pure_ai_saas", False)
        result["tags"] = info.get("tags", [])
        result["verticals"] = info.get("verticals", [])
        result["risk"] = info.get("risk", "Low")

        print(json.dumps(result, indent=2))
        return result

    # for case could scrape the web and got the favicon url
    result['favicon_url'] = favicon_url
    result['vendor_website_url'] = redirect_url

    prompt = f"""
    You are an AI classifier. Check the web page at the URL below and answer:

    1. Is this an AI SaaS product?
    2. Who is the vendor? (1-3 words)
    3. What is the system name? (1-3 words)
    4. What does the system do? Write a short description (~160 characters).
    5. Is this a *pure* AI SaaS product where AI is the core offering?
    6. Select the 2–5 most relevant Tags that best describe the system. Choose ONLY from this list: {TAGS_BANK}.
    7. Select the 1–5 most appropriate Verticals that represent the primary industries or sectors the system serves. Choose ONLY from this list: {VERTICALS_BANK}.
    8. . Assign a Risk Level from {RISK_BANK}, based on the following strict evaluation:
        - AI Model Type: If the system is a large language model (LLM), generative AI, or autonomous agent, the minimum risk level is "High".
        - Data Sensitivity: If the system handles personal, financial, healthcare, or confidential data without strong compliance signals, assign "High" or "Critical" risk.
        - Autonomy: Systems that can make decisions without human review should be rated "High" or "Critical" risk.
        - Vendor Trustworthiness: If the vendor is unknown, unverified, or based outside the U.S. or Europe (e.g., China, Russia), assign "Critical" risk unless strong compliance evidence (e.g., GDPR, SOC2) is shown.


        
    HARD RULES:
    - NEVER return empty Tags or Verticals: choose at least 2 Tags and 1 Vertical.
    - If your `description` includes terms like "AI", "LLM", "machine learning", "generative", or any clear reference to artificial intelligence:
      Then you MUST set `"is_ai": true`.
    - ONLY choose Tags and Verticals from the provided lists.
    
    WEB URL: {homepage_url}

    Respond ONLY with a valid JSON object inside a code block like this:
    {{
      "is_ai": bool,
      "vendor_name": "str",
      "system_name": "str",
      "description": "str",
      "is_pure_ai_saas": bool,
      "tags": [ "str", ... ],
      "verticals": [ "str", ... ],
      "risk": "str"
    }}
    """
    info = get_page_info_from_perplexity(prompt)

    result["is_ai"] = info.get("is_ai", False)
    result["vendor_name"] = info.get("vendor_name", "")
    result["system_name"] = info.get("system_name", "")
    result["description"] = info.get("description", "")
    result["is_pure_ai_saas"] = info.get("is_pure_ai_saas", False)
    result["tags"] = info.get("tags", [])
    result["verticals"] = info.get("verticals", [])
    result["risk"] = info.get("risk", "Low")

    print(json.dumps(result, indent=2))
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
