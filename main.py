import os
import json
import asyncio
import re
from urllib.parse import urljoin
import aiohttp
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

TAGS = ['Conversational AI', 'Search Augmented', 'Coding Assistant', 'Office Productivity', 'Customer Support',
        'Creative Writing', 'Translation', 'Grammar & Rewriting', 'Note Taking', 'Personal Companion', 'Multimodal AI',
        'Image Generation', 'Video Generation', 'Voice Assistant', 'Education & Tutoring', 'Search Engine',
        'On-Device AI', 'Data Analysis', 'Security & Compliance', 'Agent / Automation']

VERTICALS = ['General Productivity', 'Software Development', 'Marketing & Advertising', 'Customer Service',
             'Education & Training', 'Legal', 'Healthcare & Life Sciences', 'Finance & Accounting', 'Human Resources',
             'Media & Entertainment', 'Retail & eCommerce', 'Security & Risk', 'Government & Public Sector',
             'Energy & Utilities', 'Real Estate', 'Transportation & Logistics', 'Manufacturing', 'Scientific Research',
             'Design & UX', 'Recruiting & Talent']



def extract_json_block(text: str) -> str:
    match = re.search(r'\{[\s\S]+?\}', text)
    if match:
        return match.group()
    raise ValueError("No JSON object found")


async def get_page_info_from_perplexity(system_name: str, ask_for_favicon=False) -> dict:
    prompt = build_favicon_prompt(system_name) if ask_for_favicon else build_prompt(system_name)

    headers = {
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 1000
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
        return {}


def build_favicon_prompt(system_url: str) -> str:
    return f"""
Given the website URL "{system_url}", return a JSON object with a single field:
- favicon_url: A fully-qualified direct link to the site's main favicon.

Respond with only a valid JSON object. Example format:

{{
  "favicon_url": "..."
}}
"""


def build_prompt(system_name):
    return f"""
Given the system name "{system_name}", return a JSON object with the following fields:

- name: The product or system name.
- vendor: The company or organization behind it.
- description: A short description (max 160 characters).
- rating: Integer from 1 to 5 based on trustworthiness, maturity, and usage.
- url: Main homepage or product page.
- tags: Choose only from this fixed list: {TAGS}.
- risk: One of "low", "medium", "high".
- vertical: Choose only from this list: {VERTICALS}.
- hostname_regex: Regex to match the domain (e.g., "^chatgpt\\.com$").
- ai_type: One of: "pure_ai", "ai_integrated", or "infra_api".

Respond with only a JSON object.
"""

async def query_openai(system_name: str, client: openai.AsyncOpenAI) -> dict:
    try:
        prompt = build_prompt(system_name)
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        text = response.choices[0].message.content
        json_obj = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(json_obj.group(0)) if json_obj else {}
    except Exception as e:
        print(f"❌ OpenAI error for {system_name}: {e}")
        return {}


async def scrape_favicon(url: str) -> str:
    browser = None
    try:
        async with async_playwright() as p:
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
            await browser.close()
            return urljoin(final_url, favicon_href)
    except Exception as e:
        if browser:
            await browser.close()
        print(f"⚠️ Favicon scrape failed for {url}: {e}")
        return ""


def is_incomplete(ai_info: dict) -> bool:
    if not ai_info:
        return True
    return (
        ai_info.get("vendor", "").lower() == "unknown"
        or "unspecified" in ai_info.get("description", "").lower()
        or not ai_info.get("url")
        or not ai_info.get("name")
    )


async def enrich_system(system_name: str, client: openai.AsyncOpenAI) -> dict:
    print(f"🔍 Processing: {system_name}")
    ai_info = await query_openai(system_name, client)

    if is_incomplete(ai_info):
        print(f"⚠️ Incomplete info from OpenAI for {system_name}, retrying via Perplexity...")
        ai_info = await get_page_info_from_perplexity(system_name, ask_for_favicon=False)

    if not ai_info or not ai_info.get("url"):
        print(f"❌ Skipping {system_name}, still incomplete after fallback.")
        return {}

    favicon_url = await scrape_favicon(ai_info["url"])
    if not favicon_url and not ai_info.get("favicon_url"):
        print(f"⚠️ Favicon missing, asking Perplexity...")
        perplexity_info = await get_page_info_from_perplexity(system_name, ask_for_favicon=True)
        ai_info["favicon_url"] = perplexity_info.get("favicon_url", "")

    ai_info["favicon_url"] = favicon_url
    print(f"✅ Done: {system_name}")
    return ai_info


async def main(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        system_names = [line.strip() for line in f if line.strip()]

    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(5)

    async def limited_enrich_system(system_name, client):
        async with semaphore:
            return await enrich_system(system_name, client)

    tasks = [limited_enrich_system(name, client) for name in system_names]
    all_results = await asyncio.gather(*tasks)
    results = [r for r in all_results if r]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(results)} systems to {output_path}")


if __name__ == "__main__":
    asyncio.run(main("ai_systems.txt", "ai_systems.json"))