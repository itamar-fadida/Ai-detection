import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re

MIN_SCORE = 10
METADATA_MULTIPLE = 2

HIGH_AI_SCORE = 8
MEDIUM_AI_SCORE = 5
LOW_AI_SCORE = 2

HIGH_AI_WORDS = (
    " AI ", "AI-", " Artificial Intelligence ", " Machine Learning ", " Deep Learning ", " Neural Networks ",
    " Large Language Model ", " Natural Language Processing ", " Generative AI ", " ML ", " Machine Learning ", "ML-"
)

MEDIUM_AI_WORDS = (
    " Analysis ", " Data Science ", " Algorithm ", " Transformer Model ", " Sentiment Analysis ",
    " Image Processing ", " Text Processing ", " Automation ", " Crawler ", " Generation", " Automated "
)

LOW_AI_WORDS = (
    " Chatbot ", " Bot "
)

AI_DICT_DETECTION = {word.lower(): HIGH_AI_SCORE for word in HIGH_AI_WORDS}
AI_DICT_DETECTION.update({word.lower(): MEDIUM_AI_SCORE for word in MEDIUM_AI_WORDS})
AI_DICT_DETECTION.update({word.lower(): LOW_AI_SCORE for word in LOW_AI_WORDS})


async def scrape_web(url: str) -> str:
    """Scrape the webpage content using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)  # Wait up to 60 seconds

        content = await page.content()  # Get the raw HTML content
        await browser.close()
        return content


def format_web_to_open_ai(web_scrape_content: str) -> dict:
    """ Gets the web content and return dict with:
    1. icon_url
    2. matadata
    3. formatted content (no html tags, no css)
    4.
    """
    pass


def clear_web_content(web_scrape_content: str) -> str:
    """Remove HTML tags, CSS, and unnecessary text from the scraped content."""
    soup = BeautifulSoup(web_scrape_content, "html.parser")

    # Remove script, style, and meta tags
    for tag in soup(["script", "style", "meta", "noscript"]):
        tag.extract()

    # Get the visible text
    text = soup.get_text(separator=" ")

    # Clean extra spaces, new lines, and non-ASCII characters
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters

    return text.strip()


def ai_probability(web_content: str) -> int:
    """Calculate AI probability score based on detected keywords in content."""
    score = 0
    web_content_lower = web_content.lower()

    for keyword, value in AI_DICT_DETECTION.items():
        if keyword in web_content_lower:
            score += value

    return score


async def main():
    url = "https://chatgpt.com/"
    print(f"Scraping: {url}")

    # Scrape the web page
    raw_html = await scrape_web(url)

    # Clean content
    cleaned_text = clear_web_content(raw_html)

    # AI detection score
    ai_score = ai_probability(cleaned_text)

    print(f"AI Probability Score: {ai_score}")
    print(f"Extracted Text Preview: {cleaned_text[:500]}")  # Show first 500 chars


if __name__ == "__main__":
    asyncio.run(main())
