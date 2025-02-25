from dotenv import load_dotenv

import asyncio
import os
from crawl4ai import *


async def main():
    
    llm_instruction = "Extract all products with 'name', 'type', 'price' and 'price under discount' from the following content."
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=False,
        verbose=True,
    )
    llm_extration_strategy = LLMExtractionStrategy(
        provider="openai/gpt-4o-mini",
        api_token=os.getenv("OPENAI_API_KEY"),
        instruction=llm_instruction
    )
    run_config = CrawlerRunConfig(
        extraction_strategy=llm_extration_strategy,
        #css_selector="[class^='bg-highlight']",
        cache_mode=CacheMode.BYPASS,

    )
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Run the crawler on a URL
        result = await crawler.arun(
            url="https://heyharper.com/eu/en/products",
            config=run_config)

        # Print the extracted content
        print(result.markdown)
        print(result.links)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())