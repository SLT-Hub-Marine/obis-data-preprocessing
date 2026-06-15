import re
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import trafilatura
from playwright.sync_api import sync_playwright

SAMPLINGPROTOCOL_COLUMN_FPATH = "samplingProtocol_column_data/aggregate_analysis/samplingProtocol_unique_value_counts.csv"
SAVE_DIR = Path("samplingProtocol_column_Data/texts_extracted_from_urls")

if __name__ == "__main__":
    # url_pattern found 46 matches in samplingProtocol column
    url_pattern = re.compile(
        r"^https?://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$",
        re.IGNORECASE
    )

    # loose_url_pattern found 67 matches in samplingProtocol column
    loose_url_pattern = re.compile(
        r"^(https?://)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$",
        re.IGNORECASE
    )

    data = pd.read_csv(SAMPLINGPROTOCOL_COLUMN_FPATH)["unique_samplingProtocol_value"]

    metadata = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, data_value in tqdm(enumerate(data)):
            url_str = str(data_value)

            if not loose_url_pattern.match(url_str):
                continue

            downloaded = None

            # check if URL belongs to protocols.io (or other JS-heavy sites)
            if "protocols.io" in url_str:
                try:
                    # Use Playwright to render the JavaScript
                    page.goto(url_str, wait_until="networkidle")
                    downloaded = page.content()
                except Exception as e:
                    print(f"Skipping {url_str} due to Playwright error: {e}")
                    metadata.append({"samplingProtocol_uid": i, "unique_samplingProtocol_value": data_value,
                                     "successful_extraction": False,
                                     "extracted_md_fpath": "",
                                     "extraction_notes": "failed to read site content"})
                    continue
            else:
                # fall back to trafilatura's fast static fetching for regular sites
                downloaded = trafilatura.fetch_url(url_str)

            if downloaded:
                clean_markdown = trafilatura.extract(downloaded, output_format='markdown')

                if clean_markdown:
                    save_fpath = f"extracted_url_{i:05}.md"
                    with open(SAVE_DIR / save_fpath, "w", encoding="utf-8") as f:
                        f.write(clean_markdown)

                    metadata.append({"samplingProtocol_uid": i, "unique_samplingProtocol_value": data_value,
                                     "successful_extraction": True,
                                     "extracted_md_fpath": str(save_fpath),
                                     "extraction_notes": ""})
                else:
                    metadata.append({"samplingProtocol_uid": i, "unique_samplingProtocol_value": data_value,
                                     "successful_extraction": False,
                                     "extracted_md_fpath": "",
                                     "extraction_notes": "failed to extract markdown from downloaded site content"})

            else:
                metadata.append({"samplingProtocol_uid": i, "unique_samplingProtocol_value": data_value,
                                 "successful_extraction": False,
                                 "extracted_md_fpath": "",
                                 "extraction_notes": "failed to download site content"})

        browser.close()

    pd.DataFrame(metadata).to_csv(SAVE_DIR / "url_extraction_metadata.csv", index=False)


