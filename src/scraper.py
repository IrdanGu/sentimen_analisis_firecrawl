"""
scraper.py — Ambil data review menggunakan Firecrawl
"""

import os
import json
import pandas as pd
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()


class ReviewScraper:
    def __init__(self):
        self.app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    def scrape_single_page(self, url: str) -> dict:
        """
        Scrape konten dari satu halaman URL.
        Return: dict berisi markdown content
        """
        result = self.app.scrape_url(
            url,
            params={
                "formats": ["markdown", "extract"],
                "extract": {
                    "prompt": """
                        Ekstrak semua review / ulasan pengguna dari halaman ini.
                        Untuk setiap review, ambil:
                        - text: isi review / komentar
                        - rating: nilai rating (1-5) jika ada
                        - date: tanggal review jika ada
                        - product: nama produk jika ada
                        Return sebagai array of objects.
                    """
                }
            }
        )
        return result

    def scrape_multiple_urls(self, urls: list) -> list:
        """
        Scrape banyak URL sekaligus (batch scraping).
        Lebih efisien dibanding scrape satu-satu.
        """
        results = self.app.batch_scrape_urls(
            urls,
            params={
                "formats": ["extract"],
                "extract": {
                    "prompt": """
                        Ekstrak semua review / ulasan dari halaman ini.
                        Return array dengan fields: text, rating, date, product.
                    """
                }
            }
        )
        return results

    def crawl_review_site(self, base_url: str, max_pages: int = 20) -> list:
        """
        Crawl seluruh website review — cocok untuk situs dengan banyak halaman.
        Firecrawl akan otomatis follow links di dalamnya.
        """
        crawl_result = self.app.crawl_url(
            base_url,
            params={
                "limit": max_pages,
                "scrapeOptions": {
                    "formats": ["extract"],
                    "extract": {
                        "prompt": """
                            Ekstrak semua ulasan/review dari halaman ini.
                            Return array: [{ text, rating, date, product }]
                            Jika tidak ada review, return array kosong [].
                        """
                    }
                }
            }
        )
        return crawl_result

    def save_raw_data(self, data: list, filename: str = "raw_reviews.json"):
        """Simpan hasil scraping ke file JSON."""
        path = f"data/raw/{filename}"
        os.makedirs("data/raw", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Data tersimpan di {path}")

    def to_dataframe(self, raw_data: list) -> pd.DataFrame:
        """
        Ubah hasil scraping mentah menjadi DataFrame yang rapi.
        """
        reviews = []
        for page in raw_data:
            # Data ada di field 'extract' dari tiap halaman
            extracted = page.get("extract", [])
            if isinstance(extracted, list):
                reviews.extend(extracted)

        df = pd.DataFrame(reviews)
        print(f"📊 Total review terkumpul: {len(df)}")
        return df


# =============================================================
# CONTOH PENGGUNAAN
# =============================================================
if __name__ == "__main__":
    scraper = ReviewScraper()

    # Contoh 1: Scrape satu halaman review
    # (Ganti URL ini dengan website yang ingin kamu scrape)
    url = "https://www.amazon.com/product-reviews/B08N5LNQCX"
    result = scraper.scrape_single_page(url)
    print(result)

    # Contoh 2: Batch scrape dari banyak URL produk
    urls = [
        "https://example.com/product/1/reviews",
        "https://example.com/product/2/reviews",
        "https://example.com/product/3/reviews",
    ]
    # results = scraper.scrape_multiple_urls(urls)
    # df = scraper.to_dataframe(results)
    # scraper.save_raw_data(results, "batch_reviews.json")

    # Contoh 3: Crawl website review
    # crawl_result = scraper.crawl_review_site("https://example.com/reviews", max_pages=50)
    # df = scraper.to_dataframe(crawl_result)
    # print(df.head())
