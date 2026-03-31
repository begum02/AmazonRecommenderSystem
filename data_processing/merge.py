import pandas as pd
import numpy as np
from pathlib import Path
import glob
import logging

# ============================================================
# BU KOD NE YAPIYOR?
# ============================================================
# Bu script, Amazon veri setindeki farklı kategorilere ait:
#   - ürün verilerini (items)
#   - kullanıcı yorumlarını (reviews)
#
# şu adımlarla işler:
#
# 1. Klasördeki *_items ve *_reviews dosyalarını bulur
# 2. Parquet formatındaki verileri yükler
# 3. Kolon isimlerini STANDARDIZE eder (ortak formata getirir)
# 4. item_id üzerinden ürün + yorum verisini birleştirir
# 5. Eksik verileri temizler
# 6. Tüm kategorileri tek dataset haline getirir
# 7. Sonucu parquet olarak kaydeder
#
# Amaç:
# 👉 Tavsiye sistemi / ML modeli için temiz ve birleşik veri üretmek
# ============================================================


# ==============================
# LOG AYARLARI
# ==============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AmazonDataMerger:

    # =========================================
    # BAŞLANGIÇ (PATH AYARLARI)
    # =========================================
    def __init__(self, raw_data_path: Path, output_path: Path):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)/"merged"

        # output klasörü yoksa oluştur
        self.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"RAW PATH: {self.raw_data_path}")

    # =========================================
    # 1. DOSYALARI BUL
    # =========================================
    def get_files(self):

        categories = {}

        # *_items.parquet dosyalarını bul
        items_files = glob.glob(str(self.raw_data_path / "*_items.parquet"))

        for item_file in items_files:
            # dosya isminden kategori adını çıkar
            name = Path(item_file).stem.replace("_items", "")

            # aynı kategoriye ait review dosyası
            review_file = self.raw_data_path / f"{name}_reviews.parquet"

            # eğer review dosyası varsa eşleştir
            if review_file.exists():
                categories[name] = {
                    "items": item_file,
                    "reviews": str(review_file)
                }

        logger.info(f"Kategori sayısı: {len(categories)}")
        return categories

    # =========================================
    # 2. VERİYİ YÜKLE
    # =========================================
    def load(self, files):

        # parquet dosyalarını dataframe olarak yükle
        items = pd.read_parquet(files["items"], engine='fastparquet')
        reviews = pd.read_parquet(files["reviews"], engine='fastparquet')

        return items, reviews

    # =========================================
    # 3. STANDARDIZE (EN KRİTİK KISIM)
    # =========================================
    def standardize(self, items, reviews, category):

        # ============================================================
        # AMAÇ:
        # Farklı kategorilerdeki veri setlerinin kolon isimleri farklı olabilir.
        # Örneğin:
        #   - parent_asin / asin / product_id
        #   - text / review / comment
        #
        # Bu yüzden tüm verileri TEK BİR ORTAK FORMATTA topluyoruz.
        # ============================================================

        # -------------------------
        # ITEMS TABLOSU
        # -------------------------

        # kolon isimlerini standart hale getiriyoruz
        # parent_asin → item_id (tüm sistemde ortak ID olacak)
        items = items.rename(columns={
            "parent_asin": "item_id",
            "title": "title",   # zaten doğru ama garantiye alıyoruz
            "price": "price"
        })

        # her ürüne ait kategori bilgisini ekliyoruz
        items["category"] = category

        # aynı ürün birden fazla varsa (duplicate) kaldırıyoruz
        # çünkü aynı item_id tekrar etmemeli
        items = items.drop_duplicates(subset=["item_id"])

        # -------------------------
        # REVIEWS TABLOSU
        # -------------------------

        # yorum verilerindeki kolonları da standart hale getiriyoruz
        reviews = reviews.rename(columns={
            "parent_asin": "item_id",     # ürünle join için kritik
            "user_id": "user_id",
            "rating": "rating",
            "text": "review_text",        # text → review_text
            "timestamp": "timestamp"
        })

        # timestamp → datetime dönüşümü
        # çünkü model veya analiz için tarih formatı gerekli
        reviews["review_date"] = pd.to_datetime(
            reviews["timestamp"],
            unit="s",         # saniye formatı
            errors="coerce"   # hatalı değerleri NaT yap
        )

        # ============================================================
        # SONUÇ:
        # Artık tüm datasetlerde şu kolonlar aynı:
        #   item_id, user_id, rating, review_text, timestamp
        #
        # Bu sayede merge işlemi sorunsuz çalışır
        # ============================================================

        return items, reviews

    # =========================================
    # 4. MERGE (BİRLEŞTİRME)
    # =========================================
    def merge(self, items, reviews, category):

        # item_id üzerinden join yapıyoruz
        df = reviews.merge(items, on="item_id", how="inner")

        # kritik kolonlar boşsa o satırları siliyoruz
        df = df.dropna(subset=["user_id", "item_id"])

        # kullanmak istediğimiz final kolonlar
        final_cols = [
            "user_id", "item_id", "rating", "review_text",
            "title", "price", "category",
            "timestamp", "review_date"
        ]

        # sadece mevcut kolonları al (bazı datasetlerde eksik olabilir)
        df = df[[c for c in final_cols if c in df.columns]]

        logger.info(f"{category} → {len(df)} satır")

        return df

    # =========================================
    # 5. TÜM SÜRECİ ÇALIŞTIR
    # =========================================
    def run(self):
        files = self.get_files()
        all_df = []

        for cat, f in files.items():
            try:
                logger.info(f"İşleniyor: {cat}")
                items, reviews = self.load(f)
                items, reviews = self.standardize(items, reviews, cat)
                merged = self.merge(items, reviews, cat)

                if len(merged) > 0:
                    all_df.append(merged)
            except Exception as e:
                # Hata veren dosyayı raporla ama kodu durdurma
                logger.error(f"HATA: {cat} kategorisi işlenemedi! Sebep: {e}")
                continue 

        if not all_df:
            logger.error("Hiçbir kategori başarıyla birleştirilemedi!")
            return pd.DataFrame()

        master = pd.concat(all_df, ignore_index=True)
        logger.info(f"FINAL: {len(master)} satır birleştirildi.")
        return master

    # =========================================
    # 6. KAYDET
    # =========================================
    def save(self, df):

        path = self.output_path / "merged_dataset.parquet"

        # sonucu parquet olarak kaydet
        df.to_parquet(path, index=False)

        logger.info(f"KAYDEDİLDİ: {path}")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":

    BASE = Path(__file__).resolve().parent.parent

    merger = AmazonDataMerger(
        raw_data_path=Path(__file__).resolve().parent / "raw",
        output_path=BASE / "data"
    )

    logger.info("Dataset oluşturuluyor...")

    df = merger.run()

    if len(df) > 0:
        merger.save(df)
        print(df.head())
    else:
        logger.error("Dataset boş!")