from pathlib import Path
import gc
import time
import pandas as pd
from huggingface_hub import hf_hub_download


# ======================================================
# AYARLAR
# ======================================================
# Hugging Face üzerindeki veri seti kaynağı
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

# Proje kök dizini ve ham veri kayıt klasörü
PROJECT_ROOT = Path("C:/Users/BBN/Documents/amazonreviewdatasetagent")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Veri boyutunu kontrol altında tutmak için limitler
MAX_ITEMS_PER_CATEGORY = 15000
MAX_REVIEWS_PER_ITEM = 15
MAX_REVIEWS_PER_CATEGORY = 80000
MIN_RATING_NUMBER = 5

# Büyük jsonl dosyalarını belleği taşırmadan okumak için chunk boyutları
META_CHUNK_SIZE = 25000
REVIEW_CHUNK_SIZE = 50000

# İndirme sırasında ağ hatalarına karşı tekrar deneme ayarı
MAX_DOWNLOAD_RETRY = 5
RETRY_WAIT_SECONDS = 10


# ======================================================
# 15 KATEGORİ
# ======================================================
CATEGORIES = [
    "All_Beauty",
    "Health_and_Personal_Care",
    "Handmade_Products",
    "Tools_and_Home_Improvement",
    "Grocery_and_Gourmet_Food",
    "Home_and_Kitchen",
    "Pet_Supplies",
    "Office_Products",
    "Sports_and_Outdoors",
    "Toys_and_Games",
    "Baby_Products",
    "Cell_Phones_and_Accessories",
    "Patio_Lawn_and_Garden",
    "Arts_Crafts_and_Sewing"
]


# ======================================================
# YARDIMCI
# ======================================================
def already_processed(category: str) -> bool:
    """
    Aynı kategori daha önce işlenmiş mi kontrol eder.
    Hem item hem review parquet dosyası varsa bu kategori atlanır.
    """
    items_path = RAW_DIR / f"{category}_items.parquet"
    reviews_path = RAW_DIR / f"{category}_reviews.parquet"
    return items_path.exists() and reviews_path.exists()


def save_parquet(df: pd.DataFrame, path: Path):
    """DataFrame'i parquet olarak kaydeder."""
    df.to_parquet(path, index=False)
    print(f"[OK] Kaydedildi -> {path}")


def safe_download(filename: str, label: str) -> str:
    """
    Dosyayı güvenli şekilde indirir.
    İndirme hatası olursa belirlenen sayıda tekrar dener.
    """
    last_error = None

    for attempt in range(1, MAX_DOWNLOAD_RETRY + 1):
        try:
            print(f"[INFO] {label} indiriliyor | deneme {attempt}/{MAX_DOWNLOAD_RETRY}")
            path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
            )
            print(f"[OK] Hazır -> {path}")
            return path

        except Exception as e:
            last_error = e
            print(f"[WARN] {label} indirme hatası: {e}")

            if attempt < MAX_DOWNLOAD_RETRY:
                print(f"[INFO] {RETRY_WAIT_SECONDS} saniye sonra tekrar denenecek...")
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(f"{label} indirilemedi. Son hata: {last_error}")


def download_meta_file(category: str) -> str:
    """Kategoriye ait metadata dosyasını indirir."""
    return safe_download(
        filename=f"raw/meta_categories/meta_{category}.jsonl",
        label=f"Metadata {category}",
    )


def download_review_file(category: str) -> str:
    """Kategoriye ait review dosyasını indirir."""
    return safe_download(
        filename=f"raw/review_categories/{category}.jsonl",
        label=f"Review {category}",
    )


# ======================================================
# META TEMİZLEME
# ======================================================
def clean_meta_chunk(meta_chunk: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Metadata chunk'ını temizler ve filtreler.

    Kurallar:
    - parent_asin boş olmamalı
    - price > 0 olmalı
    - average_rating dolu olmalı
    - rating_number >= MIN_RATING_NUMBER olmalı
    """
    required_cols = ["parent_asin", "price", "average_rating", "rating_number"]

    for col in required_cols:
        if col not in meta_chunk.columns:
            raise ValueError(f"Metadata içinde eksik kolon: {col}")

    # Analizde kullanılabilecek kolonları mümkün olduğunca koru
    keep_cols = [
        "parent_asin",
        "title",
        "main_category",
        "average_rating",
        "rating_number",
        "price",
        "store",
        "features",
        "description",
        "categories",
        "details",
        "images",
    ]
    existing_cols = [c for c in keep_cols if c in meta_chunk.columns]
    meta_chunk = meta_chunk[existing_cols].copy()

    # Price string temizliği: "$12.99" -> "12.99"
    meta_chunk["price"] = (
        meta_chunk["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    # Sayısal kolonları güvenli şekilde dönüştür
    meta_chunk["price"] = pd.to_numeric(meta_chunk["price"], errors="coerce")
    meta_chunk["average_rating"] = pd.to_numeric(meta_chunk["average_rating"], errors="coerce")
    meta_chunk["rating_number"] = pd.to_numeric(meta_chunk["rating_number"], errors="coerce").fillna(0)

    # İş kuralı filtreleri
    meta_chunk = meta_chunk[
        (meta_chunk["parent_asin"].notna()) &
        (meta_chunk["price"] > 0) &
        (meta_chunk["average_rating"].notna()) &
        (meta_chunk["rating_number"] >= MIN_RATING_NUMBER)
    ].copy()

    meta_chunk["category"] = category
    return meta_chunk


def load_and_clean_meta_chunked(file_path: str, category: str) -> pd.DataFrame:
    """
    Metadata dosyasını chunk halinde okuyup temizler.
    Bellek taşmasını önlemek için her chunk ayrı işlenir.
    """
    print(f"[INFO] Metadata chunk'lı okunuyor: {file_path}")

    cleaned_chunks = []
    total_rows = 0
    total_kept = 0

    chunk_iter = pd.read_json(file_path, lines=True, chunksize=META_CHUNK_SIZE)

    for i, chunk in enumerate(chunk_iter, start=1):
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        print(f"[INFO] Meta chunk {i} başladı | satır sayısı: {chunk_rows:,}")

        cleaned = clean_meta_chunk(chunk, category)
        cleaned_chunks.append(cleaned)
        total_kept += len(cleaned)

        print(f"[INFO] Meta chunk {i} sonrası kalan ürün: {len(cleaned):,}")

        # Bellek temizliği
        del chunk
        del cleaned
        gc.collect()

    print(f"[INFO] Toplam okunan metadata satırı: {total_rows:,}")
    print(f"[INFO] Filtre sonrası metadata satırı: {total_kept:,}")

    if not cleaned_chunks:
        return pd.DataFrame()

    meta_df = pd.concat(cleaned_chunks, ignore_index=True)

    # Aynı parent_asin birden fazla kez gelmişse tekilleştir
    meta_df = meta_df.drop_duplicates(subset=["parent_asin"]).copy()

    # Kategori başına ürün limitini uygula
    if len(meta_df) > MAX_ITEMS_PER_CATEGORY:
        meta_df = meta_df.sample(MAX_ITEMS_PER_CATEGORY, random_state=RANDOM_STATE)

    print(f"[INFO] Final metadata ürün sayısı: {len(meta_df):,}")
    return meta_df


# ======================================================
# REVIEW TEMİZLEME
# ======================================================
def load_and_filter_reviews_chunked(file_path: str, valid_items: set) -> pd.DataFrame:
    """
    Review dosyasını chunk halinde okuyup sadece geçerli ürünlere ait
    review'ları tutar.
    """
    print(f"[INFO] Review chunk'lı okunuyor: {file_path}")

    filtered_chunks = []
    total_rows = 0
    total_kept = 0

    chunk_iter = pd.read_json(file_path, lines=True, chunksize=REVIEW_CHUNK_SIZE)

    for i, chunk in enumerate(chunk_iter, start=1):
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        print(f"[INFO] Review chunk {i} başladı | satır sayısı: {chunk_rows:,}")

        if "parent_asin" not in chunk.columns:
            print(f"[WARN] Review chunk {i} içinde parent_asin yok, atlandı.")
            del chunk
            gc.collect()
            continue

        # Sadece ihtiyacımız olan kolonları al
        preferred_cols = [
            "rating",
            "title",
            "text",
            "asin",
            "parent_asin",
            "user_id",
            "timestamp",
            "helpful_vote",
            "verified_purchase",
        ]
        existing_cols = [c for c in preferred_cols if c in chunk.columns]
        chunk = chunk[existing_cols].copy()

        # Sadece metadata'da kalan ürünlere ait review'ları tut
        chunk = chunk[chunk["parent_asin"].isin(valid_items)].copy()

        if chunk.empty:
            print(f"[INFO] Review chunk {i} sonrası kalan review: 0")
            del chunk
            gc.collect()
            continue

        # Güvenli parent_asin temizliği
        chunk = chunk[chunk["parent_asin"].notna()].copy()

        # Chunk içinde ürün başına review limit uygula
        chunk = (
            chunk.groupby("parent_asin", group_keys=False)
            .head(MAX_REVIEWS_PER_ITEM)
            .copy()
        )

        filtered_chunks.append(chunk)
        total_kept += len(chunk)

        print(f"[INFO] Review chunk {i} sonrası kalan review: {len(chunk):,}")

        del chunk
        gc.collect()

    print(f"[INFO] Toplam okunan review satırı: {total_rows:,}")
    print(f"[INFO] Chunk bazında tutulan review toplamı: {total_kept:,}")

    if not filtered_chunks:
        return pd.DataFrame()

    review_df = pd.concat(filtered_chunks, ignore_index=True)

    # Chunk'lar arası tekrarları sınırla (ürün başına üst limit)
    review_df = (
        review_df.groupby("parent_asin", group_keys=False)
        .head(MAX_REVIEWS_PER_ITEM)
        .copy()
    )

    # Kategori başına toplam review sayısını sınırla
    if len(review_df) > MAX_REVIEWS_PER_CATEGORY:
        review_df = review_df.sample(MAX_REVIEWS_PER_CATEGORY, random_state=RANDOM_STATE)

    print(f"[INFO] Final review sayısı: {len(review_df):,}")
    return review_df


# ======================================================
# KATEGORİ İŞLEME
# ======================================================
def process_category(category: str):
    """
    Tek bir kategori için uçtan uca işlem:
    1) Meta indir ve temizle
    2) Geçerli ürün setini çıkar
    3) Review indir ve sadece geçerli ürünleri tut
    4) Sonuçları parquet olarak kaydet
    """
    print("\n" + "=" * 80)
    print(f"[INFO] Kategori işleniyor: {category}")
    print("=" * 80)

    # Daha önce işlendi ise tekrar işleme
    if already_processed(category):
        print(f"[SKIP] Zaten işlenmiş: {category}")
        return

    # ---------------- META ----------------
    meta_path = download_meta_file(category)
    meta_df = load_and_clean_meta_chunked(meta_path, category)

    if meta_df.empty:
        print("[WARN] Metadata boş, kategori atlandı.")
        return

    valid_items = set(meta_df["parent_asin"].dropna().unique())
    print(f"[INFO] Geçerli ürün sayısı: {len(valid_items):,}")

    gc.collect()

    # ---------------- REVIEW ----------------
    review_path = download_review_file(category)
    review_df = load_and_filter_reviews_chunked(review_path, valid_items)

    if review_df.empty:
        print("[WARN] Review boş, kategori atlandı.")
        return

    # Review'da gerçekten kalan ürünlere göre metadata'yı hizala
    valid_items_after = set(review_df["parent_asin"].dropna().unique())
    meta_df = meta_df[meta_df["parent_asin"].isin(valid_items_after)].copy()

    # ---------------- SAVE ----------------
    save_parquet(meta_df, RAW_DIR / f"{category}_items.parquet")
    save_parquet(review_df, RAW_DIR / f"{category}_reviews.parquet")

    print(f"[SUMMARY] {category} | items={len(meta_df):,} | reviews={len(review_df):,}")

    del meta_df
    del review_df
    gc.collect()


# ======================================================
# MAIN
# ======================================================
def main():
    """Tüm kategorileri sırayla işler ve toplam süreyi raporlar."""
    start_time = time.time()

    for cat in CATEGORIES:
        try:
            process_category(cat)
        except KeyboardInterrupt:
            print("\n[STOP] İşlem kullanıcı tarafından durduruldu.")
            raise
        except Exception as e:
            # Tek kategori hatası tüm süreci durdurmasın
            print(f"[ERROR] {cat} -> {e}")
            gc.collect()
            continue

    elapsed = time.time() - start_time
    print(f"\n[OK] Tüm işlem tamamlandı. Süre: {elapsed/60:.2f} dakika")


if __name__ == "__main__":
    main()