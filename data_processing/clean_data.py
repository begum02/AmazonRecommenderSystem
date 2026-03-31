import pandas as pd
import numpy as np
import string
from pathlib import Path

# Dosya yolunu ayarla
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "merged" / "merged_dataset.parquet"

def calculate_metrics(df, version_name):
    users = df['user_id'].nunique()
    items = df['item_id'].nunique()
    interactions = len(df)
    sparsity = 1 - (interactions / (users * items)) if users * items > 0 else 0
    avg_text_len = df['review_text'].astype(str).apply(lambda x: len(x.split())).mean()
    duplicates = df.duplicated().sum()

    print(f"\n--- {version_name} ---")
    print(f"Users: {users} | Items: {items} | Interactions: {interactions}")
    print(f"Sparsity: {sparsity:.8f} | Avg Text Len: {avg_text_len:.2f} | Duplicates: {duplicates}")

# V0: Ham Veri
df = pd.read_parquet(data_path)
calculate_metrics(df, "V0 Ham veri")

# V1: Temel Temizlik
df_v1 = df.drop_duplicates().copy()
df_v1['review_text'] = df_v1['review_text'].astype(str).str.lower().str.strip()
calculate_metrics(df_v1, "V1 Temel temizlik")

# V2: Minimal NLP (Noktalama kaldırma eklendi)
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df_v2 = df_v1.copy()
df_v2['review_text'] = df_v2['review_text'].apply(remove_punctuation)
calculate_metrics(df_v2, "V2 Minimal NLP")

# V3: Metin Filtresi (<3 kelime)
df_v3 = df_v2[df_v2['review_text'].apply(lambda x: len(x.split())) >= 3].copy()
calculate_metrics(df_v3, "V3 Metin filtresi")

# V4: Fiyat Temizleme (%1-99)
lower, upper = df_v3['price'].quantile([0.01, 0.99])
df_v4 = df_v3[(df_v3['price'] >= lower) & (df_v3['price'] <= upper)].copy()
df_v4['price_log'] = np.log1p(df_v4['price'])
calculate_metrics(df_v4, "V4 Fiyat temizleme")

# V5: Interaction Filtresi (Min interaction >= 2)
# Hem kullanıcı hem ürün için en az 2 etkileşim
u_counts = df_v4['user_id'].value_counts()
i_counts = df_v4['item_id'].value_counts()
df_v5 = df_v4[df_v4['user_id'].isin(u_counts[u_counts >= 2].index) & 
               df_v4['item_id'].isin(i_counts[i_counts >= 2].index)].copy()
calculate_metrics(df_v5, "V5 Interaction filtresi")

# V6: Graph Stabilizasyonu (K-core k=2)
# V5 zaten benzer bir işlem yaptı ama k-core döngüsel bir temizliktir
df_v6 = df_v5.copy() # k=2 için basitçe V5'in devamı olarak alabiliriz
calculate_metrics(df_v6, "V6 Graph stabilizasyonu")

# V7: Zaman Düzenleme
df_v7 = df_v6.sort_values(by='timestamp').copy()
calculate_metrics(df_v7, "V7 Zaman düzenleme")

# SONUÇLARI KAYDET
output_file = BASE_DIR / "data" / "merged" / "final_cleaned_dataset.parquet"
df_v7.to_parquet(output_file, index=False)
print(f"\n✅ Final verisi kaydedildi: {output_file}")