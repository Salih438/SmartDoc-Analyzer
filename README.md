# SmartDoc Analyzer

Belge görüntülerini klasik OpenCV + NumPy yöntemleriyle analiz eden; sınır tespiti, perspektif düzeltme ve ön işleme sağlayan proje. Bu aşamada OCR ve Hough uygulanmıyor.

## Kurulum ve çalıştırma

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
streamlit run app/main.py
```

Arayüz tek PNG/JPG/JPEG dosyası alır. Görsel önizleme ekranda ölçeklenebilir; analiz ve pipeline yüklenen tam çözünürlükte çalışır. Belge tespiti başarısızsa işlem durur; düzeltilmemiş tüm görüntüyü kullanmak için kullanıcı açık onay vermelidir. Sonuç PNG ve config/adım JSON'u indirilebilir.

## Modüller

- `src/analysis/`: Unicode uyumlu dosya I/O, görüntü bilgisi ve histogram.
- `src/document/`: belge tespiti, köşe sıralama ve perspektif dönüşümü.
- `src/preprocessing/`, `src/segmentation/`: gri dönüşüm, iyileştirme, filtre, keskinleştirme, eşikleme ve morfoloji.
- `src/evaluation/`: sentetik sahneler, gürültü, metrikler ve CSV deneyleri.
- `src/pipeline.py`: yapılandırılabilir görüntü işleme hattı.
- `app/`: sunum katmanı; algoritmalar `src/` içinde kalır.

## Referans maskesi ve deney notları

Belge ve metin GT'si aynı geometrik tariflerden bağımsız çizimlerle üretilir: belge 4× çözünürlükte LINE_AA, GT ise 4× LINE_8 ile çizilir; GT'de en az %50 piksel alanı kaplayan örnek 255 yapılır. Son karşılaştırmada `doc <= 127` foreground sayısı 313,753, GT foreground 313,785, kesişim 313,748 ve farklı piksel sayısı 42 / 3,628,800 (%0.001157) çıktı. Dolayısıyla Global(127), bu bağımsız GT'nin tanımı değildir; aralarındaki fark yalnızca sınır rasterizasyonundaki 42 pikseldir. Bu sayımlar `results/gt_mask_comparison.csv` içinde saklanır.

`low_contrast_hard` sahnesinde belge/masa düzeyleri 215/205; ölçülen paper-side minus desk-side sınır farkı 10 gri seviye oldu. Noise'lu varyant ve pipeline shadow/noise sonuçları da ilgili CSV'lerde seed/parametreleriyle saklanır. Deney tabloları CSV dosyalarından kodla üretilir; başarısız tespit kayıtları atılmaz.

## Sınırlar ve sonraki aşama

Klasik kontur yaklaşımı düşük kontrast, ters polarite, kadraj dışı belge ve karmaşık sahnelerde başarısız olabilir; sonuçlar her görsel için garanti edilmez. UI'da referans görsel yoksa PSNR/SSIM üretilmez ve kalite karşılaştırması “referans yok” olarak belirtilir. OCR ayrı sonraki aşamadır.
