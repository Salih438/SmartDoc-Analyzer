# SmartDoc Analyzer

Belge görüntülerini klasik OpenCV + NumPy yöntemleriyle analiz eden; sınır tespiti, perspektif düzeltme, ön işleme ve isteğe bağlı Tesseract OCR sağlayan proje. Hough uygulanmıyor.

## Kurulum ve çalıştırma

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
streamlit run app/main.py
```

Arayüz tek PNG/JPG/JPEG dosyası alır. Görsel önizleme ekranda ölçeklenebilir; analiz ve pipeline yüklenen tam çözünürlükte çalışır. Belge tespiti başarısızsa işlem durur; düzeltilmemiş tüm görüntüyü kullanmak için kullanıcı açık onay vermelidir. Sonuç PNG ve config/adım JSON'u indirilebilir.

OCR sekmesi Tesseract sistem uygulaması ve uygun dil verileri gerektirir. `tur`, `eng` ve `osd` dosyaları kurulu olmalıdır; OCR bağımlılığı eksikse sekme Türkçe bilgilendirme verir ve diğer sekmeler çalışmaya devam eder. Tesseract Python paketi `requirements.txt` içindedir, ancak Tesseract ikilisi ayrıca kurulmalıdır.

OCR sentetik deneyi `python scripts/run_ocr_experiments.py`, elle etiketli fotoğraf değerlendirmesi `python scripts/run_real_eval.py` ile yapılır. Gerçek fotoğraflar `data/test/<ad>.jpg`, köşeler `data/test/<ad>.corners.json` (`{"corners": [[x,y], ...]}`; TL, TR, BR, BL) ve isteğe bağlı metin `data/test/<ad>.txt` olarak sağlanır. Sonuçlar `results/ocr_experiments.csv` ve `results/real_eval.csv` dosyalarına yazılır.

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

Klasik kontur yaklaşımı düşük kontrast, ters polarite, kadraj dışı belge ve karmaşık sahnelerde başarısız olabilir; sonuçlar her görsel için garanti edilmez. UI'da referans görsel yoksa PSNR/SSIM üretilmez ve kalite karşılaştırması “referans yok” olarak belirtilir.

OCR kısıtları: Tesseract çıktısı dil verilerine, yazı tipine ve görüntü kalitesine bağlıdır; sentetik CER/WER gerçek fotoğraf performansı olarak yorumlanmamalıdır. Tesseract yoksa metin metrikleri hesaplanmaz. Laplacian MAD gürültü tahmini metin/kenar içeren görüntülerde gerçek sensör gürültüsünü yüksek tahmin edebilir. Yön düzeltmede önce OSD denenir; OSD çalışmazsa dört yön adayı OCR güveniyle sınanır, fakat kısa metin veya eksik OSD verisi belirsizlik bırakabilir. CER/WER yalnızca eşleşen `.txt` ground-truth dosyası bulunduğunda hesaplanır.

## Ölçüm kısıtları ve doğrulama notları

- Global(127), sentetik belgeyle aynı geometrik tariften bağımsız üretilen LINE_8 GT maskesinin tanımı değildir; bu örnekte iki foreground arasında 42 piksel fark ölçülmüştür. Sonuçlar `results/gt_mask_comparison.csv` içindedir.
- Sentetik benchmark sınırlı, kontrollü arka plan ve gölge dağılımı kullanır; yüksek tekrar başarısı gerçek fotoğraflara genellenebilirlik kanıtı değildir.
- Düz zeminli kontrollü gri-fark taramasında Canny başarı eşiği fixed için 60, Otsu türevi için 40, auto_canny için 80 gri seviye olarak gözlendi (her hücrede 5 seed; başarı ayrıca köşe oranı ve IoU ile tanımlanır). Gürültü sigma 10 sonuçları aynı eğri CSV'sinde ayrı satırlardır; bu sınırlar evrensel değildir.
- auto_canny açık/düz zeminde düşük kontrastlı kenarı yeterli eşik aşımına getirmeyebilir: kâğıt/masa 235/205 olan low_contrast_bg sahnesinde 0/5 başarı ölçülmüştür. Bu gözlem kontrollü sentetik sahneye özgüdür; otomatik başarı varsayılmaz.
- Clutter sahnesinde belge alanı 395,200 px, dış dikdörtgen alanı 501,600 px'tir. Canny adayları alan filtresinde elenir; threshold ölçümünde seçilen aday dış dikdörtgendir (CSV'de belge IoU ve clutter IoU ayrı). `make_scene`'in önceki sürümünden ölçüme etkili fark, perspektif warp dışındaki siyah dolgunun `BORDER_REPLICATE` ile değiştirilmesidir; rectangle koordinatı ve kalınlığı değişmemiştir.
- Güçlü perspektif F1 ölçümündeki önceki ~0.09 sonucu, corner ordering'in üstteki ilk noktayı TL saymasıyla genişlik/yüksekliğin yer değiştirmesiyle tutarlıydı: çıktı aspect oranı 1.3466 iken kaynak belge 0.75'ti. Köşe başlangıcı düzeltildikten sonra GT aspect 0.7426, tespit aspect yaklaşık 0.7405–0.7416; yeni homografik GT karşılaştırmasında Otsu F1 ortalaması 0.68 (3 seed) oldu. Bu sonuç ölçüm düzeltmesi sonrası sentetik sahneye aittir.
- AppTest doğrulaması `tests/test_streamlit_app.py` içindedir; yükleme/sekme görünümü, not_found uyarısı, bozuk yüklemede st.error ve Tesseract bulunamadığında OCR sekmesinin çökmeden bilgi vermesi kapsanır. Tam doğrulama komutu `pytest`'tir. Başsız başlatma: `streamlit run app/main.py --server.headless true`.
- `PipelineConfig.detection_fallback` varsayılan olarak kapalıdır. Açıldığında Canny başarısızlığından sonra threshold denenir; görüntü sınırına yaklaşan ya da alanı görüntünün %75'ini aşan yedek aday `ok` kabul edilmez. Bu güvenlik filtresi clutter/partial-crop ölçümlerine göre sınırlanmıştır; kadrajı çok dolduran geçerli belgelerde yedeği reddedebilir.
- Fallback ölçümünde low_contrast_bg 5/5 doğru belge, low_contrast_hard 0/5 kabul (threshold tüm görüntüyü aday gösterdi), clutter 0/5 kabul (threshold dış çerçeveyi seçti), partial_crop 5/5 reddetme verdi. Dört durumda da yanlış aday sessizce `ok` yapılmadı; satırlar `results/detection_fallback_trials.csv` içindedir.
- Pipeline matrisi frontal sentetik sahnede shadow strength `{0, 0.4, 0.7, 0.9}` × noise sigma `{0, 15, 25}` ve üç seed kullanır. Her koşulun `Ok_orani` değeri ile yalnız başarılı satırlardan hesaplanan precision/recall/F1 ve auto kararları `results/pipeline_experiments_summary.csv` içindedir; rectification SSIM her koşul/seed için tek kayıttır.
- Tespit doğrulamasında skor tabanı 0.45'ten 0.67'ye alınmıştır: `noisy_shadow` sigma 40 raw deneyinde önce 5 örneğin 3'ü hatalı köşelerle `ok` dönüyordu; bu üç skor 0.517, 0.652 ve 0.583'tü. Eşik sonrası bu örnekler `low_confidence`, iki doğru aday `ok` kalır. `clutter_highcontrast` için belge/masa farkı yaklaşık 30, büyük çerçeve/masa farkı 80 gri düzeydir; büyük alan adayları `low_confidence` olarak işaretlenir. Bu alan kapısı %75–95 aralığında büyük fakat doğru belgeleri de düşük güvene alabilir; ölçüm `results/fallback_area_sweep.csv` içindedir.
- Mevcut ortamda Tesseract ikilisi PATH'te bulunmadı; `results/ocr_experiments.csv` satırları `tesseract_kurulu_değil` olarak kaydedildi ve CER/WER hesaplanmadı. `data/test` içinde gerçek fotoğraf yok; gerçek değerlendirme CSV'si yalnız başlık içeriyor.

Deneyler `scripts/run_detection_experiments.py`, `scripts/run_detection_sensitivity.py` ve `scripts/run_pipeline_experiments.py` ile yeniden üretilebilir. CSV özetleri her koşul/seed'i ve başarısız tespitleri korur; pipeline F1 ortalamasına yalnızca `Durum=ok` ve düzeltilmiş satırlar katılır, başarı oranı ayrıca verilir.
