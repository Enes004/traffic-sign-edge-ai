# BİL458 · Nihai deney özeti

640 px ana deneyinde statik INT8, FP32'ye göre ONNX dosyasını %69,47 küçülttü. Raspberry Pi 5'te ortalama ONNX inference gecikmesi %6,05 arttı ve mAP50–95 1,65 yüzde puan düştü. Ayrı eğitilmiş 320 px modeller yaklaşık %77 daha düşük gecikme ve her iki formatta daha düşük mAP50–95 verdi.

## Deney koşulları

- GTSDB FullIJCNN2013, resmî test aralığı ve veri bölme seed'i 42. **378 train / 81 validation / 227 test** görüntüsü. Sınıflar `prohibitory`, `danger` ve `mandatory`. Testte 228 etiketli kutu vardır (143, 61 ve 24).
- YOLO26n, iki çözünürlükte ayrı eğitim, her çözünürlükte seed 0, 1 ve 2. Batch 16, en çok 100 epoch, patience 25, yatay çevirme kapalı. Koşu ayarları ve validasyon geçmişi [runs/train](../runs/train/) altında.
- Altı FP32 ONNX ve altı statik INT8 ONNX, sabit batch 1. Giriş `[1,3,H,W]`, çıktı 640 px için `[1,7,8400]`, 320 px için `[1,7,2100]`. INT8 QDQ, Conv katmanları, MinMax, kanal başına ağırlık kuantizasyonu, 100 train görüntüsü ve kalibrasyon seed'i 42. [Export](../src/export_onnx.py) · [Quantization](../src/quantize_int8.py).
- Pi 5 CPU, ONNX Runtime 1.23.2, NumPy 2.2.6, OpenCV 4.12.0, 4 thread, koşu başına 5 warm-up. Confidence > 0,001, sınıf bazlı NMS IoU 0,70, en çok 300 tespit. COCO tarzı 101 noktalı AP, mAP50–95 için IoU 0,50–0,95. [Benchmark](../src/benchmark_pi_v3.py).
- Her çözünürlükte her seed için FP32 → INT8 → INT8 → FP32. Toplam **24 koşu**, her koşuda **227 × 5 = 1.135** zamanlamalı inference, toplam **27.240**. 640 px 23 Kasım 2026'da, 320 px 24 Kasım 2026'da ölçüldü. [640 px ABBA logu](../results/abba_3seed_20261123_014946.log) · [320 px ABBA logu](../results/abba_320_20261124_223345.log).

## Nihai Pi 5 sonuçları

Değerler ortalama ± örneklem standart sapmasıdır. Her seed/format için iki ABBA koşusu önce aritmetik ortalamayla birleştirildi. Sonra üç seed üzerinden `n−1` ile standart sapma hesaplandı. mAP değerleri 0–1 ölçeğindedir. Inference süresi yalnız ONNX Runtime çalışmasını kapsar. FPS her koşuda `1000 / ortalama inference ms` olarak hesaplanır. Pipeline metrikleri preprocessing ve postprocessing'i de kapsar, JPEG decode hariçtir.

| Deney | Model | mAP50 | mAP50–95 | Inference (ms) | Inference FPS | Model (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 640 px ana deney | FP32 | 0,9598 ± 0,0116 | 0,7936 ± 0,0204 | 121,37 ± 0,25 | 8,24 ± 0,02 | 9,398 |
| 640 px ana deney | INT8 | 0,9560 ± 0,0071 | 0,7772 ± 0,0200 | 128,71 ± 0,93 | 7,77 ± 0,06 | 2,869 |
| 320 px ablasyon | FP32 | 0,7538 ± 0,0061 | 0,5082 ± 0,0109 | 27,63 ± 0,06 | 36,20 ± 0,08 | 9,278 |
| 320 px ablasyon | INT8 | 0,7410 ± 0,0054 | 0,4837 ± 0,0179 | 29,66 ± 0,06 | 33,71 ± 0,07 | 2,749 |

| Karşılaştırma | mAP50 farkı (yüzde puan) | mAP50–95 farkı (yüzde puan) | Gecikme değişimi | Model boyutu değişimi |
| --- | ---: | ---: | ---: | ---: |
| 640 px FP32 → INT8 | −0,39 | −1,65 | +%6,05 | −%69,47 |
| 320 px FP32 → INT8 | −1,28 | −2,45 | +%7,38 | −%70,37 |
| 640 → 320 px FP32 | −20,60 | −28,55 | −%77,24 | −%1,28 |
| 640 → 320 px INT8 | −21,50 | −29,35 | −%76,95 | −%4,19 |

Oran değişimlerinin paydası karşılaştırmanın referans ortalamasıdır. CPU frekansı log örneklerinde yaklaşık 2,4 GHz, kaydedilen en yüksek sıcaklık 640 px için 68,6 °C ve 320 px için 66,4 °C idi. Ölçülen `get_throttled` işaretlerinde 24 koşuda throttling kaydı yoktu. Çözünürlükler ayrı günlerde çalıştırıldı. Üç seed ile sonuçlar betimsel olarak sunulmaktadır. [Tam metrikler](../results/final/summary_metrics.csv) · [Seed tabloları](../results/final/abba_seed_results.csv) · [Karşılaştırmalar](../results/final/effects.csv) · [Sistem sağlığı](../results/final/system_health_summary.csv).

### Ham kayıt doğrulaması

Özetleyici iki logun `Saved:` satırlarından 24 koşuyu, her koşunun tek satırlı CSV/JSON çiftini, model boyutlarını ve log metriklerini doğruladı. Ayrı bir NumPy hesabı 14 metriğin dört konfigürasyondaki ortalama ve `ddof=1` sapmalarını (112 hücre), 20 metrik karşılaştırmasını ve FPS/pipeline formüllerini doğruladı. İki tekrarın mAP50 ve mAP50–95 değerleri 12 seed/format hücresinin tamamında aynıydı. Eski benchmark ve smoke çıktıları bu envanterin dışındadır. Ayrı inference zaman örnekleri ham dosyalarda yer almadığı için koşu içi medyan ve p95, kaydedilmiş koşu özetlerinden alınmıştır. [Koşu envanteri ve doğrulama](../results/final/final_summary.json) · [Özetleyici](../src/summarize_benchmark.py).

## 640 px nesne boyutuna göre recall

Kutular özgün görüntüdeki alan tertillerine göre gruplandırıldı. Değerler üç seed ortalamasıdır. Aynı sınıfta confidence ≥ 0,25 ve eşleşme IoU ≥ 0,50 kullanıldı.

| Grup | GT kutusu | FP32 recall | INT8 recall | INT8 − FP32 (yüzde puan) |
| --- | ---: | ---: | ---: | ---: |
| Small | 77 | %89,61 | %88,74 | −0,87 |
| Medium | 75 | %97,78 | %97,78 | 0,00 |
| Large | 76 | %100,00 | %99,56 | −0,44 |

Her iki formatta recall küçük tertilde daha düşüktü. Eşleşmiş seed farkının small tertilde örneklem SD'si 3,00 yüzde puandır. 320 px için boyuta göre recall verisi bulunmuyor. [Recall tablosu](../results/final/small_object_analysis.csv) · [Ham analiz](../results/object_size_recall_full_20261123_011755.json).

## Hata analizi ve demo

Hata analizi yerel CPU'da 227 görüntü ve dört seed 0 modeli kullandı. Confidence > 0,25, sınıf bazlı NMS IoU 0,70, en çok 300 kutu. Tahminler azalan confidence sırasıyla eşleştirildi. İlk turda aynı sınıf ile IoU ≥ 0,50 eşleşmeleri alındı. Kalan kutularda farklı sınıf ve IoU ≥ 0,50 sınıflandırma hatası, aynı sınıf ve 0,10 ≤ IoU < 0,50 lokalizasyon hatası olarak eşleştirildi. Son turda eşleşmemiş tahminler FP ve eşleşmemiş GT kutuları FN olarak kaydedildi. Bu tanısal kategoriler, benchmark AP hesabından bağımsızdır. Seçim her kategori için farklı bir görüntü verir ve eşleşen hatalarda en yüksek IoU'yu, FP'de en yüksek confidence'ı, FN'de en büyük GT kutusunu önceler.

| Görüntü | Hata | Gözlem |
| --- | --- | --- |
| [00778](../results/failure_analysis/00778_classification_error.png) | Sınıflandırma | 640 FP32, mandatory kutusunu prohibitory diye etiketledi (confidence 0,527, IoU 0,828). Diğer üç modelde bu eşik üzerinde kutu yoktu. |
| [00886](../results/failure_analysis/00886_localization_error.png) | Lokalizasyon | Üstteki prohibitory kutusunda 320 INT8 IoU 0,471 ve 320 FP32 IoU 0,438 verdi. 640 modelleri aynı kutuda yaklaşık 0,79 IoU verdi. |
| [00659](../results/failure_analysis/00659_false_positive.png) | Yanlış pozitif | 320 FP32, görüntünün solunda GT ile eşleşmeyen prohibitory kutusu çizdi (confidence 0,837). Sağdaki GT için ayrı bir tahmin vardı. |
| [00780](../results/failure_analysis/00780_false_negative.png) | Yanlış negatif | 320 INT8, soldaki prohibitory GT kutusunu kaçırdı. Diğer üç model aynı GT ile eşleşti. |

Her figürde GT ve dört model aynı sahnede karşılaştırılır. Kutuların özgün piksel koordinatları, sınıf, confidence, IoU, eşikler ve model SHA-256 değerleri [examples.json](../results/failure_analysis/examples.json) içinde tutulur.

[30 saniyelik MP4](../results/demo/yolo26n_img640_seed0_fp32.mp4), 640 px FP32 seed 0 ile yapılan yerel CPU inference'ını gösterir. Görüntüler 00600, 00605, 00614, 00621, 00626, 00633, 00638, 00647, 00657 ve 00662. Her biri üç saniye gösterilir. [Demo üreticisi](../src/demo_inference.py).

## Rapor ve sunum görselleri

| İçerik | Vektör PDF | PNG |
| --- | --- | --- |
| Doğruluk | [PDF](../results/final/fig_accuracy_comparison.pdf) | [PNG](../results/final/fig_accuracy_comparison.png) |
| Gecikme ve FPS | [PDF](../results/final/fig_latency_fps.pdf) | [PNG](../results/final/fig_latency_fps.png) |
| Kuantizasyon etkisi | [PDF](../results/final/fig_quantization_effect.pdf) | [PNG](../results/final/fig_quantization_effect.png) |
| Çözünürlük etkisi | [PDF](../results/final/fig_resolution_effect.pdf) | [PNG](../results/final/fig_resolution_effect.png) |
| Nesne boyutuna göre recall | [PDF](../results/final/fig_small_object_recall.pdf) | [PNG](../results/final/fig_small_object_recall.png) |
| Sistem sağlığı | [PDF](../results/final/fig_system_health.pdf) | [PNG](../results/final/fig_system_health.png) |

Teslim doğrulamasında 123 pytest testi geçti. [Notebook](../notebooks/traffic_sign_edge_ai.ipynb) temiz Jupyter çekirdeğiyle tüm hücreleri çalıştırdı. Altı PDF rasterleştirildi, on PNG (altı nihai grafik, dört hata figürü) ve 720 video karesi çözümlendi.
