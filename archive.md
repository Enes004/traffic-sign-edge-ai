# Proje Deney ve Metodoloji Arşivi

Bu belge, **BİL458 Trafik İşareti Tespiti ve Edge-AI Quantization (Raspberry Pi 5)** projesi boyunca gerçekleştirilen tüm teknik aşamaları, nicel deney sonuçlarını ve alınan kararların matematiksel ile derin öğrenme temelli gerekçelerini tek bir merkezde toplamak amacıyla oluşturulmuş **canlı (yaşayan) proje arşividir**. Nihai rapor ve sunum hazırlanırken ana kaynak olarak kullanılacak, yeni deneyler tamamlandıkça ilgili bölümler güncellenecektir.

## 1. Proje Kapsamı, Donanım ve Yazılım Mimarisi

* **Araştırma Odağı:** GTSDB (German Traffic Sign Detection Benchmark) veri seti üzerinde 3 sınıflı trafik işareti tespiti yapan hafif bir nesne tespit modelinin (`YOLO26n`, $640 \times 640$) Raspberry Pi 5 CPU üzerinde **FP32** ve **INT8** hassasiyetleri arasındaki doğruluk ($\text{mAP}$, küçük levha $\text{Recall}$) ve hız ($\text{ms/görüntü}$) ödünleşiminin 3 bağımsız tohum (seed) üzerinden ölçülmesi.

* **Eğitim ve Dönüşüm Ortamı (Laptop):**
  * **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU ($7708\text{ MiB}$ VRAM)
  * **CPU:** Intel Core i7-14650HX
  * **Yazılım Yığını:** Python `3.10.12`, PyTorch `2.9.1+cu128` (CUDA 12.8), Ultralytics (`8.3.241` sürümünden YOLO26 için `8.4.215` sürümüne yükseltildi), ONNX `1.23.2`, `onnxslim 0.1.99`, ONNX Runtime `1.23.2`, SciPy.

* **Hedef Dağıtım Ortamı (Edge):** Raspberry Pi 5 (ARM64 CPU, ONNX Runtime).

### Neden TensorFlow Yerine PyTorch + ONNX Runtime Seçildi?

1. **Yerel (Native) Eğitim ve Donanım Uyumluluğu:** Ultralytics YOLO mimarilerinin geliştirme, eğitim ve ağırlık optimizasyon çekirdeği doğrudan PyTorch üzerine kuruludur. RTX 5060 GPU'nun güncel CUDA 12.8 mimarisi PyTorch `2.9.1+cu128` ile sorunsuz ve kararlı çalışmaktadır.

2. **Kısa ve Kayıpsız Dönüşüm Zinciri:** TensorFlow/TFLite yolunda izlenen $\text{PyTorch} \rightarrow \text{ONNX} \rightarrow \text{SavedModel} \rightarrow \text{TFLite}$ çok aşamalı dönüşüm hattı; tensör düzeni (`NCHW` $\leftrightarrow$ `NHWC`) değişiklikleri, desteklenmeyen operatörler ve sayısal sapma riskleri taşır.

3. **ARM64 CPU Üzerinde Deterministik INT8 Desteği:** $\text{PyTorch} \rightarrow \text{ONNX}$ (`onnxslim` ile sadeleştirilmiş) $\rightarrow \text{ONNX Runtime}$ hattı tek adımda doğrulanabilir grafik üretir ve Raspberry Pi 5 CPU üzerinde statik QDQ (Quantize–Dequantize) INT8 çalıştırmasını doğrudan destekler.

## 2. Veri Seti Hazırlığı (GTSDB $\rightarrow$ YOLO Formatı)

Ham veri seti, resmi `FullIJCNN2013.zip` arşivinden ($1{,}661{,}866{,}983$ bayt) `setup.sh` ve `src/prepare_dataset.py` aracılığıyla salt-okunur biçimde işlenerek `data/gtsdb/` altına dönüştürülmüştür.

### 2.1. Koordinat Normalizasyonu ve Matematiksel Formülasyon

Ham GTSDB etiketleri (`gt.txt`), $W \times H$ boyutlu ($1360 \times 800$ piksel) görüntüler üzerinde sol-üst ve sağ-alt piksel koordinatları $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ olarak verilmiştir. YOLO formatı için sürekli koordinat konvansiyonu ($w_{\text{px}} = x_{\max} - x_{\min}$) kullanılarak normalize merkez ve boyut değerleri hesaplanmıştır:

$$
x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2W}, \quad y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2H}, \quad w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}
$$

* **Derin Öğrenme Gerekçesi:** Koordinatların $[0, 1]$ aralığına normalize edilmesi, sınırlayıcı kutu regresyonunu (bounding box regression) giriş çözünürlüğünden bağımsız (scale-invariant) hale getirir. Böylece ağ farklı boyutlandırma (örneğin $640 \times 640$) adımlarında aynı kayıp ölçeğinde kararlı gradyanlar üretir. Tüm değerler 6 ondalık basamak hassasiyetiyle kaydedilmiş, PPM görüntüleri kalite `95` ve `4:4:4` renk alt-örneklemesi ile JPEG'e çevrilmiştir.

### 2.2. Sınıf Sıkıştırma (43 Kaynak Sınıf $\rightarrow$ 3 Resmi Tespit Kategorisi)

Resmi GTSDB tespit protokolüne (`ReadMe.txt` ve `TSD_readGTData.m`) birebir sadık kalınarak 43 alt sınıf 3 üst kategoriye eşlenmiştir:

| YOLO ID | Sınıf Adı | Görsel Karakteristik | Kaynak GTSDB Sınıf ID'leri |
| :--- | :--- | :--- | :--- |
| **`0`** | `prohibitory` (Yasaklayıcı) | Kırmızı yuvarlak (hız sınırı, sollama yasağı, giriş yok vb.) | `0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16` |
| **`1`** | `danger` (Tehlike / Uyarı) | Kırmızı üçgen (kavşak, viraj, kasis, yaya geçidi, buzlanma vb.) | `11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31` |
| **`2`** | `mandatory` (Zorunlu Yön) | Mavi yuvarlak (sağa/sola/ileri mecburi yön, ada etrafı vb.) | `33, 34, 35, 36, 37, 38, 39, 40` |
| **Hariç** | `other` (Kapsam Dışı) | Ana yol (sarı eşkenar dörtgen), yol ver, DUR, yasak sonu vb. | `6, 12, 13, 14, 17, 32, 41, 42` |

* **Dışlama Politikası (Sınıf Kirliliğini Önleme):** Bir görüntüde `other` kümesine ait *herhangi bir* etiket bulunuyorsa, yalnızca etiket satırı silinmek yerine **görüntünün tamamı veri setinden çıkarılmıştır**.
  * *Derin Öğrenme Gerekçesi:* Eğer `other` etiketi silinip görüntü eğitim setinde bırakılsaydı, karede fiziksel olarak yer alan gerçek bir trafik levhası ağa "arka plan (background)" olarak öğretilmiş olacaktı. Bu durum sahte negatif (false negative) gradyanlar üreterek tespit kafasının karar sınırlarını bozar.

### 2.3. Veri Temizliği, Negatif Örnekler ve Resmi Ayrım (`--split-mode official`)

* **Tekrarlanan Etiket Temizliği:** Ham `gt.txt` dosyasında $1{,}213$ satır ve $1{,}212$ benzersiz etiket tespit edilmiştir. Birebir tekrar eden `00340.ppm` (sınıf 14) kaydı açık uyarı ile elenmiştir.

* **Etiketsiz Negatif Görüntüler:** Hiç levha içermeyen $159$ görüntü, boş `.txt` etiket dosyalarıyla veri setine dahil edilmiştir. Bu görüntüler ağın karmaşık yol sahnelerinde arka planı levhadan ayırt etmesini sağlayarak yanlış pozitif (False Positive) oranını düşürür.

* **Veri Sızıntısını (Data Leakage) Önleme:** GTSDB video karelerinden türetildiği ve sekans ID'si içermediği için rastgele bölme (`--split-mode random`), zamansal olarak ardışık ve görsel açıdan neredeyse özdeş kareleri hem eğitim hem test kümesine düşürerek yapay yüksek skorlara yol açar. Bu nedenle resmi test aralığı (`00600`–`00899`) SHA-256 doğrulaması ile `TestIJCNN2013.zip` üzerinden teyit edilerek `--split-mode official` ile kesin biçimde ayrılmıştır.

| Küme (Split) | Görüntü Sayısı | Toplam Levha | `prohibitory` | `danger` | `mandatory` | Kullanım Politikası |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 378 | — | — | — | — | Model eğitimi ve INT8 kalibrasyonu |
| **Validation** | 81 | 104 | 68 | 25 | 11 | Erken durdurma, en iyi ağırlık seçimi, ONNX parity ve ön-kontrol |
| **Test (Held-out)** | 227 | 228 | 143 | 61 | 24 | **Dokunulmamış** — yalnızca Raspberry Pi 5 final ölçümü |

Etiket dönüşümünün doğruluğu `src/visualize_labels.py` ile deterministik olarak seçilen 12 örnek üzerinde görsel olarak teyit edilmiştir (`results/dataset_validation.jpg`).

## 3. Eğitim Metodolojisi ve Hiperparametre Gerekçeleri

### 3.1. Temel Değerlendirme Metriklerinin Matematiksel Tanımları

Nesne tespit başarımı; tahmin edilen kutu ($B_p$) ile gerçek kutu ($B_{gt}$) arasındaki Kesişim/Birleşim oranı (IoU) temel alınarak ölçülmüştür:

$$
\text{IoU} = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}, \quad P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}
$$

$$
\text{mAP}_{50} = \frac{1}{C}\sum_{c=1}^{C}\text{AP}_{c}(\text{IoU}=0.50), \quad \text{mAP}_{50\text{-}95} = \frac{1}{10}\sum_{t \in \{0.50, 0.55, \dots, 0.95\}} \text{mAP}(t)
$$

Çoklu seed deneylerinin özetinde $N = 3$ tohum için aritmetik ortalama ($\mu$) ve popülasyon standart sapması ($\sigma$) raporlanmıştır:

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}
$$

### 3.2. Seçilen Hiperparametreler ve Teknik Nedenleri

* **Yatay Aynalama Kapatıldı (`fliplr=0.0`):** Trafik işaretleri yön ve rakam duyarlıdır (örneğin mecburi sağa/sola yön okları ve hız sınırı rakamları). Görüntüyü yatay çevirmek semantik anlamı bozar ve modelin ters semboller üzerinden hatalı özellikler öğrenmesine neden olur.

* **Çoklu Seed (`0, 1, 2`) ve Determinizm (`deterministic=True`):** Orijinal GTSDB makalesi (Houben vd., 2013) klasik yöntemler (Viola-Jones, HOG+LDA, Hough Transform) kullandığı için modern derin öğrenme seed'i paylaşmamıştır; literatür karşılaştırması sabit resmi test kümesi üzerinden yapılır. $378$ görüntülük küçük eğitim setinde ağırlık başlatma ve mini-batch sıralaması kaynaklı varyansı kontrol altına almak için 3 standart tohum (`0, 1, 2`) kullanılmıştır.

* **Batch Boyutu (`batch=16`) ve VRAM Kullanımı ($\approx 2.8\text{ GB} / 7.7\text{ GB}$):**
  * PyTorch ekran kartı belleğinin tamamını otomatik doldurmaz; yalnızca aktif model ağırlıkları, aktivasyon haritaları ve gradyanlar için gereken alanı ayırır. `YOLO26n` ($2.39\text{M}$ parametre) çok hafif bir mimari olduğundan `batch=16` ve $640 \times 640$ girişte $\approx 2.8\text{ GB}$ VRAM yeterlidir.
  * $378$ eğitim görüntüsünde `batch=16`, her epoch'ta $\lceil 378 / 16 \rceil = 24$ ağırlık güncellemesi (iterasyon) sağlar. `batch=64` seçilseydi epoch başına yalnızca $6$ güncelleme kalacak, gradyan gürültüsü azalacağı için genelleme performansı düşecekti. Ayrıca seed'ler arası bilimsel karşılaştırılabilirlik için `batch=16` sabitlenmiştir.

* **Erken Durdurma (`epochs=100`, `patience=25`):** Doğrulama başarımı 25 epoch boyunca iyileşmediğinde eğitimi durdurarak küçük veri setinde aşırı öğrenmeyi (overfitting) önler ve en yüksek doğrulama skorlu kontrol noktasını (`best.pt`) saklar.

* **Çözünürlük Kapsamı ($640\text{ px}$ Ana Deney + $320\text{ px}$ Çözünürlük Ablasyonu):** Ana **FP32 vs. INT8** karşılaştırması $640 \times 640$ çözünürlükte 3 seed çiftiyle yürütülmüş; ders yönergesindeki zorunlu ablasyon şartı (Hipotez H2 / E3) kapsamında aynı `official` veri ayrımı ve birebir aynı hiperparametrelerle $320 \times 320$ çözünürlük ablasyonu (3 seed × FP32/INT8, toplam 6 ek ONNX modeli) ayrıca eğitilip doğrulanmıştır (bkz. Bölüm 8).

## 4. Mimari Geçiş Analizi: YOLO11n vs. YOLO26n

### 4.1. İlk Deney: YOLO11n (`seed=0`, FP32, `640px`)

Projenin ilk aşamasında `Ultralytics 8.3.241` ile `YOLO11n` üzerinde `seed=0` eğitimi gerçekleştirilmiştir (`patience=25`, en iyi epoch: `50`, toplam: `75` epoch, süre: `0.149` saat, `best.pt`: `5.5 MB`).

* **Model Karmaşıklığı (Fused):** $100$ katman, $2{,}582{,}737$ parametre, $6.3\text{ GFLOPs}$.
* **GPU Çıkarım Hızı (RTX 5060):** `1.8 ms` preprocess, `16.0 ms` inference, `0.6 ms` postprocess.

| Sınıf (`YOLO11n`, Seed 0 — Val) | Görüntü | Levha | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all`** | **81** | **104** | **0.974** | **0.908** | **0.972** | **0.828** |
| `prohibitory` | 43 | 68 | 0.953 | 0.902 | 0.965 | 0.807 |
| `danger` | 20 | 25 | 0.969 | 0.920 | 0.980 | 0.860 |
| `mandatory` | 10 | 11 | 1.000 | 0.902 | 0.971 | 0.816 |

### 4.2. YOLO26n'e Geçiş ve `seed=0` Performans Farkının Teknik Analizi

Ekip kararıyla daha güncel ve daha düşük hesaplama maliyetli olan `YOLO26n` mimarisine geçilmiş, `ultralytics` (`8.4.215`) ve grafik çiziminde hata veren `scipy` güncellenmiş, eski `YOLO11n` çıktıları tamamen temizlenerek baştan eğitim yapılmıştır.

| Özellik (`seed=0`, Val) | YOLO11n | YOLO26n | Göreli Değişim ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Katman Sayısı (Fused)** | 100 | 121 | $+21$ katman |
| **Parametre Sayısı** | $2{,}582{,}737$ | $2{,}390{,}221$ | $-7.45\%$ |
| **Hesaplama Maliyeti (GFLOPs)** | $6.3$ | $5.4$ | $-14.29\%$ |
| **Eğitim Süresi (`seed=0`)** | $0.149\text{ saat}$ (75 ep.) | $0.042\text{ saat}$ (65 ep.) | $-71.8\%$ |
| **Val $\text{mAP}_{50}$** | $0.972$ | $0.950$ ($0.9498$) | $-0.022$ |
| **Val $\text{mAP}_{50\text{-}95}$** | $0.828$ | $0.773$ ($0.7727$) | $-0.055$ |

**YOLO26n'in `seed=0` Skorunun YOLO11n'den Biraz Daha Düşük Çıkmasının 3 Teknik Nedeni:**

1. **Kapasite–Hız Ödünleşimi:** YOLO26n yaklaşık $\%7.45$ daha az parametre ve $\%14.29$ daha düşük FLOPs ($5.4$ vs. $6.3\text{ GFLOPs}$) kullanır. Edge CPU gecikmesini azaltmaya yönelik bu sıkıştırma, özellikle sıkı IoU eşiklerinde ($\text{mAP}_{50\text{-}95}$) milimetrik kutu oturtma kapasitesinde hafif bir ödünleşim doğurur.

2. **Küçük Doğrulama Kümesinde Ayrık Örnek Hassasiyeti:** Doğrulama setindeki `mandatory` sınıfında yalnızca $11$ levha bulunmaktadır. Bu sınıfta Recall'ın $0.902$'den $0.788$'e inmesi ($\Delta R = -0.114$), modelin yalnızca **1 veya 2 levhayı** güven eşiğinin hemen altında kaçırmasından kaynaklanır.

3. **Daha Erken Yakınsama (Plato):** YOLO26n en iyi ağırlığına $40.$ epoch'ta ulaşıp $65.$ epoch'ta erken durmuştur (YOLO11n: $50.$ / $75.$ epoch). Küçük veri setlerinde varsayılan hiperparametrelerle daha hafif mimariler daha erken yerel platoya oturabilmektedir.

Buna rağmen `YOLO26n`, $0.9408\text{ mAP}_{50}$ ortalama başarımı ile **FP32 vs. INT8** karşılaştırması için güçlü ve kararlı bir temel (baseline) sunmaktadır.

## 5. YOLO26n 3-Seed FP32 Baseline Eğitim Sonuçları (Validation)

Tüm eğitimler `python3 src/train.py --imgsz 640 --seed {0,1,2}` komutuyla aynı donanım ve deterministik ayarlarda gerçekleştirilmiştir. Her modelin en iyi kontrol noktası (`best.pt`, `5.4 MB` optimizer stripped) `models/yolo26n_img640_seed{0,1,2}.pt` yoluna kaydedilmiştir.

### 5.1. Genel Doğrulama Özeti (`val` — 81 Görüntü, 104 Levha)

| Seed | En İyi Epoch | Toplam Epoch | Süre (saat) | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ | GPU Inf. Hızı |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 40 | 65 (EarlyStop) | 0.042 | 0.9615 | 0.8678 | 0.9498 | 0.7727 | 33.6 ms |
| **Seed 1** | 100 | 100 | 0.063 | 0.9171 | 0.8788 | 0.9270 | 0.7755 | 34.9 ms |
| **Seed 2** | 51 | 76 (EarlyStop) | 0.048 | 0.9287 | 0.9415 | 0.9456 | 0.7818 | 34.0 ms |
| **Ortalama $\pm$ Std** | — | — | — | $0.9358 \pm 0.0188$ | $0.8960 \pm 0.0325$ | $0.9408 \pm 0.0099$ | $0.7767 \pm 0.0038$ | — |

### 5.2. Sınıf Bazlı Doğrulama Dağılımı (3 Seed)

| Seed | Sınıf | Görüntü | Levha | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | `prohibitory` | 43 | 68 | 0.953 | 0.895 | 0.954 | 0.779 |
| **Seed 0** | `danger` | 20 | 25 | 0.931 | 0.920 | 0.945 | 0.800 |
| **Seed 0** | `mandatory` | 10 | 11 | 1.000 | 0.788 | 0.950 | 0.738 |
| **Seed 1** | `prohibitory` | 43 | 68 | 0.930 | 0.926 | 0.964 | 0.813 |
| **Seed 1** | `danger` | 20 | 25 | 0.925 | 0.920 | 0.951 | 0.810 |
| **Seed 1** | `mandatory` | 10 | 11 | 0.897 | 0.790 | 0.866 | 0.704 |
| **Seed 2** | `prohibitory` | 43 | 68 | 0.942 | 0.956 | 0.972 | 0.796 |
| **Seed 2** | `danger` | 20 | 25 | 0.966 | 0.960 | 0.968 | 0.799 |
| **Seed 2** | `mandatory` | 10 | 11 | 0.879 | 0.909 | 0.897 | 0.750 |

**İstatistiksel Gözlemler:**

* **Yüksek İstikrar:** $\text{mAP}_{50\text{-}95}$ metriğindeki standart sapmanın yalnızca $\sigma = 0.0038$ olması, modelin farklı rastgele başlangıçlara rağmen oldukça kararlı bir lokalizasyon kalitesine yakınsadığını gösterir.
* **Azınlık Sınıf Varyansı:** Yalnızca $11$ örneğe sahip `mandatory` sınıfı en yüksek dalgalanmayı ($\text{mAP}_{50}$ aralığı $0.866\text{--}0.950$) sergilerken, örnek sayısı daha yüksek olan `prohibitory` ($0.954\text{--}0.972$) ve `danger` ($0.945\text{--}0.968$) sınıfları son derece tutarlıdır.
* **Veri Sızıntısı Kontrolü:** Yukarıdaki tüm ölçümler yalnızca `val` kümesine aittir; $227$ görüntülük `test` kümesi eğitim ve doğrulama aşamalarında hiç kullanılmamıştır.

## 6. ONNX Dönüşümü (FP32) ve Eşitlik Doğrulaması (Parity Check)

### 6.1. Dışa Aktarım (Export) Parametreleri ve Tensör Yapısı

`src/export_onnx.py` ile eğitilmiş PyTorch kontrol noktaları statik FP32 ONNX formatına dönüştürülmektedir (`opset=17`, `simplify=True` / `onnxslim 0.1.99`, `dynamic=False`, `half=False`).

* **Giriş Tensörü:** $(1, 3, 640, 640)$ (`BCHW`, FP32).
* **Çıkış Tensörü:** $(1, 7, 8400)$
  * **7 Kanalın Anlamı:** $4$ kutu koordinatı $(x, y, w, h) + 3$ sınıf olasılığı (`prohibitory`, `danger`, `mandatory`).
  * **8400 Aday Hücrenin Matematiği:** Çok ölçekli özellik piramidinin (P3/8, P4/16, P5/32) toplam uzamsal ızgara hücre sayısıdır:

    $$
    \left(\frac{640}{8}\right)^2 + \left(\frac{640}{16}\right)^2 + \left(\frac{640}{32}\right)^2 = 80^2 + 40^2 + 20^2 = 6400 + 1600 + 400 = 8400
    $$

* **Neden `opset=17` ve `simplify=True`?** `opset=17`, ARM64 ONNX Runtime ve statik QDQ quantization araçlarıyla en yüksek kararlılığı sunar. `onnxslim` ise sabit katlama (constant folding) uygulayarak gereksiz ara düğümleri siler; hem çıkarım grafiğini sadeleştirir hem de INT8 QDQ düğümlerinin temiz yerleşmesini sağlar.

### 6.2. Dosya Boyutunun `5.2 MB`'tan `9.40 MB`'a Çıkmasının Matematiksel Nedeni

Seed 0 ve Seed 1 için disk boyutları:
* `yolo26n_img640_seed{0,1}.pt`: `5.2 MB`
* `yolo26n_img640_seed{0,1}_fp32.onnx`: `9.40 MB`

Ultralytics, eğitilmiş `.pt` ağırlıklarını diskten tasarruf amacıyla yarı hassasiyetli (**FP16**, parametre başına $2\text{ bayt}$) olarak kaydeder; FP32 ONNX ise ağırlıkları tam hassasiyetli (**FP32**, parametre başına $4\text{ bayt}$) açık grafik formatında saklar:

$$
\text{FP16 Teorik Ağırlık} \approx 2{,}390{,}221 \times 2\text{ B} \approx 4.78\text{ MB} \quad (\text{+ üst veri} \approx 5.2\text{ MB})
$$

$$
\text{FP32 Teorik Ağırlık} \approx 2{,}390{,}221 \times 4\text{ B} \approx 9.56\text{ MB} \quad (\approx 9.12\text{ MiB + grafik} = 9.40\text{ MB})
$$

### 6.3. Dönüşüm Doğrulaması (Parity Check) ve `rect` Ön-İşleme Bulgusu

`src/verify_onnx.py` ile orijinal `.pt` ve dönüştürülmüş `fp32.onnx` modelleri aynı `val` kümesi üzerinde CPU'da karşılaştırılmıştır.

#### Aşama 1: Varsayılan Ayarlar (Seed 0 — `rect=True` PyTorch vs. Statik Kare ONNX)

| Metrik (Seed 0 — Val) | PyTorch (`.pt`) | ONNX (`FP32`) | Fark ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.9615 | 0.9626 | $+0.0011$ |
| **Recall** | 0.8679 | 0.9059 | $+0.0380$ |
| $\text{mAP}_{50}$ | 0.9498 | 0.9626 | $+0.0128$ |
| $\text{mAP}_{50\text{-}95}$ | 0.7805 | 0.7865 | $+0.0060$ |
| **Durum** | — | — | `CHECK` ($\max \Delta\text{mAP} = 0.0128$) |

* **Farkın Nedeni (ONNX Modeli İyileştirmedi):** GTSDB görüntüleri $1360 \times 800$ ($1.7:1$ en-boy oranı) boyutundadır. Ultralytics `.pt` doğrulamasında varsayılan olarak dikdörtgen kırpma (`rect=True`, yaklaşık $640 \times 384$) uygularken, statik $(1, 3, 640, 640)$ ONNX modeli zorunlu olarak tam kare padding (`rect=False`, $640 \times 640$) kullanır. Farklı padding sınırları ve küçük doğrulama kümesi (104 levha), eşik sınırındaki 1–2 levhanın güven skorunu hafifçe yukarı çekmiştir.

#### Aşama 2: Elma-Elma Protokolü (`batch=1, rect=False`) — Nihai Doğrulama

Hem `.pt` hem `fp32.onnx` modellerine birebir aynı kare ön-işleme (`batch=1, rect=False`) uygulanarak tekrar ölçülmüştür.

**Seed 0 Parity (Laptop CPU: `0.7 ms` preprocess, `32.1 ms` inference, `0.6 ms` postprocess):**

| Metrik (Seed 0 — Val) | PyTorch (`.pt`) | ONNX (`FP32`) | Fark ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.9626 | 0.9626 | $+0.0000$ |
| **Recall** | 0.9059 | 0.9059 | $-0.0000$ |
| $\text{mAP}_{50}$ | 0.9626 | 0.9626 | $+0.0000$ |
| $\text{mAP}_{50\text{-}95}$ | 0.7865 | 0.7865 | $+0.0000$ |
| **Durum** | — | — | **`PASS` ($\max \Delta\text{mAP} = 0.0000$)** |

**Seed 1 Parity (Laptop CPU: `.pt` `16.7 ms` inf. / ONNX FP32 `18.1 ms` inf.):**

| Metrik (Seed 1 — Val) | PyTorch (`.pt`) | ONNX (`FP32`) | Fark ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.9174 | 0.9174 | $-0.0000$ |
| **Recall** | 0.8538 | 0.8538 | $+0.0000$ |
| $\text{mAP}_{50}$ | 0.9015 | 0.9015 | $+0.0000$ |
| $\text{mAP}_{50\text{-}95}$ | 0.7555 | 0.7555 | $+0.0000$ |
| **Durum** | — | — | **`PASS` ($\max \Delta\text{mAP} = 0.0000$)** |

* **Metodolojik Sonuç:** Üç seed için `.pt` ve FP32 ONNX doğrulama metrikleri, `640 × 640` ve `batch=1` altında raporlanan dört ondalık basamağa kadar eşleşti. Raspberry Pi 5 final ölçümü de bu giriş ayarlarını kullanacak.

#### Seed Bazlı ONNX Parity Özeti (`batch=1, rect=False`)

| Seed | FP32 ONNX Boyutu | `.pt` $\text{mAP}_{50}$ | ONNX $\text{mAP}_{50}$ | `.pt` $\text{mAP}_{50\text{-}95}$ | ONNX $\text{mAP}_{50\text{-}95}$ | Durum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 9.40 MB | 0.9626 | 0.9626 | 0.7865 | 0.7865 | `PASS` ($\Delta = 0.0000$) |
| **Seed 1** | 9.40 MB | 0.9015 | 0.9015 | 0.7555 | 0.7555 | `PASS` ($\Delta = 0.0000$) |
| **Seed 2** | 9.40 MiB | 0.9354 | 0.9354 | 0.7757 | 0.7757 | `PASS` ($\Delta = 0.0000$) |

## 7. Statik INT8 Quantization (PTQ Calibration)

### 7.1. Matematiksel Temel ve PTQ Mekanizması

Bu projede uygulanan yöntem **PTQ (Post-Training Static Quantization — Eğitim Sonrası Statik Kuantizasyon)** olup **yeniden eğitim (fine-tuning) değildir**. Bu aşamada geri yayılım (backpropagation), kayıp (loss) hesabı, etiket (label) okuma veya ağırlık güncellemesi gerçekleştirilmez.

Modelin FP32 ($32$-bit ondalıklı, $4\text{ bayt}$) ağırlık ve aktivasyon değerleri, yalnızca $256$ ayrık seviye alabilen INT8 ($8$-bit tam sayı, $1\text{ bayt}$) aralığına doğrusal (affine) dönüşüm ile haritalanır:

* **Ağırlıklar (`QInt8`, per-channel):** $[-128, 127]$ aralığına,
* **Aktivasyonlar (`QUInt8`):** $[0, 255]$ aralığına sıkıştırılır.

Ölçek ($S$, scale) ve sıfır noktası ($Z$, zero-point) parametreleriyle kuantizasyon ve dekuantizasyon denklemleri şöyledir:

$$
q = \text{clip}\left(\text{round}\left(\frac{r}{S}\right) + Z,\; q_{\min},\; q_{\max}\right), \qquad \hat{r} = S \cdot (q - Z)
$$

Burada $r$ orijinal FP32 değeri, $q$ tam sayı (INT8) karşılığı, $\hat{r}$ ise yaklaşık geri kazanılmış değerdir. Yuvarlama ($\text{round}$) kaynaklı $|\hat{r} - r|$ sapması **kuantizasyon hatasını** oluşturur.

### 7.2. Kalibrasyonun Rolü ve QDQ Grafik Yapısı

* **100 Eğitim Görüntüsüyle Kalibrasyon (`train` split, `calib_seed=42`):** Ağırlıklar sabit olduğu için aralıkları doğrudan bilinir; ancak ara katman aktivasyonları girdi görüntüsüne bağlıdır. `train` kümesinden deterministik olarak seçilen $100$ görüntü yalnızca **ileri besleme (forward pass)** ile ağdan geçirilir. `CalibrationMethod.MinMax` yöntemi her katmandaki minimum ve maksimum aktivasyon sınırlarını gözlemleyerek sabit $(S, Z)$ parametrelerini hesaplar ve ONNX grafiğine gömer. Test ve doğrulama verisi kalibrasyona hiç dahil edilmez (veri sızıntısı önlenir).

* **`Conv` İşlemlerinin Kuantize Edilmesi (`op_types_to_quantize=["Conv"]`):** Üç INT8 ONNX dosyasının her birinde `Conv` işlemlerine bağlı 102 bias `INT32` olarak saklanır. Dosyalar ayrıca dönüşüm ölçeklerini ve kuantizasyon kapsamı dışındaki işlemleri içerir. Ölçülen disk boyutu küçülmesi yaklaşık %69,5.

* **QDQ (`QuantizeLinear` / `DequantizeLinear`) Düğüm Dengesi:** Ağırlıklar dosya içinde doğrudan INT8 olarak saklandığından çalışma zamanında ağırlıklar için yalnızca `DequantizeLinear` (DQ) düğümü yer alır; aktivasyonlar ise hem `QuantizeLinear` (Q) hem `DequantizeLinear` (DQ) düğümünden geçer. Bu nedenle DQ sayısı Q sayısının yaklaşık iki katıdır (`194` vs. `398`). ONNX Runtime, çıkarım sırasında `Q -> DQ -> Conv` bloklarını grafik birleştirme (graph fusion) ile tek bir donanım hızlandırmalı INT8 çekirdeğine (Raspberry Pi 5 üzerinde **ARM NEON INT8**) dönüştürür.

### 7.3. Seed Bazlı Quantization Özeti

| Seed | FP32 Boyut | INT8 Boyut | Boyut Düşüşü | `QuantizeLinear` | `DequantizeLinear` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 9.40 MB | 2.87 MB | **69.5%** | 194 | 398 |
| **Seed 1** | 9.40 MB | 2.87 MB | **69.5%** | 194 | 398 |
| **Seed 2** | 9.40 MiB | 2.87 MiB | **69.5%** | 194 | 398 |

### 7.4. Laptop CPU Doğrulama Ön-Kontrolü (`val` Split, `batch=1, rect=False`)

FP32 ve INT8 ONNX modelleri Pi 5'e aktarılmadan önce aynı `val` kümesinde (81 görüntü, 104 levha, `640px`) karşılaştırıldı. Seed 0 ve 1, Intel Core i7-14650HX üzerinde ölçüldü. Seed 2, Enes'in Intel Core i5-13500H işlemcili laptopunda ölçüldü.

#### Seed 0 — FP32 vs. INT8 (`val`, Laptop CPU)

| Metrik (Seed 0 — Val, CPU) | ONNX (`FP32`) | ONNX (`INT8`) | Fark ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Model Boyutu** | 9.40 MB | 2.87 MB | $-69.5\%$ |
| **Precision** | 0.9626 | 0.9476 | $-0.0150$ |
| **Recall** | 0.9059 | 0.9022 | $-0.0037$ |
| $\text{mAP}_{50}$ | 0.9626 | 0.9591 | $-0.0035$ |
| $\text{mAP}_{50\text{-}95}$ | 0.7865 | 0.7737 | $-0.0128$ |
| **Çıkarım Süresi (Laptop CPU)** | `32.1 ms` | `18.3 ms` | $1.75\times$ hızlanma ($-43.0\%$) |

*Seed 0 INT8 Sınıf Bazlı `val` (`0.5 ms` preprocess, `18.3 ms` inference, `0.5 ms` postprocess):*

| Sınıf (Seed 0 INT8 — Val) | Görüntü | Levha | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all`** | **81** | **104** | **0.948** | **0.902** | **0.959** | **0.774** |
| `prohibitory` | 43 | 68 | 0.932 | 0.897 | 0.944 | 0.783 |
| `danger` | 20 | 25 | 0.911 | 0.920 | 0.952 | 0.795 |
| `mandatory` | 10 | 11 | 1.000 | 0.890 | 0.981 | 0.743 |

#### Seed 1 — FP32 vs. INT8 (`val`, Laptop CPU)

| Metrik (Seed 1 — Val, CPU) | ONNX (`FP32`) | ONNX (`INT8`) | Fark ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Model Boyutu** | 9.40 MB | 2.87 MB | $-69.5\%$ |
| **Precision** | 0.9174 | 0.8814 | $-0.0360$ |
| **Recall** | 0.8538 | 0.8829 | $+0.0291$ |
| $\text{mAP}_{50}$ | 0.9015 | 0.9014 | $-0.0001$ |
| $\text{mAP}_{50\text{-}95}$ | 0.7555 | 0.7352 | $-0.0203$ |
| **Çıkarım Süresi (Laptop CPU)** | `18.1 ms` | `20.8 ms` | $+2.7\text{ ms}$ *(x86 boost dalgalanması)* |

*Seed 1 INT8 Sınıf Bazlı `val` (`0.5 ms` preprocess, `20.8 ms` inference, `0.5 ms` postprocess):*

| Sınıf (Seed 1 INT8 — Val) | Görüntü | Levha | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all`** | **81** | **104** | **0.881** | **0.883** | **0.901** | **0.735** |
| `prohibitory` | 43 | 68 | 0.899 | 0.912 | 0.963 | 0.782 |
| `danger` | 20 | 25 | 0.927 | 0.920 | 0.912 | 0.752 |
| `mandatory` | 10 | 11 | 0.818 | 0.817 | 0.829 | 0.672 |

#### Seed 2 — FP32 ve INT8 doğrulaması (`val`, Intel Core i5-13500H)

Enes, iki ONNX modelini `640 × 640`, `batch=1`, `rect=False` ayarlarında ve ONNX Runtime `1.23.2` CPU sağlayıcısıyla çalıştırdı. Export kontrolünde `.pt` ile FP32 ONNX metrikleri dört ondalık basamağa kadar eşleşti (`PASS`). INT8 kalibrasyonu `train` kümesinden seçilen 100 görüntüyle yapıldı.

| Metrik | FP32 ONNX | INT8 ONNX | Fark (INT8 − FP32) |
| :--- | ---: | ---: | ---: |
| Model boyutu | 9.40 MiB | 2.87 MiB | −69.5% |
| Precision | 0.9413 | 0.9257 | −0.0156 |
| Recall | 0.9417 | 0.9163 | −0.0254 |
| mAP50 | 0.9354 | 0.9317 | −0.0037 |
| mAP50–95 | 0.7757 | 0.7685 | −0.0072 |
| Çıkarım süresi | 30.8 ms/görüntü | 30.9 ms/görüntü | +0.1 ms/görüntü |

* **Ön-Kontrol Yorumu:**
  * **Doğruluk Korunumu:** `mAP50` değişimi Seed 0'da −0,0035, Seed 1'de −0,0001 ve Seed 2'de −0,0037. Daha sıkı kutu örtüşme eşiklerini de içeren `mAP50–95` değişimi sırasıyla −0,0128, −0,0203 ve −0,0072.
  * **Hız Ölçümü Notu:** Tablolardaki süreler laptop CPU'larında yapılan `val` koşularına ait. Raspberry Pi 5 için gecikme ve hızlanma, ısınma çalıştırmalarından sonra 227 görüntülük `test` kümesinde ölçülecek.

## 8. $320 \times 320$ Çözünürlük Ablasyonu (Eğitim, ONNX Export ve Statik INT8 PTQ)

Giriş çözünürlüğünü $640 \times 640$'tan $320 \times 320$'ye düşürmenin tespit doğruluğu, küçük levha hassasiyeti, INT8 kuantizasyon duyarlılığı ve Raspberry Pi 5 CPU gecikmesi üzerindeki etkisini adil biçimde ölçmek için `YOLO26n` mimarisi aynı `official` GTSDB ayrımı (`378` train / `81` val / `227` held-out test) üzerinde sıfırdan fine-tune edilmiştir.

### 8.1. Protokol Tutarlılığı ve Hesaplama Maliyeti

* **Birebir Hiperparametre Eşleşmesi:** `runs/train/yolo26n_img640_seed0/args.yaml` ile `runs/train/yolo26n_img320_seed0/args.yaml` arasındaki `diff` kontrolüyle 115 parametrenin tamamında yalnızca `imgsz` (`640 -> 320`), `name` ve `save_dir` alanlarının değiştiği doğrulanmıştır (`model=yolo26n.pt`, `epochs=100`, `patience=25`, `batch=16`, `deterministic=True`, `fliplr=0.0`, `seeds={0,1,2}`, PyTorch `2.9.1+cu128`, Ultralytics `8.4.215`).
* **Hesaplama ve Çıkış Tensörü Farkı:**
  * **Parametre Sayısı:** $2{,}390{,}221$ (değişmedi; tam evrişimli mimari).
  * **Hesaplama Maliyeti:** $5.4\text{ GFLOPs} \rightarrow \mathbf{1.3\text{ GFLOPs}}$ (**$4.15\times$ daha düşük işlem yükü**, $-75.9\%$).
  * **Giriş / Çıkış Şekli:** $(1, 3, 320, 320) \rightarrow (1, 7, 2100)$ ($40^2 + 20^2 + 10^2 = 1600 + 400 + 100 = 2100$ aday hücre; $640\text{ px}$'e göre $4\times$ daha az aday).

### 8.2. 3-Seed Eğitim Doğrulama Özeti (`val` — 81 Görüntü, 104 Levha, Ultralytics Varsayılan `rect=True`)

| Seed | En İyi Epoch | Toplam Epoch | Süre (saat) | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seed 0** | 86 | 100 | 0.104 | 0.8836 | 0.6405 | 0.7389 | 0.5241 | `models/yolo26n_img320_seed0.pt` |
| **Seed 1** | 83 | 100 | 0.118 | 0.9400 | 0.5692 | 0.6937 | 0.4917 | `models/yolo26n_img320_seed1.pt` |
| **Seed 2** | 83 | 100 | 0.110 | 0.8910 | 0.6809 | 0.7710 | 0.5197 | `models/yolo26n_img320_seed2.pt` |
| **Ortalama $\pm$ Std** | — | — | — | **$0.9049 \pm 0.0250$** | **$0.6302 \pm 0.0462$** | **$0.7345 \pm 0.0317$** | **$0.5118 \pm 0.0143$** | — |

#### Sınıf Bazlı Eğitim Doğrulama Dağılımı (`imgsz=320`, 3 Seed)

| Seed | Sınıf | Görüntü | Levha | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | `prohibitory` | 43 | 68 | 0.859 | 0.676 | 0.778 | 0.566 |
| **Seed 0** | `danger` | 20 | 25 | 0.791 | 0.720 | 0.797 | 0.586 |
| **Seed 0** | `mandatory` | 10 | 11 | 1.000 | 0.525 | 0.641 | 0.421 |
| **Seed 1** | `prohibitory` | 43 | 68 | 1.000 | 0.624 | 0.786 | 0.547 |
| **Seed 1** | `danger` | 20 | 25 | 0.916 | 0.720 | 0.817 | 0.595 |
| **Seed 1** | `mandatory` | 10 | 11 | 0.904 | 0.364 | 0.478 | 0.334 |
| **Seed 2** | `prohibitory` | 43 | 68 | 0.894 | 0.676 | 0.801 | 0.552 |
| **Seed 2** | `danger` | 20 | 25 | 0.779 | 0.840 | 0.873 | 0.632 |
| **Seed 2** | `mandatory` | 10 | 11 | 1.000 | 0.526 | 0.639 | 0.375 |

### 8.3. FP32 ONNX Export, Parity Kontrolü, Statik INT8 PTQ ve 6-Model Doğrulaması

Her üç seed için `src/export_onnx.py --imgsz 320`, `src/verify_onnx.py --imgsz 320`, `src/quantize_int8.py --imgsz 320` (aynı `train` kümesinden `calib_seed=42` ile 100 kalibrasyon görüntüsü, `QDQ`, `MinMax`, `op_types_to_quantize=["Conv"]`) ve `src/verify_onnx_suite.py --imgsz 320 --eval-val` çalıştırılmıştır.

* **6-Model Bütünlük & Örnek Inference Doğrulaması (`results/ablation_320/summary.json`):**
  * Altı ONNX dosyasının tamamı (`yolo26n_img320_seed{0,1,2}_{fp32,int8}.onnx`) `onnx.checker.check_model` ve `ONNX Runtime 1.23.2` (`CPUExecutionProvider`) oturum açılışından hatasız geçti.
  * Giriş/çıkış şekilleri 6/6 modelde `[1, 3, 320, 320] -> [1, 7, 2100]` (`tensor(float)`) olarak doğrulandı.
  * Gerçek `val` görüntüsü (`00004.jpg`, `benchmark_pi_v3.preprocess(..., 320)`) üzerinde sonlu (`all_finite=True`) çıktı ve `benchmark_pi_v3.decode_prediction(conf=0.25)` ile tespit üretimi teyit edildi (Seed 0: `2/2`, Seed 1: `2/2`, Seed 2: `1/1`).
  * Üç INT8 modelin her birinde `102` `Conv` (`102` `INT32` bias), `194` `QuantizeLinear`, `398` `DequantizeLinear` düğümü ve **`9.28 MiB -> 2.75 MiB` (`-70.4%`)** disk küçülmesi doğrulandı.

#### Seed Bazlı ONNX Parity ve Laptop CPU `val` Ön-Kontrolü (`imgsz=320`, `batch=1, rect=False`)

| Seed | Format | Boyut | Parity ($\Delta$ vs `.pt`) | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ | $\Delta\text{mAP}_{50}$ | $\Delta\text{mAP}_{50\text{-}95}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | FP32 ONNX | 9.28 MiB | `PASS` (0.0000) | 0.9075 | 0.5896 | 0.6938 | 0.5035 | — | — |
| **Seed 0** | INT8 ONNX | 2.75 MiB | `Q=194, DQ=398` | 0.8949 | 0.5725 | 0.6732 | 0.4785 | $-0.0205$ | $-0.0251$ |
| **Seed 1** | FP32 ONNX | 9.28 MiB | `PASS` (0.0000) | 0.9240 | 0.5695 | 0.6962 | 0.4884 | — | — |
| **Seed 1** | INT8 ONNX | 2.75 MiB | `Q=194, DQ=398` | 0.9314 | 0.5559 | 0.6644 | 0.4644 | $-0.0318$ | $-0.0240$ |
| **Seed 2** | FP32 ONNX | 9.28 MiB | `PASS` (0.0000) | 0.8530 | 0.6838 | 0.7516 | 0.5300 | — | — |
| **Seed 2** | INT8 ONNX | 2.75 MiB | `Q=194, DQ=398` | 0.9597 | 0.5987 | 0.7044 | 0.4773 | $-0.0472$ | $-0.0526$ |
| **Ortalama $\pm$ Std** | **FP32 ONNX** | **9.28 MiB** | **3/3 `PASS`** | **$0.8949 \pm 0.0304$** | **$0.6143 \pm 0.0498$** | **$0.7138 \pm 0.0267$** | **$0.5073 \pm 0.0172$** | — | — |
| **Ortalama $\pm$ Std** | **INT8 ONNX** | **2.75 MiB** | **$-70.4\%$ boyut** | **$0.9286 \pm 0.0265$** | **$0.5757 \pm 0.0176$** | **$0.6807 \pm 0.0172$** | **$0.4734 \pm 0.0064$** | **$-0.0332$** | **$-0.0339$** |

### 8.4. $640\text{ px}$ vs. $320\text{ px}$ Ön-Karşılaştırma (`val` Split, `batch=1, rect=False`, 3-Seed Ortalama $\pm$ Std)

| Konfigürasyon | GFLOPs | Boyut | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ | INT8 $\Delta\text{mAP}_{50}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`640px` FP32** | 5.4 | 9.40 MiB | $0.9404 \pm 0.0185$ | $0.9005 \pm 0.0361$ | $0.9332 \pm 0.0250$ | $0.7726 \pm 0.0128$ | — |
| **`640px` INT8** | 5.4 | 2.87 MiB | $0.9182 \pm 0.0275$ | $0.9005 \pm 0.0137$ | $0.9307 \pm 0.0236$ | $0.7591 \pm 0.0171$ | $-0.0024$ |
| **`320px` FP32** | 1.3 | 9.28 MiB | $0.8949 \pm 0.0304$ | $0.6143 \pm 0.0498$ | $0.7138 \pm 0.0267$ | $0.5073 \pm 0.0172$ | — |
| **`320px` INT8** | 1.3 | 2.75 MiB | $0.9286 \pm 0.0265$ | $0.5757 \pm 0.0176$ | $0.6807 \pm 0.0172$ | $0.4734 \pm 0.0064$ | $-0.0332$ |

* **Ön-Karşılaştırma Bulguları (H2 Doğrulaması İçin):**
  * **Çözünürlük Düşüşünün Etkisi:** Giriş boyutunun `640 -> 320` düşürülmesi teorik hesaplama maliyetini $4.15\times$ azaltırken (`5.4 -> 1.3 GFLOPs`), FP32 `val` $\text{mAP}_{50}$ değerini $-0.2194$ ($0.9332 \rightarrow 0.7138$) ve Recall değerini $-0.2862$ ($0.9005 \rightarrow 0.6143$) düşürmüştür. Uzaktaki küçük levhaların $320 \times 320$ ölçeğinde birkaç piksele sıkışması nedeniyle özellikle az örnekli `mandatory` sınıfında belirgin duyarlılık kaybı yaşanmaktadır.
  * **Bileşik Kuantizasyon Hassasiyeti:** $640\text{ px}$'te statik INT8 PTQ $\text{mAP}_{50}$ değerini neredeyse hiç etkilemezken ($-0.0024$), $320\text{ px}$'te aynı PTQ protokolü $\text{mAP}_{50}$ üzerinde $-0.0332$ ve $\text{mAP}_{50\text{-}95}$ üzerinde $-0.0339$ kayba yol açmıştır. Düşük çözünürlükteki zayıf özellik aktivasyonları 8-bit yuvarlama gürültüsüne karşı daha hassastır.
  * **Kapsam Notu:** Yukarıdaki tüm ölçümler yalnızca `val` kümesine aittir; $227$ görüntülük `test` kümesi üzerindeki kesin mAP, gecikme (ms/FPS) ve 640 vs. 320 Raspberry Pi 5 benchmark karşılaştırması Enes tarafından Pi 5 üzerinde yürütülecektir.

## 9. Raspberry Pi 5 Final Test ve Performans Ölçümleri (`test` — 227 Görüntü)

*Bu bölüm Raspberry Pi 5 üzerinde aynı `test` kümesi (`227` görüntü, `228` levha, `640px` ve `320px`, `batch=1`) ile FP32 ve INT8 ONNX çiftleri ölçüldükçe doldurulacaktır.*

| Çözünürlük | Seed | Format | Boyut (MiB) | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50\text{-}95}$ | Gecikme ($\text{ms/img}$) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **640px** | **Seed 0** | FP32 | 9.40 | — | — | — | — | — | — |
| **640px** | **Seed 0** | INT8 | 2.87 | — | — | — | — | — | — |
| **640px** | **Seed 1** | FP32 | 9.40 | — | — | — | — | — | — |
| **640px** | **Seed 1** | INT8 | 2.87 | — | — | — | — | — | — |
| **640px** | **Seed 2** | FP32 | 9.40 | — | — | — | — | — | — |
| **640px** | **Seed 2** | INT8 | 2.87 | — | — | — | — | — | — |
| **320px** | **Seed 0** | FP32 | 9.28 | — | — | — | — | — | — |
| **320px** | **Seed 0** | INT8 | 2.75 | — | — | — | — | — | — |
| **320px** | **Seed 1** | FP32 | 9.28 | — | — | — | — | — | — |
| **320px** | **Seed 1** | INT8 | 2.75 | — | — | — | — | — | — |
| **320px** | **Seed 2** | FP32 | 9.28 | — | — | — | — | — | — |
| **320px** | **Seed 2** | INT8 | 2.75 | — | — | — | — | — | — |

## 10. 3-Seed Genel Karşılaştırma ve Ödünleşim Analizi (FP32 vs. INT8 & 640px vs. 320px)

*Tüm 640px ve 320px seed çiftlerinin Raspberry Pi 5 testleri tamamlandığında Ortalama $\pm$ Std tablosu, hızlanma çarpanı ($\text{Speedup}$), boyut sıkıştırma oranı ve küçük levha Recall farkı buraya eklenecektir.*

## 11. Depo ve Sürüm Kontrolü Kaydı

* **Git Akışı:** `feat/dataset-setup` ve `feat/benchmark-results` (PR #3: `src/benchmark_pi_v3.py`, `src/analyze_object_size.py` + testler) `main` ile birleştirildi; 320px çözünürlük ablasyonu doğrudan `main` üzerinde düzenli klasör yapısıyla eklendi.
* **İzlenen Varlıklar:** `.gitignore` üzerinden `data/`, tekrarlı `runs/**/weights/` (`best.pt` / `last.pt` kopyaları) ve geçici `runs/detect/` dizinleri hariç tutularak tüm kanonik en iyi ağırlıklar (`models/yolo26n_img{640,320}_seed{0,1,2}.pt` ve 12 `.onnx` modeli), taban ağırlık (`yolo26n.pt`), eğitim log/grafikleri (`runs/train/`), doğrulama özeti (`results/ablation_320/summary.json`), `archive.md` ve `README.md` dokümantasyonu `main` dalında tutulmaktadır.
* **Görev Takibi:** GitHub Issues üzerinde 3 seed için ayrı ayrı görev kutucuklarını içeren ana takip issue'su oluşturuldu.
