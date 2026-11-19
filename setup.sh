#!/usr/bin/env bash
set -e

RAW_DIR="data/raw"
ZIP_FILE="$RAW_DIR/FullIJCNN2013.zip"
EXTRACTED_DIR="$RAW_DIR/FullIJCNN2013"
URL="https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"

echo "[1/3] Klasör yapısı kontrol ediliyor..."
mkdir -p "$RAW_DIR" models results src notebooks

# Eski konumda (data/) veri varsa tekrar indirmemek için data/raw/ altına taşı
if [ -d "data/FullIJCNN2013" ] && [ ! -d "$EXTRACTED_DIR" ]; then
    echo "Eski konumdaki data/FullIJCNN2013 klasörü data/raw/ altına taşınıyor..."
    mv "data/FullIJCNN2013" "$EXTRACTED_DIR"
fi

if [ -f "data/FullIJCNN2013.zip" ] && [ ! -f "$ZIP_FILE" ]; then
    echo "Eski konumdaki data/FullIJCNN2013.zip dosyası data/raw/ altına taşınıyor..."
    mv "data/FullIJCNN2013.zip" "$ZIP_FILE"
fi

if [ -f "$EXTRACTED_DIR/gt.txt" ]; then
    echo "[2/3] GTSDB veri seti zaten hazır ($EXTRACTED_DIR). İndirme atlanıyor."
else
    if [ ! -f "$ZIP_FILE" ]; then
        echo "[2/3] GTSDB (FullIJCNN2013.zip) indiriliyor..."
        wget -c "$URL" -O "$ZIP_FILE"
    else
        echo "[2/3] Zip dosyası mevcut, doğrudan çıkartılıyor..."
    fi
    echo "Arşiv çıkartılıyor (lütfen bekleyin)..."
    unzip -q -o "$ZIP_FILE" -d "$RAW_DIR/"
    echo "Çıkartma tamamlandı."
fi

echo "[3/3] Kurulum hazır! Toplam görüntü ve etiket satırı:"
ls "$EXTRACTED_DIR"/*.ppm | wc -l
wc -l "$EXTRACTED_DIR/gt.txt"
