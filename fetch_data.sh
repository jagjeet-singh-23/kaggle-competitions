#!/usr/bin/env bash
# Pull BirdCLEF+ 2026 (16.1 GB) and LANL Earthquake (10.4 GB) competition data.
# Requires competition rules accepted on kaggle.com first, else every call 403s.
# Kaggle serves large single files zipped, so everything lands as .zip and is expanded here.
set -euo pipefail
cd "$(dirname "$0")"

creds=$(python3 -c "import json;d=json.load(open('$HOME/.kaggle/kaggle.json'));print(d['username']+':'+d['key'])")
API=https://www.kaggle.com/api/v1/competitions/data

fetch() {  # fetch <url> <out>
  echo ">> $2"
  curl -fL -C - --retry 5 --retry-delay 5 --progress-bar -u "$creds" -o "$2" "$1"
}

expand() {  # expand <path> -- unzip in place if it is actually a zip
  [ "$(head -c2 "$1")" = "PK" ] || return 0
  echo ".. unzipping $1"
  unzip -oq "$1" -d "$(dirname "$1")" && rm -f "$1"
}

# Metadata first so EDA can start while the bulk archives download.
for f in taxonomy.csv train.csv train_soundscapes_labels.csv sample_submission.csv recording_location.txt; do
  fetch "$API/download/birdclef-2026/$f" "BirdCLEF/data/$f"
  expand "BirdCLEF/data/$f"
done

fetch "$API/download/LANL-Earthquake-Prediction/sample_submission.csv" LANL_EQ_Prediction/data/sample_submission.csv
fetch "$API/download/LANL-Earthquake-Prediction/train.csv" LANL_EQ_Prediction/data/train.csv.zip
expand LANL_EQ_Prediction/data/train.csv.zip

# Bulk: 16.1 GB of ogg, and the 2,624 LANL test segments.
fetch "$API/download-all/birdclef-2026" BirdCLEF/data/birdclef-2026.zip
unzip -oq BirdCLEF/data/birdclef-2026.zip -d BirdCLEF/data && rm -f BirdCLEF/data/birdclef-2026.zip

fetch "$API/download-all/LANL-Earthquake-Prediction" LANL_EQ_Prediction/data/lanl.zip
unzip -oq LANL_EQ_Prediction/data/lanl.zip -d LANL_EQ_Prediction/data && rm -f LANL_EQ_Prediction/data/lanl.zip

echo "done"
