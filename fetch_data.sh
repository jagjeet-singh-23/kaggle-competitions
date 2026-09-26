#!/usr/bin/env bash
# Pull the competition data: BirdCLEF+ 2026 (16.1 GB), LANL Earthquake (10.4 GB),
# Enveda CASMI-26 (3.0 GB). Portable -- bash, curl and unzip only, no kaggle CLI.
#
# The CLI buffers a whole file in RAM before writing it, which lost a 3 GB download
# to an OOM reboot; curl -C - streams to disk and resumes.
#
# Requires the rules accepted on kaggle.com for each competition first, else every
# call 403s. Kaggle serves large files zipped even when the URL ends .csv, so
# everything lands as .zip and is expanded here.
#
#   ./fetch_data.sh                 # all three
#   ./fetch_data.sh molecule        # just one (molecule | birdclef | lanl)
set -euo pipefail
cd "$(dirname "$0")"

creds=$(python3 -c "import json;d=json.load(open('$HOME/.kaggle/kaggle.json'));print(d['username']+':'+d['key'])")
API=https://www.kaggle.com/api/v1/competitions/data

want() {   # want <name> -- true if no filter was given, or this name was asked for
  [ $# -eq 1 ] && [ ${#SEL[@]} -eq 0 ] && return 0
  for a in ${SEL[@]+"${SEL[@]}"}; do [ "$a" = "$1" ] && return 0; done
  return 1
}

fetch() {  # fetch <url> <out>
  mkdir -p "$(dirname "$2")"
  echo ">> $2"
  curl -fL -C - --retry 5 --retry-delay 5 --progress-bar -u "$creds" -o "$2" "$1"
}

expand() {  # expand <path> -- unzip in place if the file is actually a zip
  [ "$(head -c2 "$1")" = "PK" ] || return 0
  echo ".. unzipping $1"
  unzip -oq "$1" -d "$(dirname "$1")" && rm -f "$1"
}

SEL=("$@")

if want birdclef; then
  # Metadata first so EDA can start while the bulk archive downloads.
  for f in taxonomy.csv train.csv train_soundscapes_labels.csv \
           sample_submission.csv recording_location.txt; do
    fetch "$API/download/birdclef-2026/$f" "BirdCLEF/data/$f"
    expand "BirdCLEF/data/$f"
  done
  fetch "$API/download-all/birdclef-2026" BirdCLEF/data/birdclef-2026.zip
  unzip -oq BirdCLEF/data/birdclef-2026.zip -d BirdCLEF/data
  rm -f BirdCLEF/data/birdclef-2026.zip
fi

if want lanl; then
  fetch "$API/download/LANL-Earthquake-Prediction/sample_submission.csv" \
        LANL_EQ_Prediction/data/sample_submission.csv
  fetch "$API/download/LANL-Earthquake-Prediction/train.csv" \
        LANL_EQ_Prediction/data/train.csv.zip
  expand LANL_EQ_Prediction/data/train.csv.zip
  fetch "$API/download-all/LANL-Earthquake-Prediction" LANL_EQ_Prediction/data/lanl.zip
  unzip -oq LANL_EQ_Prediction/data/lanl.zip -d LANL_EQ_Prediction/data
  rm -f LANL_EQ_Prediction/data/lanl.zip
fi

if want molecule; then
  M=enveda-CASMI26-molecule-id-mass-spectra
  for f in train.parquet test.parquet sample_submission.csv visible_test_answers.csv; do
    fetch "$API/download/$M/$f" "MoleculeID-from-Mass-Spectra/data/$f"
    expand "MoleculeID-from-Mass-Spectra/data/$f"
  done
fi

echo "done"
