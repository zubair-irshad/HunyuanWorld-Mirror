#!/usr/bin/env bash
set -e

SRC_DIR="/home/mirshad7/Downloads/data/Omni6DPose/SOPE_webdataset"
OUT_DIR="/home/mirshad7/Downloads/data/Omni6DPose/SOPE_reversed_fixed"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

for f in "$SRC_DIR"/sope-*.tar; do
    echo "Extracting $f ..."
    tar -xf "$f"
done

# echo "🔧 Fixing duplicated filenames (0000.0000_* → 0000_* ) ..."
# # rename all files like "0000.0000_color.png" → "0000_color.png"
# find "$OUT_DIR" -type f -name "*.*_*" | while read -r file; do
#     dir=$(dirname "$file")
#     base=$(basename "$file")
#     new=$(echo "$base" | sed -E 's/^([0-9]+)\.\1_/\1_/')
#     if [[ "$base" != "$new" ]]; then
#         mv "$file" "$dir/$new"
#     fi
# done

# echo "✅ All done! Fixed files are under $OUT_DIR"