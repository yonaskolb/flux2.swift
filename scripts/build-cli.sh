#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCHEME="${SCHEME:-flux2-cli}"
DESTINATION="${DESTINATION:-platform=macOS}"
CONFIGURATION="${CONFIGURATION:-Release}"

# Keep DerivedData outside the repo. Putting it under ./build triggers Xcode
# package container conflicts for dependencies.
DERIVED_DATA="${DERIVED_DATA:-$HOME/Library/Developer/Xcode/DerivedData/Flux2CLI}"

xcodebuild \
  -scheme "$SCHEME" \
  -destination "$DESTINATION" \
  -configuration "$CONFIGURATION" \
  -derivedDataPath "$DERIVED_DATA" \
  build

PRODUCT_DIR="$DERIVED_DATA/Build/Products/$CONFIGURATION"
BIN="$PRODUCT_DIR/flux2-cli"
OUT_DIR="$ROOT_DIR/.build"
mkdir -p "$OUT_DIR"
cp -f "$BIN" "$OUT_DIR/flux2-cli"

# Copy required SwiftPM resource bundles next to the executable so MLX/Hub can
# locate embedded resources at runtime (e.g. metallib).
shopt -s nullglob
for bundle in "$PRODUCT_DIR"/*.bundle; do
  name="$(basename "$bundle")"
  if [[ -d "$OUT_DIR/$name" ]]; then
    rm -r "$OUT_DIR/$name"
  fi
  cp -R "$bundle" "$OUT_DIR/$name"
done

if [[ -d "$PRODUCT_DIR/PackageFrameworks" ]]; then
  if [[ -d "$OUT_DIR/PackageFrameworks" ]]; then
    rm -r "$OUT_DIR/PackageFrameworks"
  fi
  cp -R "$PRODUCT_DIR/PackageFrameworks" "$OUT_DIR/PackageFrameworks"
fi

echo "Built: $OUT_DIR/flux2-cli (+ bundles)"
