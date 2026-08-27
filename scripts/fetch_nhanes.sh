#!/bin/bash
#
# Download a large NHANES file using several connections at once.
#
#   scripts/fetch_nhanes.sh <url> [-o output] [-n connections] [--max-bytes N]
#
# e.g.
#   scripts/fetch_nhanes.sh https://ftp.cdc.gov/pub/NHANES/LargeDataFiles/PAXMIN_H.xpt
#   scripts/fetch_nhanes.sh <url> -n 24
#   scripts/fetch_nhanes.sh <url> --max-bytes 4000000 -o sample.bin   # test the link
#
# Why this exists: ftp.cdc.gov throttles each connection to roughly 90 KB/s, so
# a single wget on the 8.7 GB PAXMIN_H.xpt reports an ETA of about 30 hours.
# The limit is per connection, not per client, so N connections give
# approximately N times the throughput. Sixteen brings it under two hours.
#
# Each connection fetches a byte range into its own part file, and the parts are
# concatenated at the end. Rerunning skips parts that are already complete, so
# an interrupted download resumes rather than starting over.
#
# A plain interrupted download to a network drive can leave a full-length file
# padded with zeros, which looks complete but is not. This script only writes
# the output once every part is present and the total size matches what the
# server advertised.

set -euo pipefail

URL=""
OUTPUT=""
CONNECTIONS=16
MAX_BYTES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)     OUTPUT="$2"; shift 2 ;;
        -n|--connections) CONNECTIONS="$2"; shift 2 ;;
        --max-bytes)     MAX_BYTES="$2"; shift 2 ;;
        -*)              echo "Unknown option: $1" >&2; exit 1 ;;
        *)               URL="$1"; shift ;;
    esac
done

if [[ -z "$URL" ]]; then
    echo "Usage: $0 <url> [-o output] [-n connections] [--max-bytes N]" >&2
    exit 1
fi

[[ -n "$OUTPUT" ]] || OUTPUT="$(basename "$URL")"

# Ask the server how big the file is
TOTAL=$(curl -sIL "$URL" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
if [[ -z "${TOTAL}" || "${TOTAL}" -le 0 ]]; then
    echo "Could not determine the size of ${URL}" >&2
    exit 1
fi

if [[ "$MAX_BYTES" -gt 0 && "$MAX_BYTES" -lt "$TOTAL" ]]; then
    TOTAL="$MAX_BYTES"
    echo "Limiting to the first ${TOTAL} bytes (test mode)"
fi

echo "URL         : ${URL}"
echo "Output      : ${OUTPUT}"
echo "Size        : ${TOTAL} bytes"
echo "Connections : ${CONNECTIONS}"
echo

PARTS_DIR="${OUTPUT}.parts"
mkdir -p "$PARTS_DIR"

CHUNK=$(( (TOTAL + CONNECTIONS - 1) / CONNECTIONS ))
pids=()

for (( i = 0; i < CONNECTIONS; i++ )); do
    START=$(( i * CHUNK ))
    [[ "$START" -lt "$TOTAL" ]] || break
    END=$(( START + CHUNK - 1 ))
    [[ "$END" -lt "$TOTAL" ]] || END=$(( TOTAL - 1 ))
    WANT=$(( END - START + 1 ))
    PART="${PARTS_DIR}/part.$(printf '%03d' "$i")"

    # Skip parts that are already the right length, so a rerun resumes
    if [[ -f "$PART" ]]; then
        HAVE=$(stat -c%s "$PART" 2>/dev/null || echo 0)
        if [[ "$HAVE" -eq "$WANT" ]]; then
            echo "part $(printf '%03d' "$i"): already complete"
            continue
        fi
        rm -f "$PART"
    fi

    echo "part $(printf '%03d' "$i"): bytes ${START}-${END}"
    curl -sS --retry 5 --retry-delay 5 --retry-connrefused \
         -r "${START}-${END}" -o "$PART" "$URL" &
    pids+=($!)
done

failed=0
for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" || failed=1
done

if [[ "$failed" -ne 0 ]]; then
    echo "At least one connection failed. Rerun the same command to resume." >&2
    exit 1
fi

# Check every part before assembling anything
for (( i = 0; i < CONNECTIONS; i++ )); do
    START=$(( i * CHUNK ))
    [[ "$START" -lt "$TOTAL" ]] || break
    END=$(( START + CHUNK - 1 ))
    [[ "$END" -lt "$TOTAL" ]] || END=$(( TOTAL - 1 ))
    WANT=$(( END - START + 1 ))
    PART="${PARTS_DIR}/part.$(printf '%03d' "$i")"
    HAVE=$(stat -c%s "$PART" 2>/dev/null || echo 0)
    if [[ "$HAVE" -ne "$WANT" ]]; then
        echo "part $(printf '%03d' "$i") is ${HAVE} bytes, expected ${WANT}." >&2
        echo "Rerun the same command to fetch the missing pieces." >&2
        exit 1
    fi
done

echo
echo "Assembling ${OUTPUT}..."
cat "${PARTS_DIR}"/part.* > "$OUTPUT"

FINAL=$(stat -c%s "$OUTPUT")
if [[ "$FINAL" -ne "$TOTAL" ]]; then
    echo "Assembled ${FINAL} bytes but expected ${TOTAL}. Leaving parts in place." >&2
    exit 1
fi

rm -rf "$PARTS_DIR"
echo "Done: ${OUTPUT} (${FINAL} bytes)"
echo
echo "Size matching is necessary but not sufficient - a padded file matches too."
echo "Check the contents with:"
echo "    python scripts/check_data_integrity.py --cohort <G|H>"
