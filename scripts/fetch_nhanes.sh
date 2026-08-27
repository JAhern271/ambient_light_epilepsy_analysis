#!/bin/bash
#
# Download a large NHANES file using several connections at once.
#
#   scripts/fetch_nhanes.sh <url> [-o output] [-n connections] [--max-bytes N]
#
# e.g.
#   scripts/fetch_nhanes.sh https://ftp.cdc.gov/pub/NHANES/LargeDataFiles/PAXMIN_H.xpt
#   scripts/fetch_nhanes.sh <url> --max-bytes 4000000 -o sample.bin   # test the link
#
# Why this exists: ftp.cdc.gov throttles each connection to roughly 90 KB/s, so
# a single wget on the 8.7 GB PAXMIN_H.xpt reports an ETA of about 30 hours.
# The limit is per connection, not per client, so N connections give
# approximately N times the throughput. Sixteen brings it under two hours.
#
# Each connection fetches a byte range into its own part file. A connection that
# times out mid-transfer is retried, and picks up from the bytes already on
# disk rather than starting that range again - over a transfer this long, a
# dropped connection is expected rather than exceptional.
#
# A plain interrupted download to a network drive can leave a full-length file
# padded with zeros, which looks complete but is not. This script writes the
# output only once every part is present and the total matches the size the
# server advertised.

set -euo pipefail

URL=""
OUTPUT=""
CONNECTIONS=16
MAX_BYTES=0
MAX_ATTEMPTS=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)      OUTPUT="$2"; shift 2 ;;
        -n|--connections) CONNECTIONS="$2"; shift 2 ;;
        --max-bytes)      MAX_BYTES="$2"; shift 2 ;;
        -*)               echo "Unknown option: $1" >&2; exit 1 ;;
        *)                URL="$1"; shift ;;
    esac
done

if [[ -z "$URL" ]]; then
    echo "Usage: $0 <url> [-o output] [-n connections] [--max-bytes N]" >&2
    exit 1
fi

[[ -n "$OUTPUT" ]] || OUTPUT="$(basename "$URL")"

TOTAL=$(curl -sIL "$URL" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
if [[ -z "${TOTAL}" || "${TOTAL}" -le 0 ]]; then
    echo "Could not determine the size of ${URL}" >&2
    exit 1
fi

if [[ "$MAX_BYTES" -gt 0 && "$MAX_BYTES" -lt "$TOTAL" ]]; then
    TOTAL="$MAX_BYTES"
    echo "Limiting to the first ${TOTAL} bytes (test mode)"
fi

PARTS_DIR="${OUTPUT}.parts"
mkdir -p "$PARTS_DIR"
MANIFEST="${PARTS_DIR}/manifest"

# Chunk boundaries depend on the connection count, so resuming with a different
# -n would invalidate every part already downloaded. Hold the original.
if [[ -f "$MANIFEST" ]]; then
    # shellcheck disable=SC1090
    source "$MANIFEST"
    if [[ "$SAVED_TOTAL" -ne "$TOTAL" ]]; then
        echo "The file size changed since this download started" >&2
        echo "  was ${SAVED_TOTAL}, now ${TOTAL}. Delete ${PARTS_DIR} and start again." >&2
        exit 1
    fi
    if [[ "$SAVED_CONNECTIONS" -ne "$CONNECTIONS" ]]; then
        echo "Resuming with ${SAVED_CONNECTIONS} connections, not ${CONNECTIONS}:"
        echo "  the existing parts were split that way and would otherwise be discarded."
        CONNECTIONS="$SAVED_CONNECTIONS"
    fi
else
    printf 'SAVED_TOTAL=%s\nSAVED_CONNECTIONS=%s\n' "$TOTAL" "$CONNECTIONS" > "$MANIFEST"
fi

echo "URL         : ${URL}"
echo "Output      : ${OUTPUT}"
echo "Size        : ${TOTAL} bytes"
echo "Connections : ${CONNECTIONS}"
echo

CHUNK=$(( (TOTAL + CONNECTIONS - 1) / CONNECTIONS ))

part_path() { printf '%s/part.%03d' "$PARTS_DIR" "$1"; }
part_size() { stat -c%s "$1" 2>/dev/null || echo 0; }

# Fetch one byte range, resuming from whatever is already on disk.
fetch_part() {
    local idx="$1" start="$2" end="$3"
    local part err want have from attempt=0
    part="$(part_path "$idx")"
    err="${part}.err"
    want=$(( end - start + 1 ))

    while (( attempt < MAX_ATTEMPTS )); do
        have="$(part_size "$part")"

        if (( have == want )); then
            rm -f "$err"
            return 0
        fi
        if (( have > want )); then          # can only be corruption; start over
            rm -f "$part"
            have=0
        fi

        from=$(( start + have ))
        # curl's stderr is kept, not discarded: an unsupported option or a
        # refused range fails instantly and silently otherwise, and the loop
        # then spins to its attempt limit with nothing in the log to say why.
        # --speed-time/--speed-limit abandon a stalled connection so it is
        # retried rather than hanging until the TCP timeout.
        curl -sS --fail --retry 3 --retry-delay 5 --speed-time 60 --speed-limit 1024 -r "${from}-${end}" "$URL" >> "$part" 2>"$err" || true

        attempt=$(( attempt + 1 ))

        if (( "$(part_size "$part")" == have )); then
            if [[ -s "$err" ]]; then
                echo "part $(printf %03d "$idx") attempt ${attempt}: $(head -1 "$err")" >&2
            else
                echo "part $(printf %03d "$idx") attempt ${attempt}: no bytes received" >&2
            fi
            sleep 10
        fi
    done

    echo "part $(printf '%03d' "$idx"): gave up after ${MAX_ATTEMPTS} attempts" >&2
    return 1
}

pids=()
for (( i = 0; i < CONNECTIONS; i++ )); do
    START=$(( i * CHUNK ))
    [[ "$START" -lt "$TOTAL" ]] || break
    END=$(( START + CHUNK - 1 ))
    [[ "$END" -lt "$TOTAL" ]] || END=$(( TOTAL - 1 ))
    WANT=$(( END - START + 1 ))
    HAVE="$(part_size "$(part_path "$i")")"

    if (( HAVE == WANT )); then
        echo "part $(printf '%03d' "$i"): already complete"
        continue
    fi
    if (( HAVE > 0 )); then
        echo "part $(printf '%03d' "$i"): resuming at ${HAVE} of ${WANT} bytes"
    else
        echo "part $(printf '%03d' "$i"): bytes ${START}-${END}"
    fi

    fetch_part "$i" "$START" "$END" &
    pids+=($!)
done

failed=0
for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" || failed=1
done

# Check every part before assembling anything
for (( i = 0; i < CONNECTIONS; i++ )); do
    START=$(( i * CHUNK ))
    [[ "$START" -lt "$TOTAL" ]] || break
    END=$(( START + CHUNK - 1 ))
    [[ "$END" -lt "$TOTAL" ]] || END=$(( TOTAL - 1 ))
    WANT=$(( END - START + 1 ))
    HAVE="$(part_size "$(part_path "$i")")"
    if (( HAVE != WANT )); then
        echo "part $(printf '%03d' "$i") is ${HAVE} bytes, expected ${WANT}." >&2
        failed=1
    fi
done

if (( failed )); then
    echo "Incomplete. Rerun the identical command to resume from what is on disk." >&2
    exit 1
fi

echo
echo "Assembling ${OUTPUT}..."
# Match only part.NNN: the .err files live in the same directory, and a
# bare part.* glob would concatenate them into the output.
cat "${PARTS_DIR}"/part.[0-9][0-9][0-9] > "$OUTPUT"

FINAL="$(part_size "$OUTPUT")"
if (( FINAL != TOTAL )); then
    echo "Assembled ${FINAL} bytes but expected ${TOTAL}. Leaving parts in place." >&2
    exit 1
fi

rm -rf "$PARTS_DIR"
echo "Done: ${OUTPUT} (${FINAL} bytes)"
echo
echo "Size matching is necessary but not sufficient - a padded file matches too."
echo "Check the contents with:"
echo "    python scripts/check_data_integrity.py --cohort <G|H>"
