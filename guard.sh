#!/bin/sh
# Kill the child process if free memory falls below MIN_AVAIL_MB, so a runaway job
# dies instead of the machine. This exists because loading the 6.4 GB-as-Arrow
# train.parquet whole froze a 15 GB laptop twice and cost two reboots.
#
#   MIN_AVAIL_MB=2500 ./guard.sh python3 -u MoleculeID-from-Mass-Spectra/search.py
#
# Exits 137 if it had to kill, otherwise the child's own status.
: "${MIN_AVAIL_MB:=2000}"
: "${GUARD_POLL:=2}"

avail_mb() {
    if [ "$(uname)" = "Darwin" ]; then
        # No MemAvailable on macOS. Free + inactive + speculative pages are what the
        # kernel can hand out without swapping; purgeable is counted inside inactive.
        vm_stat | awk '
            /page size of/            { ps = $8 }
            /Pages free/              { f = $3 }
            /Pages inactive/          { i = $3 }
            /Pages speculative/       { s = $3 }
            END { gsub(/\./, "", f); gsub(/\./, "", i); gsub(/\./, "", s)
                  print int((f + i + s) * ps / 1048576) }'
    else
        awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo
    fi
}

"$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    a=$(avail_mb)
    if [ "${a:-999999}" -lt "$MIN_AVAIL_MB" ]; then
        echo "guard: ${a}MB available < ${MIN_AVAIL_MB}MB, killing $pid" >&2
        kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        exit 137
    fi
    sleep "$GUARD_POLL"
done
wait "$pid"
