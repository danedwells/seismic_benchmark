#!/usr/bin/env bash
# Re-run build_station_availability.py until it exits cleanly.
# Safe because the script skips already-cached events on each restart.

SCRIPT="$(dirname "$0")/build_station_availability.py"
ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "=== Attempt $ATTEMPT ==="
    python "$SCRIPT"
    CODE=$?
    if [ $CODE -eq 0 ]; then
        echo "Done after $ATTEMPT attempt(s)."
        break
    fi
    echo "Exited with code $CODE — restarting in 2s..."
    sleep 2
done
