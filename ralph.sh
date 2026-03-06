#!/bin/bash
LOGDIR="ralph_logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/ralph_$(date +%Y%m%d_%H%M%S).log"

count=0
failures=0
MAX_CONSECUTIVE_FAILURES=3
REFLECT_EVERY=5

total_input_tokens=0
total_output_tokens=0

echo "Logging to $LOGFILE"

run_claude() {
    local label="$1"
    local input_file="$2"

    raw=$(cat "$input_file" | claude --model sonnet --dangerously-skip-permissions -p --output-format json 2>&1)
    exit_code=$?

    # Extract text result and token usage
    result=$(echo "$raw" | jq -r '.result // .message // "no result"')
    input_tok=$(echo "$raw" | jq -r '.usage.input_tokens // 0')
    output_tok=$(echo "$raw" | jq -r '.usage.output_tokens // 0')
    total_tok=$(echo "$raw" | jq -r '(.usage.input_tokens // 0) + (.usage.output_tokens // 0)')

    total_input_tokens=$((total_input_tokens + input_tok))
    total_output_tokens=$((total_output_tokens + output_tok))

    echo "$result" | tee -a "$LOGFILE"
    echo "  [tokens] input=${input_tok} output=${output_tok} total=${total_tok} | session_total=$((total_input_tokens + total_output_tokens))" | tee -a "$LOGFILE"

    return $exit_code
}

while true; do
    count=$((count + 1))
    start_ts=$(date +%s)

    if [ $((count % REFLECT_EVERY)) -eq 0 ]; then
        echo "=== Loop #${count} [REFLECT] | $(date) ===" | tee -a "$LOGFILE"
        run_claude "reflect" reflect.md
    else
        echo "=== Loop #${count} | $(date) ===" | tee -a "$LOGFILE"
        run_claude "prompt" prompt.md
    fi
    exit_code=$?

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "--- Loop #${count} finished | exit=$exit_code | ${elapsed}s elapsed | cumulative tokens: input=${total_input_tokens} output=${total_output_tokens} ---" | tee -a "$LOGFILE"

    if tail -20 "$LOGFILE" | grep -qiE "you've hit your limit|hit your limit|rate.limit|resets [0-9]+(am|pm)|account.*limit|usage.*limit|quota.*exceeded"; then
        WAIT_HOURS=3
        echo "*** RATE LIMIT DETECTED — sleeping ${WAIT_HOURS} hours until $(date -d "+${WAIT_HOURS} hours") ***" | tee -a "$LOGFILE"
        sleep $((WAIT_HOURS * 3600))
        failures=0
        continue
    fi

    if [ "$exit_code" -ne 0 ]; then
        failures=$((failures + 1))
        echo "WARNING: failure $failures/$MAX_CONSECUTIVE_FAILURES" | tee -a "$LOGFILE"
        if [ "$failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            echo "ERROR: $MAX_CONSECUTIVE_FAILURES consecutive failures, stopping." | tee -a "$LOGFILE"
            exit 1
        fi
    else
        failures=0
    fi
done