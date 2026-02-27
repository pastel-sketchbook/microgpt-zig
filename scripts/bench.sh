#!/usr/bin/env bash
set -euo pipefail

bin=./zig-out/bin/microgpt-zig
best=""

for i in 1 2 3; do
  out=$(/usr/bin/time -l "$bin" 2>&1)
  timeln=$(echo "$out" | grep 'real')
  wall=$(echo "$timeln" | awk '{print $1}')
  user=$(echo "$timeln" | awk '{print $3}')
  sys=$(echo "$timeln" | awk '{print $5}')
  rss=$(echo "$out" | grep 'maximum resident' | awk '{print $1}')
  rss_mb=$(echo "scale=1; $rss / 1048576" | bc)
  instr=$(echo "$out" | grep 'instructions retired' | awk '{print $1}')
  instr_b=$(echo "scale=1; $instr / 1000000000" | bc)
  loss=$(echo "$out" | grep 'step 1000' | sed 's/.*loss //')
  s1=$(echo "$out" | grep 'sample  1:' | sed 's/.*: //')
  echo "run $i: ${wall}s wall, ${user}s user, ${sys}s sys, ${rss_mb}MB RSS, ${instr_b}B instr | loss=$loss sample1=$s1"
  if [ -z "$best" ] || [ "$(echo "$wall < $best" | bc)" -eq 1 ]; then
    best=$wall
  fi
done

echo ""
echo "best: ${best}s"
