#!/usr/bin/env bash
# Run a shell command on the VM through `colab console` (bypasses the Jupyter kernel, so it works while a
# keepalive cell is running). The output is cut between two markers printed with printf; the echoed (and
# terminal-wrapped) command line only holds the format string, never the marker itself, so the markers are matched
# anywhere in a line. Retries (up to 3 times) only when the command never started.
# Usage: bash colab/vmsh.sh '<command>' [session]
CMD="$1"; S="${2:-expert}"
cut_out='{ if (index($0, "__VMSH_END__")) { s = substr($0, 1, index($0, "__VMSH_END__") - 1); if (f && s != "") print s; exit }
          if (f) print
          if (index($0, "__VMSH_BEGIN__")) { f = 1; s = substr($0, index($0, "__VMSH_BEGIN__") + 14); if (s != "") print s } }'
for try in 1 2 3; do
  out=$(printf '%s\n' "printf '__VMSH_%s__\n' BEGIN; $CMD; printf '\n__VMSH_%s__\n' END" \
    | timeout 120 bash "$(dirname "$0")/c.sh" console -s "$S" 2>&1 | tr -d '\r' \
    | perl -pe 's/\e\[[0-9;?>]*[a-zA-Z]//g; s/\e\].*?(?:\a|\e\x5c)//g; s/\e[()][A-Z0-9]//g')
  if echo "$out" | grep -q "__VMSH_END__"; then
    echo "$out" | awk "$cut_out"
    exit 0
  fi
  # the command started but its end was not seen: print what came back, never run it a second time
  echo "$out" | grep -q "__VMSH_BEGIN__" && { echo "$out" | awk "$cut_out"; exit 1; }
  sleep 3
done
exit 1
