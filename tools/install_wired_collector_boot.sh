#!/usr/bin/env bash
# Keep macOS's GPU wired collector off across reboots (P122/P123): installs a LaunchDaemon that runs
# `sysctl iogpu.disable_wired_collector=1` at every boot, and runs it once now. Needs sudo once; nothing else is changed.
#
#   sudo tools/install_wired_collector_boot.sh             install (or reinstall) and apply now
#   sudo tools/install_wired_collector_boot.sh --uninstall remove the daemon (the value stays until the next reboot)
#
# Without it the collector unwires the model's memory (prefill 352-941 instead of 1087-1267 tok/s and decode stalls in
# llm_context_benchmarks on an M3 Ultra, OBS-ENG-220). The setting only keeps GPU memory that is already wired from being
# unwired; it changes no file, no model output, and is undone by --uninstall plus a reboot.
set -euo pipefail
LABEL=com.engine.iogpu-disable-wired-collector
PLIST=/Library/LaunchDaemons/$LABEL.plist
if [[ $EUID -ne 0 ]]; then echo "run with sudo" >&2; exit 2; fi
if [[ "${1:-}" == --uninstall ]]; then
  launchctl bootout system "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $PLIST (iogpu.disable_wired_collector stays $(sysctl -n iogpu.disable_wired_collector) until the next reboot)"
  exit 0
fi
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/usr/sbin/sysctl</string><string>iogpu.disable_wired_collector=1</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
EOF
chown root:wheel "$PLIST"
chmod 644 "$PLIST"
launchctl bootout system "$PLIST" 2>/dev/null || true
launchctl bootstrap system "$PLIST"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [[ "$(sysctl -n iogpu.disable_wired_collector)" == 1 ]] && break
  sleep 0.5
done
v=$(sysctl -n iogpu.disable_wired_collector)
echo "installed $PLIST; iogpu.disable_wired_collector=$v (applied by the daemon now and at every boot)"
[[ "$v" == 1 ]]
