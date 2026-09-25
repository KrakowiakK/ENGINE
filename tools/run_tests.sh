#!/usr/bin/env bash
# Unit tests, with the one piece of setup they cannot do themselves.
#
# MLX resolves its Metal library relative to the running executable, and an xctest bundle does not
# sit next to the engine's. Without this copy every test that touches an MLXArray dies at
# "Failed to load the default metallib" -- and, worse, xctest reports "0 tests passed", so the
# suite looks GREEN while running nothing. Hence the check at the end.
#
#   tools/run_tests.sh [--filter PrefixStoreTests]
set -euo pipefail
cd "$(dirname "$0")/.."
# H60: the full suite is assumed to need ~270 GB (an unmeasured estimate; the attribution to NGramStorageTests is
# unverified -- the only measured growth was XCTest autorelease accumulation in IncrementalDetokenizerTests, fixed with an
# autoreleasepool). Next to a serving engine (~237 GB wired) that could reach ~510 GB of 512 and get either process killed. Refuse while an engine runs unless free memory covers it or ENGINE_TESTS_ALLOW_WITH_ENGINE=1.
if pgrep -x engine >/dev/null && [ "${ENGINE_TESTS_ALLOW_WITH_ENGINE:-0}" != "1" ]; then
  FREE_GB=$(vm_stat | awk '/Pages free|Pages inactive|Pages speculative/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*16384/1e9}')
  if [ "$FREE_GB" -lt 300 ]; then
    echo "REFUSED: an engine is running and only ${FREE_GB} GB are free; the suite may need ~270 GB (unmeasured estimate)." >&2
    echo "  stop the engine, or run a light filter (e.g. --filter ServeHardeningTests) with ENGINE_TESTS_ALLOW_WITH_ENGINE=1" >&2
    exit 4
  fi
fi
METALLIB=.build/release/mlx.metallib
[ -f "$METALLIB" ] || { echo "no $METALLIB -- run: swift build -c release --product engine && tools/build-mlx-metallib.sh" >&2; exit 2; }

swift build --build-tests >/dev/null
BUNDLE=$(find .build -name "*PackageTests.xctest" -maxdepth 3 | head -1)
[ -n "$BUNDLE" ] || { echo "no test bundle built" >&2; exit 2; }
cp "$METALLIB" "$BUNDLE/Contents/MacOS/"

OUT=$(swift test "$@" 2>&1) || { echo "$OUT" | grep -E "error:|failed" | head -20; exit 1; }
echo "$OUT" | grep -E "Test Case .*(passed|failed)|Test Suite .*(passed|failed)" || true
RAN=$(echo "$OUT" | grep -c "Test Case .* passed" || true)
[ "$RAN" -gt 0 ] || { echo "REFUSED: the suite reported success without running a single test" >&2; exit 3; }
echo "$RAN test case(s) passed"
