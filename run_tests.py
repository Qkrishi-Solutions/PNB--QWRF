from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

TEST_SUITES = [
    "address_expansion/tests",
    "financial_data_validation/tests",
    "ml_ready_dataset/tests",
    "synthetic_portfolio/tests",
    "synthetic_training_dataset/tests",
]

def main():
    for suite in TEST_SUITES:
        print(f"\n{'=' * 60}\nRunning: {suite}\n{'=' * 60}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(ROOT / suite)],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"\nFAILED: {suite}")
            return result.returncode

    print("\nALL TEST SUITES PASSED.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
