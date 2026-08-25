import subprocess
import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent
TARGET = PLUGIN_DIR / "python_deps"
PACKAGES = ("transformers==4.52.1", "tokenizers==0.21.4")


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-deps",
        "--target",
        str(TARGET),
        *PACKAGES,
    ]
    print("Installing isolated IndexTTS 2.5 compatibility packages...")
    subprocess.check_call(command)
    print(f"Private dependencies installed in {TARGET}")


if __name__ == "__main__":
    main()
