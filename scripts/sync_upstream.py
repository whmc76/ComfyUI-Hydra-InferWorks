"""Maintainer tool: vendor the pinned official IndexTTS 2.5 inference source."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


REPOSITORY = "index-tts/index-tts"
COMMIT = "b5ea881bec284b72f0b1cc04e0a724ff0c6b93e9"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "top_tts_vendor"
SKIPPED_SUFFIXES = (
    "/.ipynb_checkpoints/audio-checkpoint.py",
    "/.ipynb_checkpoints/commons-checkpoint.py",
    "/.ipynb_checkpoints/diffusion_transformer-checkpoint.py",
    "/.ipynb_checkpoints/flow_matching-checkpoint.py",
    "/.ipynb_checkpoints/length_regulator-checkpoint.py",
    "/.ipynb_checkpoints/model-checkpoint.py",
    "/facodec/modules/JDC/bst.t7",
)


def download(session: requests.Session, path: str) -> str:
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{path}"
    response = session.get(url, timeout=180)
    response.raise_for_status()
    relative = Path(path).relative_to("indextts")
    target = DESTINATION / "indextts" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return path


def rewrite_namespace() -> None:
    for path in (DESTINATION / "indextts").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        source = source.replace("from indextts", "from top_tts_vendor.indextts")
        source = source.replace("import indextts", "import top_tts_vendor.indextts")
        if path.name == "infer_v2_5.py":
            source = source.replace("os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'\n", "")
        path.write_text(source, encoding="utf-8", newline="\n")


def main() -> None:
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
    session = requests.Session()
    session.headers["User-Agent"] = "ComfyUI-Top-TTS-upstream-sync"
    response = session.get(tree_url, timeout=60)
    response.raise_for_status()
    files = [
        item["path"]
        for item in response.json()["tree"]
        if item["type"] == "blob"
        and item["path"].startswith("indextts/")
        and not item["path"].endswith(SKIPPED_SUFFIXES)
    ]

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download, session, path) for path in files]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            if index % 25 == 0 or index == len(futures):
                print(f"Downloaded {index}/{len(futures)} source files")

    (DESTINATION / "__init__.py").write_text("", encoding="utf-8")
    rewrite_namespace()

    license_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/LICENSE"
    license_response = session.get(license_url, timeout=60)
    license_response.raise_for_status()
    (ROOT / "UPSTREAM_MODEL_LICENSE.txt").write_bytes(license_response.content)
    print(f"Vendored IndexTTS source from {COMMIT}")


if __name__ == "__main__":
    main()
