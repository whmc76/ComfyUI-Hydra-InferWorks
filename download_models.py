import argparse
import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = PLUGIN_DIR.parents[1]
DEFAULT_MODEL_DIR = COMFYUI_DIR / "models" / "IndexTTS-2.5"


def download_from_huggingface(model_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="IndexTeam/IndexTTS-2.5", local_dir=str(model_dir))


def download_from_modelscope(model_dir: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(model_id="IndexTeam/IndexTTS-2.5", local_dir=str(model_dir))


def download_auxiliary_models(model_dir: Path) -> None:
    if str(PLUGIN_DIR) not in sys.path:
        sys.path.insert(0, str(PLUGIN_DIR))
    from top_tts_vendor.indextts.utils.model_download import ensure_models_available

    ensure_models_available(str(model_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official IndexTTS 2.5 weights for local ComfyUI inference.")
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm acceptance of the upstream bilibili Model Use License Agreement.",
    )
    args = parser.parse_args()
    if not args.accept_license:
        parser.error("Read UPSTREAM_MODEL_LICENSE.txt, then rerun with --accept-license")

    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "modelscope":
        download_from_modelscope(model_dir)
    else:
        download_from_huggingface(model_dir)
    download_auxiliary_models(model_dir)
    print(f"IndexTTS 2.5 local model is ready at: {model_dir}")


if __name__ == "__main__":
    main()
