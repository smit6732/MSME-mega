"""
scripts/download_model.py

The ONE-TIME setup step mentioned in the README. Run this once, after
installing requirements.txt, to fetch the local LLM's weights:

    python scripts/download_model.py

What it does: downloads a single GGUF model file from Hugging Face Hub
and caches it in the local models/ folder. That's it -- it doesn't
install anything, doesn't touch the running app, and never runs
automatically.

Why this has to be a separate, explicit step instead of something that
happens on first app launch: a language model, even a small one, is a
few hundred MB. That can't ship as plain source code in this repo, and
silently downloading a few hundred MB the first time someone runs
`streamlit run app.py` would be a bad surprise and would violate the
"the running app needs no internet access" promise. So: download once,
here, deliberately -- and after that, app.py (via core/llm_phrasing.py)
only ever reads the file that's already on disk. It never makes a
network request itself.

If this script is never run, or fails (no internet, Hugging Face
unreachable), the app still works completely normally -- it just falls
back to the template-based fix-list sentences everywhere, silently and
automatically. See core/llm_phrasing.py's module docstring for how that
fallback is implemented.
"""

import os
import sys

# Must match core/llm_phrasing.py's MODEL_PATH exactly -- both files
# compute it the same way (relative to the repo root) so there's no
# risk of the download script and the app looking in different places.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIRECTORY = os.path.join(REPO_ROOT, "models")

# Qwen2.5-0.5B-Instruct, Q4_K_M quantization. See README.md ("Local AI
# model") for why this specific model + quantization was chosen.
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def main():
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    destination_path = os.path.join(MODEL_DIRECTORY, MODEL_FILENAME)

    if os.path.isfile(destination_path):
        print(f"Model already present at {destination_path} -- nothing to do.")
        print("Delete that file first if you want to re-download it.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub isn't installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Downloading {MODEL_FILENAME} from {HUGGINGFACE_REPO_ID} ...")
    print("This happens ONCE. After this, the app runs fully offline.")

    downloaded_path = hf_hub_download(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DIRECTORY,
    )

    print(f"Done. Model saved to: {downloaded_path}")
    print("You can now run `streamlit run app.py` -- no internet needed from here on.")


if __name__ == "__main__":
    main()
