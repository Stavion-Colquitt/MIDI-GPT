"""Create Stavion-Colquitt/MIDI-GPT-LM2 and upload hf_export/. Uses the huggingface_hub login already on this PC.

    py -3.12 hf_push.py            # private repo (default)
    py -3.12 hf_push.py --public   # public repo
"""
import os, sys
from huggingface_hub import HfApi
api = HfApi(); repo = "Stavion-Colquitt/MIDI-GPT-LM2"; public = "--public" in sys.argv
api.create_repo(repo, repo_type="model", private=not public, exist_ok=True)
if public: api.update_repo_settings(repo, private=False)
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_export")
info = api.upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                         commit_message="MIDI GPT (LM 2): weights, tokenizer, code, card and example MIDI (val ppl 3.69)")
print("uploaded:", info.commit_url if hasattr(info, "commit_url") else info)
print("private" if not public else "public", sorted(f for f in api.list_repo_files(repo)))
