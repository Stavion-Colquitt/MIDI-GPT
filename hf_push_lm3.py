"""Create Stavion-Colquitt/MIDI-GPT-LM3 (private) and upload hf_export_lm3/. Same as hf_push.py for LM 2.

The repo stays private on purpose: Stavion makes it public himself, together with the GitHub release.
The generated/ folder (local test output of generate.py) is not uploaded.

    py -3.12 hf_push_lm3.py
"""
import os
from huggingface_hub import HfApi
api = HfApi(); repo = "Stavion-Colquitt/MIDI-GPT-LM3"
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_export_lm3")
info = api.upload_folder(folder_path=d, repo_id=repo, repo_type="model", ignore_patterns=["generated/*", "generated/**", "**/__pycache__/**"],
                         commit_message="MIDI GPT (LM 3): controllable weights, tokenizer, code, card and example MIDI (val loss 1.029)")
print("uploaded:", info.commit_url if hasattr(info, "commit_url") else info)
stale = [f for f in api.list_repo_files(repo) if "__pycache__" in f]
if stale:
    api.delete_files(repo_id=repo, repo_type="model", delete_patterns=stale,
                     commit_message="remove local __pycache__ files")
    print("deleted stale:", stale)
print("repo is", "PRIVATE" if api.model_info(repo).private else "PUBLIC")
print(sorted(api.list_repo_files(repo)))
