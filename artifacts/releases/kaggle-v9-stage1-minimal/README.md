## Kaggle v9 Stage 1 Minimal Backup

This directory documents the minimal GitHub backup needed to continue from the
successful `v9` Stage 1 state after Kaggle limits reset.

Release tag:

- `kaggle-v9-stage1-minimal-20260423`

Release assets:

- `kaggle-v9-stage1-minimal.tar`
- `SHA256SUMS.txt`

Contents inside the tarball:

- `stage1_sft/adapter_model.safetensors`
- `stage1_sft/adapter_config.json`
- `stage1_sft/README.md`
- `stage1_sft/chat_template.jinja`
- `stage1_sft/tokenizer.json`
- `stage1_sft/tokenizer_config.json`
- `stage1_sft/training_args.bin`
- `artifacts/stage1_sft/training_summary.json`
- `artifacts/stage1_sft/dataset_summary.json`

Restore:

```bash
shasum -a 256 -c SHA256SUMS.txt
tar -xf kaggle-v9-stage1-minimal.tar
```

Why this is enough:

- Stage 1 SFT completed successfully in `v9`.
- Stage 2 GRPO or packaging can start from the final Stage 1 adapter; the full
  `15G` working directory and intermediate checkpoints are not required.
