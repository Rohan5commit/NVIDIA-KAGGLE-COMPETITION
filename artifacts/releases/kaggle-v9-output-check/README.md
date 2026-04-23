## Kaggle v9 Output Backup

This directory documents the GitHub release backup for the downloaded Kaggle
artifact folder `kaggle-v9-output-check`.

Release tag:

- `kaggle-v9-output-check-20260423`

Release assets:

- `kaggle-v9-output-check.tar.part-00`
- `kaggle-v9-output-check.tar.part-01`
- `kaggle-v9-output-check.tar.part-02`
- `kaggle-v9-output-check.tar.part-03`
- `kaggle-v9-output-check.tar.part-04`
- `kaggle-v9-output-check.tar.part-05`
- `kaggle-v9-output-check.tar.part-06`
- `kaggle-v9-output-check.tar.part-07`
- `kaggle-v9-output-check.tar.part-08`
- `SHA256SUMS.txt`

Restore:

```bash
cat kaggle-v9-output-check.tar.part-* > kaggle-v9-output-check.tar
shasum -a 256 -c SHA256SUMS.txt
tar -xf kaggle-v9-output-check.tar
```

Notes:

- This preserves the full downloaded `/tmp/kaggle-v9-output-check` folder.
- The folder is too large for normal git history, so it is stored as release
  assets instead of tracked repo files.
