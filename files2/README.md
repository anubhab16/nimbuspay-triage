# NimbusPay ticket triage: LoRA fine-tune

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Loaded in 4-bit?** No. The base is loaded in fp16 and the adapter is applied on top (`PeftModel.from_pretrained`). Please load it the same way.
- **Adapter:** `adapter/` (PEFT format, r=16, alpha=32, all linear projections, 18.5M parameters). Final run: `aug_lr5e-4` (cleaned data plus date-shift augmentation, lr 5e-4, 3 epochs).
- **Best result:** dev exact match 94.5%, mean field accuracy 98.9% (200 dev rows, best of 9 runs selected on dev, so optimistic; see `report.md`).
- **Decoding used for `predictions_test.jsonl`:** `model.generate(**inputs, do_sample=False, max_new_tokens=200)`, batch size 1, the base model's chat template with the system and user messages from the data, raw text out. Qwen's default generation config also sets `repetition_penalty=1.1` and `do_sample=True`; I set `do_sample=False` explicitly and left the penalty at its default. The same predictions were obtained at batch size 8 (0 of 400 rows differ).

## Files
`notebook.ipynb`, `report.md`, `cleaning_log.csv`, `predictions_test.jsonl`, `adapter/`, `experiments_log.json`.

## Reproducing
1. Put `train.jsonl`, `dev.jsonl`, `test_inputs.jsonl`, `score.py` in `MyDrive/vima3ya/data/` and use a Colab T4 runtime.
2. Optional: put `experiments_log.json` in `MyDrive/vima3ya/`. The notebook then shows the earlier experiments from the log and only retrains the final run (about 15 minutes).
3. Run `notebook.ipynb` from the top. Without the log it trains all 7 runs (about 15 minutes each) before the final predictions.
