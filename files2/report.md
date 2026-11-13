# Report: NimbusPay ticket triage, LoRA fine-tune

**Result.** Qwen2.5-1.5B-Instruct with a 16-bit LoRA (r=16, 18.5M trainable parameters) reaches **94.5% exact match and 98.9% mean field accuracy on dev** (final run `aug_lr5e-4`). This is optimistic: it is the best of 9 runs picked on the same 200 dev rows, and the 95% interval for one run is about 90 to 97%. I expect the harder hidden test to score lower.

## 1. Data audit (Part A)

**Method.** Labels are deterministic functions of the text, so I wrote a rule-based reference labeller from SCHEMA.md and calibrated it on dev first (**0 disagreements on 200 rows**). Disagreements on train are then candidate label errors. Category is not rule-derivable, so I checked it with a 5-fold cross-validated TF-IDF classifier. The labeller repairs training data only and is never used at inference.

| Problem | Rows | How found | Action |
|---|---|---|---|
| Junk text (`test`, `asdf asdf`, empty) with a full made-up label | 10 | body under 12 chars or junk token, yet label has amount and txn_id | drop |
| Exact copy of a dev ticket | 30 | identical `user` message in dev | drop |
| Exact duplicate inside train | 60 | same `user` message as an earlier row | drop later copy |
| Truncated JSON label | 25 | `json.loads` fails | rebuilt from surviving prefix plus rules, kept only if it agrees with every surviving field (all did) |
| v1 schema (`prio`, `dd/mm/yyyy`) | 70 | key set is not the 8 v2 keys | renamed, dates to ISO, priority re-derived |
| Amount in rupees, not paise | 25 | label is exactly 1/100 of the rule value | set to rule value |
| txn_id not normalised | 20 | not `NP` + 10 digits | normalised |
| Priority breaks SCHEMA rules | 60 (+70 v1 rows) | re-derived from category, amount, legal flag | set to rule value |
| `needs_human` inconsistent | 45 found, 0 edited | 43 vanish once priority is fixed, 2 once their rupee amount is fixed | none |

**Result:** 1,445 rows in, **1,345 out** (100 dropped, 200 repaired), all logged in `cleaning_log.csv`. Priority was flipped on about 60 rows while `needs_human` kept its true value, so priority was the corrupted field. The cleaner flags 0 rows on clean dev. The one remaining disagreement with the labeller is `tr-00171` (amount and a limit share a sentence); the label is right, so I kept it.

**Negative result.** The category classifier reached about 98% cross-validated accuracy and its confident disagreements were defensible, so I found no category corruption (it cannot see noise at the 1 to 2 percent level).

**Observations.** About 2.7% of tickets (38 train, 3 dev, 11 test) are 3,000 to 9,000 character quoted threads with decoy numbers and dates; the rest are under 350 characters. Every real label is dated 2026, so the SCHEMA year-rollover rule is never exercised.

**Disclosure.** I wrote the labeller's regexes after reading train and dev and used dev to validate the auditor; a few small rules (for example any "card" means `card`) were chosen to match dev. No dev row was used for training.

## 2. Fine-tuning setup (Part C)

- **Stack:** `transformers` + `peft` + `bitsandbytes` with my own loop, so I control which tokens carry loss.
- **Template and loss:** the model's ChatML template with the system role. Loss only on the assistant JSON plus the end-of-turn token (so it learns to stop).
- **Sequence length:** the label is at the end, so right-truncation would delete it. `MAX_LEN` is the longest training example rounded up to a multiple of 64 (printed in the notebook), so long threads are trained whole. Batches are packed under a token budget and the output head runs only at label positions, because the vocabulary is 152k.
- **LoRA:** r=16, alpha=32, dropout 0.05, all seven linear projections, 18.5M parameters (asserted under 50M). fp16 autocast with fp32 adapter weights (a T4 has no bf16), AdamW, cosine schedule, about 16 examples per step, 3 epochs, peak 8.9 GB.

## 3. Baselines and experiments (Parts B and D)

One seed per run, strict scoring, one dev row = 0.5 points. Probe = txn_date accuracy on 86 re-dated dev tickets.

| Run | What changed | Dev EM % | Dev field % | EM excl. 30 leaked % | Probe % | Peak GB | Min |
|---|---|---|---|---|---|---|---|
| base_fixed_prompt | no fine-tuning | 0.0 | 0.0 | 0.0 | | 6.0 | |
| base_rules_prompt | rules in prompt | 0.0 | 48.9 | 0.0 | | 6.0 | |
| ref | cleaned, 16-bit, r16, lr 2e-4 | 87.5 | 97.6 | 87.6 | 57.0 | 8.9 | 13.7 |
| raw | raw train.jsonl | 82.0 | 96.8 | 80.0 | | 8.9 | 14.5 |
| qlora | 4-bit NF4 | 79.0 | 96.0 | 78.8 | | 8.3 | 16.7 |
| lr_1e-4 | lr 1e-4 | 61.5 | 93.1 | 62.4 | | 8.9 | 13.7 |
| lr_5e-4 | lr 5e-4 | 93.5 | 98.6 | 92.9 | 58.1 | 8.9 | 13.6 |
| aug_dates | + date-shift augmentation | 90.0 | 98.2 | 90.0 | 94.2 | 8.9 | 16.1 |
| aug_lr5e-4 | augmentation + lr 5e-4 | **94.5** | **98.9** | 94.1 | **98.8** | 8.9 | 15.4 |

Paired against `ref` (right where ref is wrong / wrong where ref is right; exact sign test): raw 7/18 (p=0.043), qlora 3/20 (p=0.0005), lr_1e-4 3/55 (p<0.0001), lr_5e-4 17/5 (p=0.017), aug_dates 12/7 (p=0.36), aug_lr5e-4 18/4 (p=0.004).

- **Baselines.** Both score 0 exact matches: the fixed prompt makes the base model invent its own JSON, and with the rules in the prompt it fills many keys with `null`.
- **Cleaning helps.** Raw costs 5.5 points, or 7.6 once the 30 leaked dev tickets are excluded. p=0.043 is borderline, so I call it supported, not proven.
- **QLoRA is worse and saves little.** 8.5 points lower, 0.6 GB less memory and slower to train. Tested at lr 2e-4 only.
- **Learning rate matters most.** 1e-4 undertrains (final loss 0.0135 vs 0.0055). 5e-4 clearly beats 2e-4. I did not try higher, so the optimum is unknown.
- **Augmentation.** The dev gain is noise (90.0 vs 87.5). The probe shows it: `lr_5e-4` alone scores 58.1%, the same as `ref`, while augmentation gives 94.2% and 98.8%. So the augmentation, not the learning rate, taught the date arithmetic. The probe is synthetic and built with the same rules as the augmentation, so it favours it by construction; 31 of its 86 rows are year-rollover cases absent from real train and dev labels, and I do not know whether the test has any. The augmentation costs nothing on dev, so it is cheap insurance, not demonstrated benefit.
- **Final choice.** I fixed the rule in advance: use the combination if it reached 92% on dev. It reached 94.5%. Against `lr_5e-4` alone the gap is 2 rows, which is noise.
- **Decoding.** Qwen's default generation config has `repetition_penalty=1.1`. I generated with `do_sample=False` and the default penalty, as plain `generate` would. On the first 60 dev rows the default gave 88.3% and penalty 1.0 gave 90.0%, a difference of one row (`dv-00056`). Batch size 8 and 1 gave identical test outputs on all 400 rows, and a 20-row regeneration matched 20/20.

## 4. Error analysis (Part E, final model, dev)

11 of 200 rows are wrong. Per field: category 97.5%, priority 96.5%, amount 99.5%, txn_date 100%, txn_id 99.5%, channel 100%, language 100%, needs_human 98.0%. By category (exact): fraud 86.7% (n=15), payment_failed 90.6% (n=53), other 93.8% (n=16), refund 94.5% (n=55), 100% for kyc, offers and account_access (the first three names were reconstructed from the failure list). All 4 `needs_human` errors cascade from a priority error. The 3 long threads in dev were all correct, too few to conclude anything.

**A. Threshold errors, category and amount correct (4 rows).** The model reads the amount but applies the 5k or 50k rule wrongly: `dv-00005` (₹49,680, gold P2, predicted P1), `dv-00056` (₹44,206.75, gold P2, predicted P1), `dv-00127` (₹4,836, gold P3, predicted P2), `dv-00037` ("284669 rupees", gold P1, predicted P2). Hypothesis: comparing a multi-digit number to a threshold is hard for a 1.5B model that sees digits as separate tokens. I did not count near-threshold examples in train.

**B. Category confusions (5 rows).** Three involve fraud, the costly boundary because it forces P1. `dv-00157` (Hinglish with typos, money taken after an OTP request) was predicted `account_access`, pulled by the word OTP. `dv-00046` ("a payment on my statement is wrong") is vague and was predicted `refund`. `dv-00010` and `dv-00177` (a top-up that did not arrive; "a red flag on my top up") were predicted `fraud` instead of `payment_failed`, and the second text is genuinely ambiguous. `dv-00042` (tracking a card on its way, gold `other`) was predicted `refund` for a reason I cannot identify.

**C. Digit-copy slips (2 rows).** `dv-00017`: 18082725 became 18082525 (from "Rs 1,80,827.25"). `dv-00145`: `np 18526 61784` became `NP1852661778`. I first suspected the repetition penalty (it penalises tokens already in the prompt, including digits) and tested it: rerunning the 8 failing rows of groups A to C with `repetition_penalty=1.0` fixed one row, `dv-00056` (a threshold error), and neither digit-copy row. So the penalty does not explain group C and I do not know the cause. Both are long digit strings (spaced or with Indian grouping), which may simply be hard to copy exactly, but I have not tested that.

**With one more week:** (1) amount-shift augmentation across the 5k and 50k thresholds with labels recomputed from the SCHEMA, targeting group A (untested); (2) fraud-versus-other hard negatives from train for group B; (3) 3 seeds for the best two configurations, plus lr 1e-3 and r=32, since 5e-4 was the edge of what I tested; (4) a held-out threshold probe like the date one.

## 5. AI tools used

I used Claude (Anthropic) extensively. Claude drafted the audit code, the training notebook, the experiment design and the first version of this report, and helped me read the results. I ran every cell on Colab, and I am responsible for all code and numbers here. No model or API produced labels or predictions. The reference labeller is plain regex code written from SCHEMA.md.
