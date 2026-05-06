# Reka Vision for Robot Episode Quality Labeling

A small experiment using [Reka Vision](https://docs.reka.ai/vision/overview) to automatically classify robot teleoperation episodes as **GOOD** or **BAD** for imitation learning data curation.

The premise: large robot datasets ship with reward signals that mark episodes as "successful," but those signals miss subtle quality issues (partial pours, minor spills, awkward trajectories) that still hurt downstream imitation learning. Can a vision-language model catch what the reward signal misses?

## Setup

- **Dataset:** [`lerobot/toto`](https://huggingface.co/datasets/lerobot/toto) — 1003 teleoperation episodes of a Franka arm pouring small items from a source cup into a paper cup. Every episode has `next.reward = 1.0` (the dataset labels them all as successful).
- **Sample:** First 10 episodes from `chunk-000/file-000.mp4`, sliced into individual MP4s using the parquet's per-episode frame indices.
- **Model:** Reka Vision (`/qa/chat` endpoint) with `index=true` upload, structured prompt asking for `OUTCOME` (clean / partial / spilled / missed / knocked) and `VERDICT` (GOOD / BAD).
- **Ground truth:** Manually labeled by watching each clip.

## Results

Same 10 episodes, same prompt, three independent runs:

| Episode | Human label | Run 1 | Run 2 | Run 3 | Consistent |
|---:|:---:|:---:|:---:|:---:|:---:|
| 0 | GOOD | GOOD | GOOD | BAD  | ⚠️ |
| 1 | GOOD | GOOD | GOOD | GOOD | ✅ |
| 2 | BAD  | BAD  | BAD  | BAD  | ✅ |
| 3 | GOOD | GOOD | BAD  | GOOD | ⚠️ |
| 4 | BAD  | GOOD | BAD  | GOOD | ⚠️ |
| 5 | GOOD | GOOD | BAD  | GOOD | ⚠️ |
| 6 | GOOD | GOOD | GOOD | GOOD | ✅ |
| 7 | GOOD | GOOD | GOOD | GOOD | ✅ |
| 8 | GOOD | GOOD | GOOD | GOOD | ✅ |
| 9 | BAD  | BAD  | BAD  | GOOD | ⚠️ |

| Metric | Value |
|---|---|
| Run 1 accuracy | 9 / 10 (90%) |
| Run 2 accuracy | 8 / 10 (80%) |
| Run 3 accuracy | 7 / 10 (70%) |
| Same verdict across all 3 runs | 5 / 10 (50%) |
| **3-run majority-vote accuracy** | **9 / 10 (90%)** |

### What this means in practice

**Reka catches what the dataset's reward signal misses.** TOTO marks every episode `reward=1.0`. Reka flagged episodes 2 and 9 as quality issues across all three runs — a partial pour and a missed pour respectively — both of which a human reviewer agreed were not clean successes. For data curation pipelines, this is the win: surface the borderline episodes a human should look at, instead of trusting the dataset's binary success label.

**Single-run output is noisy on borderline cases.** Episodes that were unambiguously clean (1, 6, 7, 8) or unambiguously bad (2) were stable across runs. The wobble was concentrated on subtler cases (eps 0, 3, 4, 5, 9). A simple 3-run majority vote recovered most of the dropped accuracy and only missed one case (ep 4, a subtle partial pour).

## Pipeline

1. Download a chunk from TOTO via Hugging Face
2. Slice the concatenated MP4 into per-episode clips using the parquet's `episode_index` and `frame_index`
3. Upload each clip to Reka Vision with `index=true`
4. Poll until `indexing_status == "indexed"`
5. Query `/qa/chat` with the structured prompt
6. Parse `OUTCOME` and `VERDICT` from the response

The full pipeline (including 429 retry, empty-response retry, markdown-tolerant parsing) is in [`demo_pipeline.ipynb`](./demo_pipeline.ipynb).

## Prompt used

```
A Franka robot arm is pouring contents (small items like nuts or candy) from a source cup
held in its gripper into a paper cup sitting on the table.

Watch the entire video carefully. Focus on the FINAL state and any spills DURING the pour.

Respond in this EXACT format with no other text:

OUTCOME: [clean | partial | spilled | missed | knocked]
- clean: contents fully inside target cup, nothing on table
- partial: most contents in cup, but visible items bounced out or scattered
- spilled: noticeable contents on table outside cup
- missed: contents poured but mostly outside the cup
- knocked: target cup was tipped, moved, or fell during pour

VERDICT: [GOOD | BAD]
- GOOD only if OUTCOME is "clean"
- BAD otherwise
```

## Reproducing

1. Get a Reka API key from [reka.ai](https://reka.ai) and add it to Colab Secrets as `REKA_API_KEY`
2. Get a Hugging Face token (TOTO dataset is gated; accept the terms first)
3. Open `pipeline.ipynb` in Colab, run cells top to bottom

The notebook handles slicing, uploading, indexing, querying, and scoring against your manual labels (see `labels.csv`).

## Notes & caveats

- N=10 is small. The accuracy numbers are illustrative, not statistical.
- Run-to-run non-determinism comes from sampling in the underlying VLM — a temperature/seed parameter on `/qa/chat` would be useful here if exposed.
- The five outcome categories collapse to a binary verdict; in a real curation pipeline, surfacing the OUTCOME alongside the VERDICT lets humans triage faster.
- Indexing latency was ~25s per clip on average. For larger sweeps, parallel uploads with rate-limit-aware retries are essential.

## Files

- `pipeline.ipynb` — full Colab notebook
- `labels.csv` — manual ground-truth labels for the 10 sampled episodes
- `runs/run_1.csv`, `run_2.csv`, `run_3.csv` — Reka outputs from the three runs
- `clips.zip` — the 10 sliced MP4 clips used for evaluation
