# Running the pixel experiment

Thread P can run on free GitHub Actions CPU runners or on a machine with 1 to 8 GPUs. The pendulum results in this repo came from
Actions (45 jobs, about 80 minutes). The GPU path passes its tests and smoke run on cloud CPU, but it has never run on a real GPU.

Pendulum is the only family that passed Gate GP, so it is the only one that can give verdicts.

## On GitHub Actions

Open the Actions tab, choose `pixel-pendulum`, and run it with mode `full`. Results are in the `thread-p-pendulum-results` artifact.
"Re-run failed jobs" repeats no finished work.

## On a GPU machine

The repo is public, so no token is needed:

```bash
git clone https://github.com/meta-lucky-builds/world-model-adaptation && cd world-model-adaptation && bash scripts/gpu_run.sh --families pendulum
```

The run keeps going if the terminal closes. In a notebook, start it from a terminal, because the script watches the log until the run ends. To watch again:

```bash
bash scripts/gpu_run.sh --status
tail -f results/thread_p_gpu/run.log
```

## What it does, in order

1. Installs only what is missing. A preinstalled torch is left alone.
2. Preflight: fast tests, structural checks, and one tiny job on each GPU. It stops here if something is broken.
3. Measures how many jobs per GPU give the most throughput, and prints a time estimate.
4. Pilot: one seed per family, on validation data only, picks the smallest image size that passes Gate GP. Thresholds never change.
5. Main verdicts (H1p, H12p, H4p, H3p), then H5p and H14p, then the descriptive H16 and H17. A verdict table prints after each stage.
6. Packs the results. It pushes them to a new `results/thread-p-<timestamp>` branch only if `GITHUB_TOKEN` is set.

## Where results are

`results/thread_p_gpu/<run>/` holds `summary.txt`, `summary.json`, `jobs/`, `logs/`, and `manifest.json`.
`results/thread_p_gpu/thread_p_<run>.tar.gz` is the same as one file, without the rebuildable `cache/`. Without a token, download it before stopping the machine.

## If something fails

| You see | What it means | What to do |
|---|---|---|
| `Fast tests FAILED` or `PREFLIGHT FAILED` | The code or machine is broken | Read the printed lines. Do not rerun as is |
| `nvidia-smi sees a GPU, but this torch cannot use CUDA` | The image's torch is a CPU build | Use an image with a CUDA torch and rerun |
| `No GPU found` | The machine shows no GPU | Check `nvidia-smi`, or change machine |
| `FAMILY puck cannot pass Gate GP` | That family gets no verdict. This is a valid outcome | Nothing. The run continues with the others |
| `Failed after one retry` for some jobs | Those jobs crashed. Others are fine | Let it finish. Rerunning the same command retries only those |
| The machine died or the terminal hung | Finished jobs are saved | Rerun the same command. It skips finished jobs |
| `Not pushed: ...` | No token, or the push failed. Results are safe | Download the tar it names |

Rerunning after a code change starts a new folder, so results of different code never mix.

## Test without a GPU

```bash
bash scripts/gpu_run.sh --smoke
```

Tiny settings, minutes on a CPU, numbers mean nothing.
