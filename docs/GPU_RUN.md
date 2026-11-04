# Running Thread P on a rented GPU

Works on 1 to 8 GPUs. Aimed at about 2 hours on 4x RTX 4090. That estimate has not been checked on a GPU yet. The run prints its own,
from measured rates, before the pilot and again after it.

## Start

In a Jupyter terminal, with your own token and the commit you were given:

```
export GITHUB_TOKEN=<your token>
git -c credential.helper='!f(){ echo username=x-access-token; echo password=$GITHUB_TOKEN; };f' clone https://github.com/meta-lucky-builds/intr
cd intr && git checkout <commit> && bash scripts/gpu_run.sh
```

Do not put the token inside the clone URL. That stores it on disk. If it is already there, the run removes it from `origin`.

The run keeps going if the terminal closes. To watch again:

```
cd intr && bash scripts/gpu_run.sh --status
tail -f results/thread_p_gpu/run.log
```

## What it does, in order

1. Installs only what is missing. A preinstalled torch is left alone.
2. Preflight: fast tests, structural checks, and one tiny job on each GPU. It stops here if something is broken, before you spend much.
3. Measures how many jobs per GPU give the most total throughput, and prints an estimate.
4. Pilot: one seed per family, on validation data only, picks the smallest image size that passes Gate GP. Thresholds never change.
5. Main verdicts (H1p, H12p, H4p, H3p), then H5p and H14p, then the descriptive H16 and H17. A verdict table prints after each stage.
6. Packs results and pushes them to a new branch, `results/thread-p-<timestamp>`. It never pushes to main or the mission branch.

## Where results are

`results/thread_p_gpu/<run>/` holds everything: `summary.txt`, `summary.json`, `jobs/`, `logs/`, `manifest.json`.
`results/thread_p_gpu/thread_p_<run>.tar.gz` is the same as one file, without the rebuildable `cache/`.
The branch holds the same without `checkpoints/`, which are too large for a push.

If there is no token, or the push fails, the last lines say so and name the tar. Download it from the Jupyter file browser before stopping the machine.

## If something fails

| You see | What it means | What to do |
|---|---|---|
| `Fast tests FAILED` or `PREFLIGHT FAILED` | The code or machine is broken. Little money spent | Send the printed lines back. Do not rerun as is |
| `nvidia-smi sees a GPU, but this torch cannot use CUDA` | The image's torch is a CPU build | Use an image with a CUDA torch and rerun |
| `No GPU found` | The machine shows no GPU | Check `nvidia-smi`, or change machine |
| `FAMILY puck cannot pass Gate GP` | That family gets no verdict. This is a valid outcome, not a crash | Nothing. The run continues with the others |
| `Failed after one retry` for some jobs | Those jobs crashed. Others are fine | Let it finish. Rerunning the same command retries only those |
| The machine died or the terminal hung | Finished jobs are saved | Rerun the same command. It skips finished jobs |
| `Not pushed: ...` | The push failed. Results are safe | Download the tar it names |

Rerunning after a code change starts a new folder, so results of different code never mix.

## Test without a GPU

```
bash scripts/gpu_run.sh --smoke
```

Tiny settings, minutes on a CPU, numbers mean nothing.
