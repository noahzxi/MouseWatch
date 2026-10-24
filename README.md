# MouseWatch

MouseWatch is a Windows program that records pointer movement and compares it with one enrolled person. A small always-on-top window reports whether the current movement matches that profile, with a match percent. Closing the window stops collection.

The running check does not lock the workstation. `desktop/mousewatch/lock/workstation.py` is empty, and the app never calls `LockWorkStation`. On a profile that has no personal template, the window can still show the words "checking" or "mismatch held". Those words describe the score. They do not lock the machine.

The public dataset is [Mousetrack](https://data.mendeley.com/datasets/3rfsbcmgfw/1) (Mendeley, 1,755 users, 31,990,875 actions). The population model in this repo was trained on that file. The live aim-drill check was tuned on a five-minute Human Benchmark aim recording from one desktop user.

[Watch a quick demo!](https://www.youtube.com/watch?v=pH1crN75Nss}

## Quick start

From a new PowerShell window, score live movement against the saved five-minute aim profile for user `100000`:

```powershell
cd "C:\Users\Noah\Programming Projects\MouseWatch\desktop"
python -m mousewatch --no-tray --output test.csv --profile ..\models\aim-master.joblib
```

The first lines should say the app is comparing movement with that profile and user `100000`. The window starts at "Need more movement". The first match appears after 40 actions, then updates every 5. Do the same kind of aim drill the profile was built from. Stop with Ctrl+C.

Only one collector can run at a time. If the process prints `MouseWatch is already running.`, close the existing window first.

`test.csv` is a log of this session. It is not the enrollment. `--user-id` only labels new rows in that log. The verdict uses `owner_id` stored inside the profile.

## What the window means

With the aim profile, the headline and the match percent come from the personal template:

| Match | Headline | Color |
|---|---|---|
| 70% or more | This is you | green |
| 50% to 70% | Not sure | amber |
| below 50% | This is not you | red |
| no score yet | Need more movement | gray |

"This is not you" requires two scores. A single score below 70% stays on "Not sure". The percent on screen is the median of the last 21 scores, so one odd patch of targets does not flip the verdict. A gap longer than 60 seconds starts a new session and clears those scores.

The number is a distance from the enrolled person's own windows. Higher match means closer to that person. It is not a probability and it is not a vote among the 1,697 population classes.

## How a recording is made

Collection runs on Windows only. The listener uses Raw Input on a hidden window so a slow collector does not stall the pointer. Distances come from `GetCursorPos`, in screen pixels, after Windows pointer speed and acceleration. The process turns on DPI awareness.

Movement is cut into slices:

- A slice closes after about 130 ms of continuous movement (`--segment-ms`, default 130).
- A pause of 100 ms also closes the open slice (`--pause-ms`, default 100).
- A button press closes the open slice. Releasing the button writes a separate click row.
- The angle is `atan(dy/dx)` in degrees, rounded and clamped to −90–90. It is the slope of that slice, with y positive downward. It is not a compass heading.
- Distance is the straight-line length of the slice in pixels. Velocity is that distance divided by the slice duration, in pixels per second.
- A stroke shorter than 4 pixels (`--min-distance`, default 4) is left out of the dataset CSV. It is still written to the motion sidecar and still fed to the live check.
- Scroll-wheel rows stay in the sidecar only.

Rows are appended. The writer flushes every 50 rows, and also about every 2 seconds while the pointer is still. A log is written beside the CSV as `collector.log`.

In the public Mousetrack file, a row with `duration` set and the movement columns `NULL` is a pause between actions. The collector uses that same shape for a click: `duration` is how long the button was held, in milliseconds, and `angle`, `distance`, and `velocity` are `NULL`. The loader accepts either an empty field or the text `NULL`.

## Dataset CSV

Semicolon-separated, UTF-8. Header:

```text
idman;duration;angle;distance;velocity;local_date;local_time
```

| Column | Stroke | Click |
|---|---|---|
| `idman` | user id | user id |
| `duration` | `NULL` | hold time in milliseconds |
| `angle` | degrees, −90–90 | `NULL` |
| `distance` | straight-line pixels | `NULL` |
| `velocity` | pixels per second | `NULL` |
| `local_date` | `YYYY-MM-DD` | `YYYY-MM-DD` |
| `local_time` | `HH:MM:SS.fffffff` | `HH:MM:SS.fffffff` |

Times are local. The fractional part has 7 digits. Internally, time is stored as 100-nanosecond ticks (`10_000_000` ticks per second).

## Motion sidecar

Every slice, including short strokes and wheel notches, is also appended to `<trace stem>.motion.csv`. Header:

```text
user_id;when_ns;saved;duration_ms;angle;distance;velocity;path_px;jitter_px;reversals;samples;accel;drag;button;approach_px_s;slowdown;click_gap_ms;wheel_notches
```

| Column | Meaning |
|---|---|
| `saved` | `1` if the row was also written to the dataset CSV |
| `path_px` | path length inside the slice |
| `jitter_px` | path length minus the straight-line distance, floored at 0 |
| `reversals` | consecutive samples whose directions point against each other |
| `samples` | number of pointer samples in the slice |
| `accel` | mean absolute change in speed, pixels per second squared |
| `drag` | `1` if a button was held when the slice closed |
| `button` | empty, `left`, `right`, `middle`, or `wheel` |
| `approach_px_s` | path speed over the 200 ms before the button went down |
| `slowdown` | early-half speed minus late-half speed over that 200 ms; positive means the pointer slowed |
| `click_gap_ms` | time since the previous down of the same button |
| `wheel_notches` | signed wheel steps |

`when_ns` is the collector clock in nanoseconds. An older 11-column sidecar cannot be mixed with this header. Fine-tuning then ignores the sidecar and fits the personal template on the dataset CSV alone.

The live aim profile does not score jitter, acceleration, drag, approach, slowdown, click gap, or wheel. Those columns are recorded so a later template can use them. The current template uses flick speed, flick distance, and click length.

## Two models

Fine-tuning writes both into one joblib file. Once that file contains a personal template, the live window uses the template and ignores the population score.

### Population model

`models/pointerprint.joblib` is a `StandardScaler` followed by `GaussianNB` (`var_smoothing=1e-6`). Class priors are uniform after fitting. The score of a window is a capped likelihood ratio against a standard normal in scaled space. Each feature's penalty is capped at 4.5, which is `0.5 × z²` at three standard deviations. A higher score is a closer match to that class. This is not a 1,697-way softmax.

Training, from `models/metrics.json`:

| | |
|---|---|
| Rows | 31,990,875 |
| Users in the file | 1,755 |
| Users trained | 1,697 |
| Users skipped | 58 (fewer than 3 training windows) |
| Train / test cut | earliest 40% of each user after sorting by time; later 60% held out |
| Window | 50 actions, non-overlapping |
| Active window | at least 10 strokes; idle windows are dropped |
| Cap per user | 80 train windows and 20 test windows, seed 0 |
| Train windows | 92,978 |
| Eval windows | 32,443 genuine and 32,443 impostor |
| Genuine accept rate | 0.710 |
| Impostor accept rate | 0.311 |
| Genuine median score | 6.446 |
| Impostor median score | −2.630 |
| Reported threshold | 2.573, the median of the per-user thresholds |

Each user's own threshold is the 5th percentile of that user's held-out enrollment scores. The latest 20% of their training windows are held out of the Gaussian fit when there are at least 3 fit windows and 2 calibration windows. Otherwise the same windows are used for both. Velocity is clipped to 0–5,000 px/s and duration to 0–10,000 ms before the logs.

On this dataset, `duration` is pause time. The 19 population features are stroke count, pause count, pause/stroke ratio, angle mean and standard deviation, mean absolute angle, four angle bins, median and standard deviation of `log(1 + distance)`, median and standard deviation of `log(1 + velocity)`, median and standard deviation of `log(1 + duration)`, fraction of pauses under 200 ms, fraction of pauses over 2,000 ms, and median gap in seconds.

The population file is the base for fine-tuning. Its scaler is not refit. A new user id is appended as its own class. An id that already exists is blended toward the new windows, and no feature mean is allowed to move more than 2 standardized units from that class's original mean. User id `1` is already one of the 1,697 classes, so fine-tuning id `1` only nudges a stranger's Gaussian. Enroll a desktop user as a new id, such as `100000` or `100001`.

The Gaussian path also has a timing rule used when a loaded profile has no personal template: idle windows do not add or clear mismatch time, 8 seconds of low-score movement is labeled "checking", and 25 seconds is labeled "mismatch held". The saved aim profile has a personal template, so the live window does not use this rule.

### Personal template

This is the score on screen for `models/aim-master.joblib` and for any profile written by `python -m mousewatch.finetune`. It was tuned on repeated aim-test rounds, where two people can flick at a similar speed and still differ in how long they hold the click. It is a check for that kind of drill. It is not a general decision for every desktop task.

Each score uses 40 actions and four numbers:

| Feature | Definition |
|---|---|
| `log_velocity_median` | median of `log(1 + velocity)`, velocity clipped to 0–5,000 px/s |
| `log_distance_median` | median of `log(1 + distance)` |
| `fast_stroke_frac` | fraction of strokes faster than 2,000 px/s |
| `log_duration_median` | median of `log(1 + click duration ms)`, duration clipped to 0–10,000; 0 if the window has no click |

A window with fewer than 8 strokes is skipped. The center of each feature is the median of the enrollment windows. The scale is `1.4826 × MAD`, floored at 0.01. Missing values are filled with the center. Each feature adds `min(0.5 × z², 12)`, and the four penalties are summed. The accept limit is the 90th percentile of enrollment penalties.

Match percent stays high around the enrolled user's usual distance and falls steeply past it. With no stored typical penalty, the curve is `100 / (1 + penalty / limit)`, which is 100 at distance 0 and 50 at the limit. Profiles written by the current fitter store the median enrollment penalty and use a steeper curve around that point.

The template needs at least 3 windows. About five minutes of the same aim drill is enough. After a change to these features or to the window length, rebuild the profile. An older file is scored with the feature names stored inside it, but the window length and the match curve come from the code that is running.

## Saved files

| File | What it is |
|---|---|
| `models/pointerprint.joblib` | Population model. Fine-tuning reads it and does not rewrite it. |
| `models/metrics.json` | Held-out metrics for that training run. |
| `models/aim-master.joblib` | Personal profile for user `100000`, built from the five-minute aim recording. |
| `desktop/old/traces.csv` | That recording, 23:52–23:57 on 2026-10-17, user `100000`. |
| `desktop/old/traces.motion.csv` | Motion sidecar for the same recording. |
| `data/6681.csv` | Browsing trace for user `6681`, 02:28–02:52 on 2026-10-18. Same seven columns as the dataset CSV. It is a recording, not a fine-tuned profile. |

A fine-tune with `--output master.joblib` writes `desktop/master.joblib` on this machine. That file is local. The copy stored in the repo is `models/aim-master.joblib`.

Other CSVs and joblibs under `desktop/` (`traces.csv`, `test.csv`, `user-100001.csv`, `master-100001.joblib`, and their sidecars) are local session logs. They are not required to run the saved aim profile.

## Setup

Python 3.11 or newer. From the repository root:

```powershell
python -m pip install -r requirements.txt
```

That installs numpy, scikit-learn, joblib, and pytest. Capture uses the standard library and Tk for the window. Loading a profile and fine-tuning need the scientific packages.

Run collector commands from the `desktop` directory so Python can import `mousewatch`. Fine-tuning adds the repository `src` directory itself. Training the population model from the repository root needs `PYTHONPATH=src`, or an editable install of the `pointerprint` package.

## Collect

```powershell
cd desktop
python -m mousewatch --user-id 100000 --no-tray --output traces.csv --profile missing.joblib
```

`--profile missing.joblib` points at a file that is not there, so this recording is not scored against a profile already on disk. The window says "No master profile". Stop with Ctrl+C. The command writes `traces.csv` and `traces.motion.csv` in `desktop`.

If you omit `--profile`, the app loads `master.joblib` beside the output file. Recording to `desktop\user-100001.csv` without `--profile` would compare against `desktop\master.joblib` when that file exists. It would not overwrite it. Fine-tuning is what writes a profile, and the default output of fine-tune is `master.joblib` beside the trace.

`MouseWatch.bat` starts `pythonw -m mousewatch` with no arguments. That writes `%LOCALAPPDATA%\MouseWatch\traces.csv` as user `1` and looks for `master.joblib` beside that file, not `desktop\master.joblib`. With the tray icon enabled, stop from the tray menu. `--no-tray` leaves stop to Ctrl+C.

## Build a personal profile

From `desktop`, after the trace and its sidecar exist:

```powershell
python -m mousewatch.finetune --user-id 100000 --trace old\traces.csv --base-model ..\models\pointerprint.joblib --output master.joblib
```

Pass `--output` whenever a `master.joblib` already sits next to the trace. The default writes `master.joblib` beside the trace and replaces that file. The population file is only read.

The command keeps the shared scaler fixed, updates or adds the Gaussian class for that user id, sets `owner_id`, and fits the personal template. If the sidecar has at least 3 rows for that user and the current 18-column header, the template is fit on the sidecar. Otherwise it is fit on the dataset rows. The Gaussian update itself always uses the dataset rows, in population windows of 50 actions, and it needs at least 3 of those windows.

Rebuild the saved aim profile from the recording in the repo with the command above. Watch it with either `master.joblib` or `..\models\aim-master.joblib`.

## Watch a live session

```powershell
python -m mousewatch --no-tray --output test.csv --profile master.joblib
```

Use a different output file from the enrollment trace. To watch the profile stored in the repo, from `desktop`:

```powershell
python -m mousewatch --no-tray --output test.csv --profile ..\models\aim-master.joblib
```

A new Python process picks up source changes. There is nothing to reinstall. Close the previous collector first.

## A second person

Use a new id, a new trace, and a new profile path. These commands leave `desktop\master.joblib` and `models\aim-master.joblib` untouched. The base model is the population file, not the aim profile.

```powershell
cd desktop
python -m mousewatch --user-id 100001 --no-tray --output user-100001.csv --profile missing.joblib
python -m mousewatch.finetune --user-id 100001 --trace user-100001.csv --base-model ..\models\pointerprint.joblib --output master-100001.joblib
python -m mousewatch --no-tray --output user-100001-test.csv --profile master-100001.joblib
```

Aim for about five minutes of the same drill before fine-tuning. Return to user `100000` with `--profile master.joblib` or `--profile ..\models\aim-master.joblib`.

## Retrain the population model

From the repository root, with the Mousetrack file available:

```powershell
$env:PYTHONPATH = "src"
python -m pointerprint.train --dataset "C:\Users\Noah\Downloads\dataset.csv" --model-out models/pointerprint.joblib --metrics-out models/metrics.json
```

Optional flags are `--train-fraction` (default 0.4), `--window` (50), `--stride` (50), `--max-train-windows` (80), `--max-test-windows` (20), and `--seed` (0). The command exits with an error if the held-out genuine median does not beat the impostor median. Retraining replaces `models/pointerprint.joblib`. It does not rebuild `models/aim-master.joblib`. Run fine-tune again after a population retrain if new personal profiles should start from that base.

## Command reference

Collector, from `desktop`:

| Flag | Default | Role |
|---|---|---|
| `--user-id` | `1` | Integer written to `idman` |
| `--output` | `%LOCALAPPDATA%\MouseWatch\traces.csv` | Dataset CSV path |
| `--profile` | `master.joblib` beside `--output` | Profile to score against |
| `--pause-ms` | `100` | Stillness that ends a slice |
| `--segment-ms` | `130` | Maximum slice length |
| `--min-distance` | `4` | Strokes shorter than this stay out of the dataset CSV |
| `--no-tray` | tray on | Stop with Ctrl+C instead of the tray menu |

Fine-tune, from `desktop`:

| Flag | Default | Role |
|---|---|---|
| `--user-id` | required | `idman` value to enroll |
| `--trace` | `%LOCALAPPDATA%\MouseWatch\traces.csv` | Dataset CSV |
| `--base-model` | `models/pointerprint.joblib` | Population model, read only |
| `--output` | `master.joblib` beside the trace | Personal profile to write |

## Tests

From the repository root:

```powershell
python -m pytest tests desktop/tests -q
```

`tests` covers the dataset loader, population features, the Gaussian model, the personal template, and the saved `pointerprint.joblib` metrics. `desktop/tests` covers capture, fine-tune, and the live verdict. The packages under `model/` and `shared/` are not on that test path.

## Layout

| Path | Role |
|---|---|
| `src/pointerprint` | Dataset loader, population features, Gaussian model, personal template, training entry point. |
| `desktop/mousewatch` | Windows collector, motion sidecar, fine-tune command, live verdict window. |
| `desktop/MouseWatch.bat` | Starts the collector with no arguments, through `pythonw`. |
| `models` | Population model, its metrics, and the saved aim profile. |
| `desktop/old` | The five-minute user `100000` recording and its sidecar. |
| `data/6681.csv` | A browsing trace for user `6681`. |
| `model`, `shared` | Unused package stubs. The collector and the live check do not import them. |

## Limits

- The personal template separates people on a repeated aim drill. A different task, a different pointer speed, or a different display scale changes distance and velocity. Click length is the feature that stayed stable across aim rounds in the enrollment this profile was built from.
- The population accept rates are window-level rates on held-out Mousetrack actions. They are not the match percent shown in the desktop window.
- The app records and displays a match. It does not lock Windows, take a screenshot, or send traces anywhere.
- Enroll desktop users with an id outside the population set. Id `1` is already taken.
