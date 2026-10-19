# Resume procedure — StableLM Gate E utility run

**STATUS: COMPLETE (2026-10-19). Gate E PASS, 6/6 conditions within ±0.5%.**
Verdict: `utility_v2/gate_e_reanalysis.json` (`all_within_margin: true`).
No further re-arm is needed. The notes below are retained for reference /
in case a re-run is ever required.

The Gate E utility run is **resumable and supervised**. If everything was killed
(full runtime restart), re-arm it with one command:

```bash
cd /workspace/nes_run_20261016_anchor_ablation/stablelm_qualification
setsid nohup ./run_utility_supervisor.sh >/dev/null 2>&1 < /dev/null &
```

The supervisor:

- relaunches `exp_stablelm_utility.py` if it dies (checks every 60 s);
- is single-instance (flock on `utility_supervisor.lock`);
- exits once `utility_v2/comparisons.json` exists (all conditions done).

Resume semantics: completed conditions are in `utility_v2/nll/*.json` and are
skipped; in-flight conditions resume from `utility_v2/nll/*.partial.json`
(per-batch checkpoint, rework ≤ ~7 min per condition).

Check progress:

```bash
/workspace/nes_env/bin/python - <<'PY'
import json,glob
for f in sorted(glob.glob('utility_v2/nll/*.json')):
    d=json.load(open(f))
    print(f.split('/')[-1], (str(d['n_done'])+'/580' if 'n_done' in d else 'COMPLETE ppl '+str(round(d['ppl'],4))))
PY
```

After all 7 conditions complete, run the independent re-analysis:

```bash
/workspace/nes_env/bin/python reanalyze_utility.py \
  --utility-dir utility_v2 --output utility_v2/gate_e_reanalysis.json
```

Health signals: `utility_supervisor.heartbeat` (updated every 60 s),
`utility_supervisor.log` (start/relaunch events), `utility_v2.log` (runner stdout).

Disk note: each embedded condition stages one ~1.5 GB checkpoint under
`utility_v2/staging/`, removed after that condition is evaluated.
