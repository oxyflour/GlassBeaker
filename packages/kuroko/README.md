# MI Gradient Viewer

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

## What it does

- Upload one or more CST `.ffs` files (one per RX port)
- Upload a channel JSON
- Visualize each port as a 3D far-field pattern surface
- Compute MI distribution over channel realizations
- Compute the gradient of mean MI with respect to the local angular gain of the selected port
- Sum MI over a full horizontal rotation sweep of the selected imported pattern before computing gradients
- Recolor the pattern by gradient magnitude

## Supported channel JSON formats

1. Simple snapshot/path format:

```json
{
  "snapshots": [
    {
      "paths": [
        {
          "aoa_theta_deg": 70,
          "aoa_phi_deg": 30,
          "gain": [0.8, 0.1],
          "pol": [[1, 0], [0, 0]]
        }
      ]
    }
  ]
}
```

2. Sionna CDL JSON (`CDL-D.json`-style)

Approximate cluster-level evaluation using `aoa`, `zoa`, `powers`, and `xpr`.

## Gradient meaning

For the selected RX port and each angular sample `k`, the app computes

- `d(sum_yaw MI_yaw)/ds_k`, where `s_k` is a local multiplicative scaling of the complex field sample,
- then converts it to `dMI/dgain_k` by dividing by the local linear gain.

The horizontal sweep uses the selected pattern's imported `phi` grid as yaw angles and samples paths at
`local_phi = channel_phi - yaw`.

The default gradient surface coloring is `|dMI/dgain|`.

For nulls or very small gains, `|dMI/dgain|` can become visually sharp. The UI also offers `|dMI/dlog(gain)|`, which is often more stable for interpretation.
