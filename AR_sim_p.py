import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate(n_total, coeffs, change_points, p):
    """
    Generate piecewise AR(p) time series with segment-specific noise.

    Parameters
    ----------
    n_total : int
        Total number of observations to generate.
    coeffs : list
        Per-segment parameters. Each element can be:
          - list/tuple: [phi1, ..., phip, mu, sigma]
          - dict: {"phi": array_like length p, "mu": float, "sigma": float}
    change_points : list
        Segment boundaries. If there are S segments, provide either:
          - length S-1 list of interior boundaries (exclusive end indices), e.g., [cp1, cp2, ...]
          - or full boundaries [0, cp1, cp2, ..., n_total]; both forms are accepted.
    p : int
        AR order.

    Returns
    -------
    pandas.DataFrame
        Columns: Date (0..n_total-1), N (series), epsilon (innovations), segment (0-based)
    """
    if n_total <= 0:
        raise ValueError("n_total must be positive.")
    if p <= 0:
        raise ValueError("p must be positive.")
    if len(coeffs) < 1:
        raise ValueError("Provide at least one segment in coeffs.")

    # Normalize coeffs into (phi, mu, sigma)
    norm_coeffs = []
    for seg, c in enumerate(coeffs):
        if isinstance(c, dict):
            phi = np.asarray(c["phi"], dtype=float)
            mu = float(c.get("mu", 0.0))
            sigma = float(c.get("sigma", 1.0))
        else:
            c = list(c)
            if len(c) < p + 2:
                raise ValueError(f"Segment {seg}: expected at least p+2 values [phi1..phip, mu, sigma].")
            phi = np.asarray(c[:p], dtype=float)
            mu = float(c[p])
            sigma = float(c[p+1])
        if phi.shape[0] != p:
            raise ValueError(f"Segment {seg}: phi length {phi.shape[0]} != p={p}.")
        if sigma < 0:
            raise ValueError(f"Segment {seg}: sigma must be non-negative.")
        norm_coeffs.append((phi, mu, sigma))

    # Normalize change points into full closed-open segment boundaries [b0, b1, ..., bS]
    # We want boundaries like [0, ..., n_total] with S = len(coeffs)
    cp = list(change_points)
    # If user passed [0, ..., n_total], strip edges
    if len(cp) >= 2 and cp[0] == 0 and cp[-1] == n_total:
        cp = cp[1:-1]
    # Validate and build full boundaries
    if len(cp) != len(norm_coeffs) - 1:
        raise ValueError(
            f"change_points implies {len(cp)+1} segments but coeffs has {len(norm_coeffs)} segments."
        )
    if any((c <= 0 or c >= n_total) for c in cp):
        raise ValueError("All change points must be in (0, n_total).")
    if any(cp[i] <= cp[i-1] for i in range(1, len(cp))):
        raise ValueError("change_points must be strictly increasing.")
    boundaries = [0] + cp + [n_total]  # closed-open [bk, b{k+1})

    # Pre-allocate
    y = np.zeros(n_total, dtype=float)
    eps = np.zeros(n_total, dtype=float)
    seg_idx = np.zeros(n_total, dtype=int)

    # Pre-generate innovations per segment and mark segment index per t
    for k in range(len(norm_coeffs)):
        start, end = boundaries[k], boundaries[k+1]
        phi, mu, sigma = norm_coeffs[k]
        if sigma == 0:
            eps[start:end] = mu  # deterministic "noise" = constant shift
        else:
            eps[start:end] = np.random.normal(loc=mu, scale=sigma, size=end - start)
        seg_idx[start:end] = k

    # Initialize first p values (cold start). Options: zeros (default) or draw from noise.
    # Zero init is fine for most simulation demos; transient dies out.
    # If you prefer random warm-up: y[:p] = np.random.normal(0, 1, size=p)
    # (Keep zeros here to match your original behavior.)
    # Simulate
    for t in range(p, n_total):
        k = seg_idx[t]
        phi, _, _ = norm_coeffs[k]
        lags = y[t - np.arange(1, p + 1)]  # same order: [y[t-1], ..., y[t-p]]
        y[t] = np.dot(phi, lags) + eps[t]

    dates = np.arange(n_total)
    df = pd.DataFrame({
        "Date": dates,
        "N": y,
        "epsilon": eps,
        "segment": seg_idx
    })
    return df

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_time_series(
    df,
    value_col='N',
    segment_col='segment',
    time_col=None,                   # e.g., 'Date'; if None, uses df.index
    colors=None,                     # dict {seg: color}, list/tuple, single color, or mpl colormap
    title="AR Time Series",
    filename=None,
    lw=1.8,
    alpha=1.0,
    legend=True
):
    """
    Plot a time series with per-segment colors.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain value_col and segment_col (and optionally time_col).
    value_col : str
        Column name for the y-values.
    segment_col : str
        Column name with integer-like segment ids.
    time_col : str or None
        Column name for the x-axis; if None, uses DataFrame index.
    colors : dict | list/tuple | str | matplotlib Colormap | None
        - dict: {segment_id: color}
        - list/tuple: will be cycled over segments (by sorted unique ids)
        - str: single color for all segments
        - Colormap: colors sampled by normalized segment rank
        - None: default tab10 mapping
    title : str
        Plot title.
    filename : str or None
        If provided, saves the figure to this path (transparent PNG if endswith .png).
    lw : float
        Line width.
    alpha : float
        Line alpha.
    legend : bool
        Whether to show legend.
    """
    if value_col not in df.columns or segment_col not in df.columns:
        raise ValueError(f"df must have columns '{value_col}' and '{segment_col}'")

    y = df[value_col].to_numpy()
    seg = df[segment_col].to_numpy()

    if time_col is None:
        x = df.index.to_numpy()
    else:
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in df")
        x = df[time_col].to_numpy()

    # Build a color resolver
    uniq = np.unique(seg)

    def default_tab10():
        cmap = mpl.colormaps['tab10']
        return {s: cmap(i % 10) for i, s in enumerate(sorted(uniq))}

    if colors is None:
        color_map = default_tab10()
    elif isinstance(colors, dict):
        # Fill missing segments with a default palette
        color_map = {**default_tab10(), **colors}
    elif isinstance(colors, (list, tuple)):
        color_map = {s: colors[i % len(colors)] for i, s in enumerate(sorted(uniq))}
    elif isinstance(colors, mpl.colors.Colormap):
        # Normalize by rank among unique segments
        ranks = {s: i for i, s in enumerate(sorted(uniq))}
        denom = max(len(uniq)-1, 1)
        color_map = {s: colors(ranks[s] / denom) for s in uniq}
    else:
        # Single color string
        color_map = {s: colors for s in uniq}

    # Find contiguous runs where segment id stays constant
    # indices: [0, ..., n]. Breaks when seg[i] != seg[i-1]
    n = len(df)
    if n == 0:
        raise ValueError("Empty DataFrame")
    breaks = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    # breaks = [start0, end0 (=start1), end1 (=start2), ..., end_last=n]
    # runs: [breaks[i], breaks[i+1])
    plt.figure(figsize=(12, 6))
    seen = set()
    for i in range(len(breaks)-1):
        a, b = breaks[i], breaks[i+1]
        s_id = seg[a]
        lbl = None if s_id in seen else f"segment {s_id}"
        plt.plot(x[a:b], y[a:b], color=color_map[s_id], lw=lw, alpha=alpha, label=lbl)
        seen.add(s_id)

    plt.title(title)
    plt.xlabel(time_col if time_col is not None else "Time")
    plt.ylabel(value_col)
    if legend:
        plt.legend(frameon=False, ncol=min(len(uniq), 4), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()

    if filename:
        # If PNG, keep transparent background (nice for slides); else default save
        if str(filename).lower().endswith(".png"):
            plt.savefig(filename, dpi=300, transparent=True)
        else:
            plt.savefig(filename, dpi=300)
    plt.show()


# ----------------------------
# Example: repeat experiment num_runs times
# ----------------------------
if __name__ == "__main__":
    IDA_COLORS = {
        "blue": "#3B75AF",
        "red": "#EA3728",
        "green": "#398223",
        "orange": "#EF8636"
    }

    # AR(3) in seg1, AR(1) in seg2, AR(2) in seg3 — but we must choose a single p for the process.
    # Pick p=3 and pad shorter ARs with zeros at the end of phi.
    n_total = 1500
    p = 3
    coeffs = [
        # seg 0: phi1..phi3, mu, sigma
        [0.9, 0.01, 0.01, 0.0, 1.0],
        [0.01, 0.9, 0.01, 0.0, 1.0],
        [0.01, 0.01, 0.9, 0.0, 1.0],
    ]
    change_points = [500, 1000]  # segments: [0,500), [500,1000), [1000,1500)

    df = simulate(n_total, coeffs, change_points, p)
    df.to_csv(f"AR_{p}_simulated_series.csv", index=False)

    plot_time_series(df, colors={0: IDA_COLORS["blue"], 1: IDA_COLORS["red"], 2: IDA_COLORS["green"]},
                    title=f"AR ({p}) simulated data.png",
                    filename=f"AR_{p}_simulated_series.png")
