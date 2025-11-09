import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from .forecast import linear_prediction

def plot_word_timeseries(
    word: str,
    occurence_freq: pd.DataFrame,
    occurrences_abs: pd.DataFrame | None = None,
    train_years=(2020, 2022),
    target_year=2024,
    year_range: tuple[int, int] | None = None,
    show=True,
    dpi=300,
    xtick_step: int = 2,        
    percent_y: bool = True,     
):
    if word not in occurence_freq.columns:
        raise ValueError(f"word '{word}' not in vocabulary.")

    series = occurence_freq[word].dropna()
    years_all = series.index.values.astype(int)

    y0, y1 = train_years
    train = series.loc[(series.index >= y0) & (series.index <= y1)]
    q = linear_prediction(train, target_year=target_year)

    x_train = train.index.values.astype(float)
    m, b = np.polyfit(x_train, train.values, 1)
    years_trend = np.arange(y0, target_year + 1)
    trend_vals = m * years_trend + b

    if year_range is None:
        year_range = (int(years_all.min()), int(years_all.max()))

    n_years = year_range[1] - year_range[0] + 1
    width = max(6, n_years * 0.25)

    fig, (ax, ax_text) = plt.subplots(
        1, 2, figsize=(width, 3.8), dpi=dpi, gridspec_kw={'width_ratios': [2.3, 1]}
    )

    ax.plot(years_all, series.values, marker='o', label="observed")
    ax.plot(years_trend, trend_vals, linestyle='--', color='orange', label="trend")
    ax.axvline(2022 + 11/12, color="red", linestyle="--", linewidth=1.2, label="ChatGPT release")

    ax.set_xlim(year_range)
    tick_years = list(range(year_range[0], year_range[1] + 1, max(1, xtick_step)))
    ax.set_xticks(tick_years)
    ax.tick_params(axis="x", rotation=90)

    p = series.get(target_year, np.nan)
    y_min = min(series.min(), np.nanmin(trend_vals))
    y_max = max(series.max(), np.nanmax(trend_vals))
    y_margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)

    if percent_y:
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))

    ax.set_title(word)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Frequency ({year_range[0]}–{year_range[1]})")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),   
        ncol=3,
        fontsize=8,
        frameon=False
    )

    ax_text.axis("off")
    text_lines = []
    if occurrences_abs is not None and word in occurrences_abs.columns:
        abs_counts = occurrences_abs[word].dropna()
        freq_series = occurence_freq[word].dropna()
        years = abs_counts.index.intersection(freq_series.index).astype(int)
        y0r, y1r = year_range
        years = np.sort(years[(years >= y0r) & (years <= y1r)])

        text_lines.append("Occurrences per Year:\n")
        total_occ = 0
        for year in years:
            abs_value = int(abs_counts.loc[year])
            freq_value = float(freq_series.loc[year])
            total_occ += abs_value
            text_lines.append(f"{year}: {abs_value:>6}   {freq_value:>8.3%}")
        text_lines.append("")
        text_lines.append(f"Total: {int(total_occ):>6}")
        text_lines.append("")

    ax_text.text(0, 1, "\n".join(text_lines), va="top", ha="left", fontsize=9, family="monospace")

    if show:
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.show()

    return {
        "p": float(p) if np.isfinite(p) else None,
        "q": float(q),
        "delta": float((p - q)) if np.isfinite(p) else None,
    }



def plot_word_scatter(
    df,
    metric: str = "delta",
    llm_words: set[str] | None = None,
    min_p=1e-5,
    min_q=1e-5,
    annotate_top_n=5,
    figsize=(7, 4.5),
):

    def annotate_top_words(ax, df, word_set, metric, n=15):
        top = df[df.index.isin(word_set)].sort_values(metric, ascending=False).head(n)
        for word, row in top.iterrows():
            ax.text(row["p"], row[metric], " " + word, fontsize=6, va="center", ha="left")


    df = df.copy()
    df = df[(df["p"] > min_p) & (df["q"] > min_q)].dropna(subset=[metric, "p"])

    colors = np.full(len(df), "tab:blue", dtype=object)
    if llm_words:
        colors[df.index.isin(llm_words)] = "tab:red"

    sizes = 12 + 200 * np.clip(df["p"].values,0.0,0.02)  

    _, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.scatter(df["p"], df[metric], s=sizes, c=colors, linewidths=0)

    ax.set_xscale("log")
    ax.set_ylim(0)
    ax.axhline(0, color="gray", lw=0.8, ls="--")

    # mean line
    mean_val = df[metric].mean()
    ax.axhline(mean_val, color="black", lw=1.0, ls=":", label=f"Mean {metric} = {mean_val:.2e}")


    ax.set_xlabel("Frequency p (log scale)")
    match metric:
        case "delta":
            ax.set_ylabel("Frequency gap between 2024 and 2022")
        case "growth":
            ax.set_ylabel("Frequency growth between 2024 and 2022")
        case "ratio":
            ax.set_ylabel("Frequency ratio between 2024 and 2022")
        case _:
            ax.set_ylabel(metric)

    # top_n of llm_words
    annotate_top_words(ax, df, llm_words, metric, n=annotate_top_n)
    # all words except top_n of llm_words
    annotate_top_words(ax, df, set(df.index) - llm_words, metric, n=annotate_top_n)

    handles, labels = [], []
    if llm_words:
        handles.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:red', markersize=6))
        labels.append("LLM lexicon")
    if handles:
        ax.legend(handles, labels, frameon=False, loc="upper left")

    plt.tight_layout()
    plt.show()
