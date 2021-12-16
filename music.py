import os
import json
import inspect
from datetime import datetime

import tabulate
import random
import numpy as np
import librosa
import librosa.display

from umap import UMAP
from matplotlib import pyplot as plt
from seaborn import scatterplot
from collections import Counter


def parse_song_data():
    """Calculate and plot L2 normed CENS chromagram, alongside the Volume of the signal."""
    song_files = [fname for fname in os.listdir("./wavs") if fname.endswith(".wav")]
    fig, axes = plt.subplots(
        len(song_files), 2, sharex=True, figsize=(16, len(song_files) * 1.1)
    )
    song_data = []
    L = 20  # seconds
    dur = 180  # seconds
    for i, song_file in enumerate(song_files):
        song_file_path = os.path.join("./wavs", song_file)
        print(song_file_path)
        print("=" * len(song_file_path))

        print("Loading...")
        x, sr = librosa.load(song_file_path, duration=dur)
        Y = np.abs(x) ** 2

        print("Generating chroma...")
        C = librosa.feature.chroma_cens(y=Y, sr=sr)
        ax1 = axes[i][0]
        spec_plt = librosa.display.specshow(
            C, x_axis="time", y_axis="chroma", cmap="gray_r", ax=ax1
        )
        fig.colorbar(spec_plt, ax=ax1)
        if i != len(song_files) - 1:
            ax1.set_xlabel("")
            ax1.set_xticklabels([])

        print("Calculating Volume...")
        ax2 = axes[i][1]
        rms = librosa.feature.rms(x)
        volume = np.clip(0.4 * np.log10(rms[0] / 0.002), 0, 1)
        ax2.plot(librosa.times_like(rms), volume)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel("dB")
        if i != len(song_files) - 1:
            ax2.set_xlabel("")
            ax2.set_xticklabels([])
        else:
            ax2.set_xlabel("Time")

        song_data.append(
            {
                "song_file": song_file,
                "Y": Y,
                "C": C,
                "rms": rms[0],
                "volume": volume,
            }
        )
        print()

    pad = 5
    for ax, col in zip(axes[0], ["Chromagram", "Volume"]):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    for ax, row in zip(axes[:, 0], song_files):
        ax.annotate(
            row.split(".")[0],
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )

    fig.subplots_adjust(
        left=0.2, right=0.98, wspace=0.1, hspace=0.3, bottom=0.07, top=0.93
    )

    return song_data


def parse_song_file_name(file_name):
    """Extract the metadata embedded in the file name."""
    return file_name.split(".")[0].split("_")


def make_shingles(D, L, dur, hop=None):
    """Make shingles of the array D."""
    N = D.shape[1]
    LL = int(L * N / dur)

    if hop is None:
        H = 1
    else:
        H = int(hop * N / dur)

    return np.stack(
        [D[:, (i * H) : (i * H + LL)] for i in range(0, int((N - LL) / H) + 1)]
    )


def compute_scores(test_idx, shingle_idx, data):
    """Compute the scores between a query and the available data."""
    x = data[test_idx]["D"][shingle_idx]

    scores = []
    for j, sd in enumerate(data):
        if j == test_idx:
            continue
        score_arr = np.linalg.norm(sd["D"] - x, axis=1)
        min_idx = int(np.where(score_arr == score_arr.min())[0][0])
        scores.append((score_arr[min_idx], sd["song_id"], j, min_idx))
    scores.sort()
    return scores


def apply_embedding(song_data, f):
    """Apply the embedding function to the raw data."""
    data = []
    for sd in song_data:
        new_D = f(sd)
        data.append({"song_id": parse_song_file_name(sd["song_file"]), "D": new_D})
    return data


def run_experiment(n_samples, song_data, f, quiet=True):
    """Run an experiment with an encoding function `f`.

    Parameters
    ----------
    n_samples : int
        The number of trials to run in the experiment.
    song_data : list
        The list of data dictionaries for each piece of music.
    f : callable
        A function or callable that takes in a song_data dictionary, and outputs a 1-D array.
    quiet : bool
        (Default True) Indicate how much detail to give in the printouts.
    """

    # Define a shortcut function for checking if two pieces are the same.
    def match(s_id_1, s_id_2):
        return s_id_1[:1] == s_id_2[:1]

    print("Plotting umap....")
    umap, fig_name = plot_umap(song_data, f)

    # Compute the encodings for each shingle
    print("Applying embedding...")
    data = apply_embedding(song_data, f)
    N_s = len(data)

    # Set up the lists of values to track
    top_found = []
    num_in_top = []
    ave_dist = []
    score_orders = []
    queries = []
    times = []

    # Run the experiment n_samples times.
    print("Running experiments...")
    for i in range(n_samples):

        # Choose a random song
        test_idx = random.randrange(N_s)
        search = data[test_idx]

        # Choose a random shingle.
        shingle_idx = random.randrange(len(search["D"]))
        song_id = data[test_idx]["song_id"]
        queries.append([song_id, test_idx, shingle_idx])

        # Calculate the score for each other song.
        print(f"\n{i+1}/{n_samples}", song_id, shingle_idx)
        start = datetime.now()
        scores = compute_scores(test_idx, shingle_idx, data)
        time = (datetime.now() - start).total_seconds()
        times.append(time)
        score_orders.append(scores)

        # Optionally print out a table of the scores.
        if not quiet:
            print()
            print(
                tabulate.tabulate(
                    [
                        (score,) + song_id + (shingle_idx,)
                        for score, song_id, _, shingle_idx in scores
                    ],
                    headers=["score", "composer", "piece", "performer", "shingle idx"],
                )
            )

        # Check if the top is a match
        all_matches = [match(other_id, song_id) for _, other_id, _, _ in scores]
        tf = all_matches[0]
        top_found.append(tf)

        # Check what fraction of the matches are in the top.
        num_matches = sum(all_matches)
        num_top_matches = all_matches.index(False)
        nit = num_top_matches / num_matches
        num_in_top.append(nit)

        # Calculate the average distance of matches from the top.
        ave = sum(i for i, m in enumerate(all_matches) if m) / num_matches
        ave_dist.append(ave)

        print(tf, nit, ave)

    # Print the results from this experiment.
    results = [
        np.mean(top_found),
        np.mean(num_in_top),
        np.mean(ave_dist),
        np.mean(times),
    ]
    print()
    print(tabulate.tabulate([results], headers=["P_f", "<n>", "<<d>>", "<t>"]))

    # Save the results.
    with open("RESULTS.json", "r") as res_fh:
        res_j = json.load(res_fh)
    res_j.append(
        {
            "method": f.__doc__,
            "method_func": inspect.getsource(f),
            "sample_size": n_samples,
            "fig_name": fig_name,
            "results": [
                {
                    "query": q,
                    "scores": scr,
                    "top_found": tf,
                    "fraction_in_top": fit,
                    "ave_dist": ad,
                    "time": t,
                }
                for q, scr, tf, fit, ad, t in zip(
                    queries, score_orders, top_found, num_in_top, ave_dist, times
                )
            ],
            "summary": dict(
                zip(
                    [
                        "fraction_found",
                        "average_first_match",
                        "average_average_distance",
                        "average_time"
                    ],
                    results,
                )
            ),
        }
    )
    with open("RESULTS.json", "w") as res_fh:
        json.dump(res_j, res_fh, indent=2)

    return {
        "top_found": top_found,
        "num_in_top": num_in_top,
        "ave_dist": ave_dist,
        "score_orders": score_orders,
        "queries": queries,
    }


def print_report():
    """Print a readable MarkDown report from the RESULTS.json file."""
    doc_lines = []
    with open("RESULTS.json", "r") as f:
        results = json.load(f)

    doc_lines.append("# Results")

    doc_lines.append("## Summary")
    doc_lines.append(
        "Summarize the results for each method of representing the "
        "shingles. `P_f` is the fraction of trials in which the top "
        "result found a match (the closer to 1 the better), `[n]` is "
        "the fraction of possible matches at the top of the list "
        "(the larger the better), and last but not least, `[[d]]` is an "
        "average over the average distances of matching results from "
        "the top of the list, and `[t]` is the average time taken to "
        "calculate the score ranking."
    )
    doc_lines.append(
        tabulate.tabulate(
            [
                (r["method"], r["sample_size"]) + tuple(r["summary"].values())
                for r in results
            ],
            headers=[
                "Method Description",
                "Sample Size",
                "`P_f`",
                "`[n]`",
                "`[[d]]`",
                "`[t]`",
            ],
            tablefmt="github",
        )
    )

    doc_lines.append("## Method Details")

    methods_done = set()
    for run_info in results:
        if run_info["method"] in methods_done:
            continue
        doc_lines.append(f"### {run_info['method']}")
        doc_lines.append(f"```python\n{run_info['method_func']}\n```")
        methods_done.add(run_info["method"])

    doc_lines.append("## Run Details")

    for run_info in results:
        doc_lines.append(f"### {run_info['method']} ({run_info['sample_size']})")
        doc_lines.append(
            "\t".join(
                f"{key.replace('_', ' ')}: **{value:0.3}**"
                for key, value in run_info["summary"].items()
            )
        )
        doc_lines.append(
            "The number of misses caused by each entry. In other words, "
            "how often did each entry score best but was an incorrect match."
        )

        # Find the pieces of music that are most often the targets of confusion.
        gangers = [
            tuple(res["scores"][0][1]) for res in run_info["results"] if not res["top_found"]
        ]
        c = Counter(gangers)

        doc_lines.append(
            tabulate.tabulate(
                [k + (v,) for k, v in c.most_common()],
                ["composer", "piece", "performer", "count"],
                tablefmt="github",
            )
        )

    with open("RESULTS.md", "w") as f:
        f.write("\n\n".join(doc_lines))


def explore_umap_params(shingles, labels, param_list, n_trials):
    """Test each of the n_neighbor params in param_list n_trials times."""
    fig, axes = plt.subplots(
        len(param_list), n_trials, figsize=(6 * n_trials, 4 * len(param_list))
    )
    umaps = {}
    for i, n in enumerate(param_list):
        print(f"Running n_neighbors={n}...")
        for j in range(n_trials):
            umaps[(n, j)] = UMAP(n_neighbors=n).fit(shingles)
            x, y = umaps[(n, j)].transform(shingles).T
            scatterplot(x=x, y=y, hue=labels, ax=axes[i][j], alpha=0.6)
            axes[i][j].set_title(f"n_neighbors={n}, trial {j}")
    return umaps


def make_shingle_set(song_data, f):
    """Calculate all the shingles, corresponding with relevant metadata."""
    shingles = []
    composers = []
    pieces = []
    performers = []
    indexes = []
    for sd in song_data:
        new_data = [a for a in f(sd)]
        shingles.extend(new_data)
        indexes.extend(range(len(new_data)))

        song_id = parse_song_file_name(sd["song_file"])
        composers.extend([song_id[0]] * len(new_data))
        pieces.extend([song_id[1]] * len(new_data))
        performers.extend([song_id[2]] * len(new_data))
    return shingles, composers, pieces, performers, indexes


def plot_umap(song_data, f):
    """Plot the umap projection for the given embedding function `f`."""
    shingles, composers, pieces, performers, indexes = make_shingle_set(song_data, f)
    umap = UMAP(n_neighbors=10).fit(shingles)

    # Plot the global UMAP.
    fig, axes = plt.subplots(len(set(composers)) + 1, 1)
    x, y = umap.transform(shingles).T
    scatterplot(x=x, y=y, hue=composers, ax=axes[0])
    axes[0].set_title("Global UMAP")

    # Print a separate map for each of the composers.
    for i, composer in enumerate(set(composers)):
        some_shingles = [s for s, c in zip(shingles, composers) if c == composer]
        some_labels = [
            (pc, pr)
            for c, pc, pr in zip(composers, pieces, performers)
            if c == composer
        ]
        x, y = umap.transform(some_shingles).T
        scatterplot(x=x, y=y, hue=some_labels, ax=axes[i + 1])
        axes[i + 1].set_xlim(axes[0].get_xlim())
        axes[i + 1].set_ylim(axes[0].get_ylim())
        axes[i + 1].set_title(f"UMAP for {composer}")

    fig.suptitle(f"UMAPs for {f.__name__}")
    fig_name = f"{f.__name__}_umaps.jpg"
    fig.savefig(fig_name)
    return umap, fig_name
