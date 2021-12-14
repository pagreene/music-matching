import os
import json
import inspect
import tabulate
import random
import numpy as np
from matplotlib import pyplot as plt
import librosa, librosa.display


def parse_song_data():
    """Calculate and plot L2 normed CENS chromagram, alongside the Volume of the signal."""
    song_files = [fname for fname in os.listdir(".") if fname.endswith('.wav')]
    fig, axes = plt.subplots(len(song_files), 2, sharex=True, figsize=(16, len(song_files)*1.1))
    song_data = []
    L = 20  # seconds
    dur = 180  # seconds
    for i, song_file in enumerate(song_files):
        print(song_file)
        print("="*len(song_file))

        print("Loading...")
        x, sr = librosa.load(song_file, duration=dur)
        Y = np.abs(x)**2

        print("Generating chroma...")
        C = librosa.feature.chroma_cens(y=Y, sr=sr)
        ax1 = axes[i][0]
        spec_plt = librosa.display.specshow(
            C,
            x_axis="time",
            y_axis="chroma",
            cmap='gray_r',
            ax=ax1
        )
        fig.colorbar(spec_plt, ax=ax1)
        if i != len(song_files) - 1:
            ax1.set_xlabel("")
            ax1.set_xticklabels([])
        
        print("Calculating Volume...")
        ax2 = axes[i][1]
        rms = librosa.feature.rms(x)
        volume = np.clip(0.4*np.log10(rms[0]/0.002), 0, 1)
        ax2.plot(librosa.times_like(rms), volume)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel("dB")
        if i != len(song_files) - 1:
            ax2.set_xlabel("")
            ax2.set_xticklabels([])
        else:
            ax2.set_xlabel("Time")
        
        print("Generating shingles...")
        S_D = make_shingles(C, L, dur)

        song_data.append({
            "song_file": song_file,
            "Y": Y,
            "C": C,
            "rms": rms[0],
            'volume': volume,
            'S_D': S_D
        })
        print()

    pad = 5
    for ax, col in zip(axes[0], ["Chromagram", "Volume"]):
        ax.annotate(
            col,
            xy=(0.5,1),
            xytext=(0, pad),
            xycoords='axes fraction',
            textcoords='offset points',
            size='large',
            ha='center',
            va='baseline'
        )

    for ax, row in zip(axes[:, 0], song_files):   
        ax.annotate(
            row.split('.')[0],
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords='offset points',
            size='large',
            ha='right',
            va='center'
        )

    fig.subplots_adjust(left=0.2, right=0.98, wspace=0.1, hspace=0.3, bottom=0.07, top=0.93)
    
    return song_data


def parse_song_file_name(file_name):
    """Extract the metadata embedded in the file name."""
    composer, piece, performer = file_name.split('.')[0].split('_')
    return {'composer': composer, 'piece': piece, 'performer': performer}


def make_shingles(D, L, dur):
    """Make shingles of the array D."""
    N = D.shape[1]
    LL = int(L*N/dur)
    return np.stack([D[:, i:(i + LL)] for i in range(N - LL + 1)])


def run_experiment(n_samples, song_data, f):
    """Test retrievability."""
    def match(s_id_1, s_id_2):
        return s_id_1['composer'] == s_id_2['composer'] and s_id_1['piece'] == s_id_2['piece']

    
    data = []
    for sd in song_data:
        new_D = f(sd)
        data.append({'song_file': sd['song_file'], 'D': new_D})
        

    top_found = []
    num_in_top = []
    ave_dist = []
    score_orders = []
    queries = []

    N_s = len(data)

    for i in range(n_samples):
        test_idx = random.randrange(N_s)
        search = data[test_idx]
        
        shingle_idx = random.randrange(len(search['D']))
        
        song_id = parse_song_file_name(search['song_file'])
        queries.append([song_id, shingle_idx])
    
        print(f"\n{i+1}/{n_samples}", song_id, shingle_idx)

        x = search['D'][shingle_idx]

        scores = []
        for j, sd in enumerate(data):
            if j == test_idx:
                continue

            score_arr = np.linalg.norm(sd['D'] - x, axis=1)
            scores.append((score_arr.min(), parse_song_file_name(sd['song_file'])))
        
        scores.sort()
        
        print()
        print(tabulate.tabulate(
            [(score, s['composer'], s['piece'], s['performer']) for score, s in scores],
            headers=['score', 'composer', 'piece', 'performer']
        ))
    
        score_orders.append(scores)
        all_matches = [match(other_id, song_id) for _, other_id in scores]
        
        # Check if the top is a match
        tf = all_matches[0]
        top_found.append(tf)

        # Check what fraction of the matches are in the top.
        num_matches = sum(all_matches)
        num_top_matches = all_matches.index(False)
        nit = num_top_matches/num_matches
        num_in_top.append(nit)

        # Caluclate the average distance of matches from the top.
        ave = sum(i for i, m in enumerate(all_matches) if m)/num_matches
        ave_dist.append(ave)

        print(tf, nit, ave)

    results = [np.mean(top_found), np.mean(num_in_top), np.mean(ave_dist)]

    print()
    print(tabulate.tabulate(
        [results],
        headers=["P_f", "<n>", "<<d>>"]
    ))

    # Save the results.
    with open("../RESULTS.json", "r") as res_fh:
        res_j = json.load(res_fh)
    res_j.append({
        "method": f.__doc__,
        "method_func": inspect.getsource(f),
        "results": [
            {'scores': scr, 'top_found': tf, 'fraction_in_top': fit, 'ave_dist': ad}
            for scr, tf, fit, ad in zip(score_orders, top_found, num_in_top, ave_dist)
        ],
        "summary": dict(zip(
            ['fraction_found', 'average_first_match', 'average_average_distance'],
            results
        ))
    })
    with open("../RESULTS.json", "w") as res_fh:
        json.dump(res_j, res_fh)
        
    return {
        'top_found': top_found,
        'num_in_top': num_in_top,
        'ave_dist': ave_dist,
        'score_orders': score_orders,
        'queries': queries
    }


def print_report():
    doc_lines = []
    with open("../RESULTS.json", "r") as f:
        results = json.load(f)
        
    doc_lines.append("# Results")
    doc_lines.append("## Summary")

    doc_lines.append(tabulate.tabulate(
        [
            (r['method'],) + tuple(r['summary'].values())
            for r in results
        ],
        headers=["Method Description", "P_f", "<n>", "<<d>>"],
        tablefmt="github"
    ))

    doc_lines.append("## Run Details")

    for run_info in results:
        doc_lines.append(f"### {run_info['method']}")
        doc_lines.append(f"```python\n{run_info['method_func']}\n```")

    with open("../RESULTS.md", "w") as f:
        f.write('\n\n'.join(doc_lines))    

