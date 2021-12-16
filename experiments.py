<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
import os
>>>>>>> 26da787 (Reorganize repo a bit, avoid filesize limit.)
from sklearn.decomposition import PCA

from music import *


# Load and get chromas from the song data.
<<<<<<< HEAD
song_data = parse_song_data()
=======
try:
    import pickle
    song_data = []
    for fname in os.listdir('./data'):
        with open(os.path.join('./data', fname), 'rb') as pf:
            song_data.append(pickle.load(pf))
except:
    print("Did not find pickle with song data. Loading it manually from ./wavs folder.")
    song_data = parse_song_data()
>>>>>>> 26da787 (Reorganize repo a bit, avoid filesize limit.)


# Define the functions that were attempted.
def f0(sd):
    """Simply flatten the arrays, at full resolution."""
    sh = sd["S_D"].shape
    return sd["S_D"].reshape(sh[0], sh[1] * sh[2])


def f1(sd):
    """Add volume as a channel, then simply flatten the arrays, at full resolution."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Re-partition it into documents
    N = F.shape[1]
    L = 20
    LL = int(L * N / 180)
    F_D = np.stack([F[:, i : (i + LL)] for i in range(N - LL + 1)])
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1] * sh[2])


def f2(sd):
    """Downsample and then flatten the standard arrays (no volume)."""
    C_ds = sd["C"][:, ::43]
    S_D = make_shingles(C_ds, 20, 180)
    sh = S_D.shape
    return S_D.reshape(sh[0], sh[1] * sh[2])


def f3(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = F[:, ::43]

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1] * sh[2])


def f4(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = np.stack([np.median(F_bit, axis=1) for F_bit in make_shingles(F, 1, 180, 1)])

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1] * sh[2])


def f5(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = np.stack([np.mean(F_bit, axis=1) for F_bit in make_shingles(F, 1, 180, 1)])

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1] * sh[2])


def f6(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median (CORRECTED)."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = np.stack(
        [np.median(F_bit, axis=1) for F_bit in make_shingles(F, 1.5, 180, 1)]
    ).T

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)

    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1] * sh[2])


def f7(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean (CORRECTED)."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = np.stack(
        [np.mean(F_bit, axis=1) for F_bit in make_shingles(F, 1.5, 180, 1)]
    ).T

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)

    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1] * sh[2])


shingles, _, _, _, _ = make_shingle_set(song_data, f7)
pca_70pc = PCA(n_components=36).fit(shingles)


def f8(sd):
    """Apply PCA (36 components, 70% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = np.stack(
        [np.mean(F_bit, axis=1) for F_bit in make_shingles(F, 1.5, 180, 1)]
    ).T

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)

    sh = F_D.shape
    return pca_70pc.transform([a for a in F_D.reshape(sh[0], sh[1] * sh[2])])


pca_80pc = PCA(n_components=60).fit(shingles)


def f9(sd):
    """Apply PCA (60 components, 80% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = np.stack(
        [np.mean(F_bit, axis=1) for F_bit in make_shingles(F, 1.5, 180, 1)]
    ).T

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)

    sh = F_D.shape
    return pca_80pc.transform([a for a in F_D.reshape(sh[0], sh[1] * sh[2])])


pca_90pc = PCA(n_components=101).fit(shingles)


def f10(sd):
    """Apply PCA (101 components, 90% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = np.stack(
        [np.mean(F_bit, axis=1) for F_bit in make_shingles(F, 1.5, 180, 1)]
    ).T

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)

    sh = F_D.shape
    return pca_90pc.transform([a for a in F_D.reshape(sh[0], sh[1] * sh[2])])


pca_60pc = PCA(n_components=22).fit(shingles)


def f11(sd):
    """Apply PCA (22 components, 60% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean."""

    # Add the volume to the top of the chroma array.
    F = np.block([[sd["volume"]], [sd["C"]]])

    # Downsample features.
    F_ds = np.stack(
        [np.mean(F_bit, axis=1) for F_bit in make_shingles(F, 1.5, 180, 1)]
    ).T

    # Re-partition it into documents
    F_D = make_shingles(F_ds, 20, 180)

    sh = F_D.shape
    return pca_60pc.transform([a for a in F_D.reshape(sh[0], sh[1] * sh[2])])


# Run the experiments.
funcs = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]
sample_sizes = [10, 10] + [500] * (len(funcs) - 2)
for f, sample_size in zip(funcs, sample_sizes):
    print(f.__doc__)
    run_experiment(sample_size, song_data, f)

# Print the summary report
print_report()
