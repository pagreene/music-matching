# Results

## Summary

Summarize the results for each method of representing the shingles. `P_f` is the fraction of trials in which the top result found a match (the closer to 1 the better), `[n]` is the fraction of possible matches at the top of the list (the larger the better), and last but not least, `[[d]]` is an average over the average distances of matching results from the top of the list, and `[t]` is the average time taken to calculate the score ranking.

| Method Description                                                                                                        |   Sample Size |   `P_f` |    `[n]` |   `[[d]]` |       `[t]` |
|---------------------------------------------------------------------------------------------------------------------------|---------------|---------|----------|-----------|-------------|
| Downsample and then flatten the standard arrays (no volume).                                                              |           500 |   0.668 | 0.359    |   5.13633 | 0.0017565   |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz.                                            |           500 |   0.636 | 0.344333 |   5.45567 | 0.00199245  |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median.                                  |           500 |   0.214 | 0.180667 |   6.78967 | 0.00058974  |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean.                                    |           500 |   0.228 | 0.188    |   6.815   | 0.000677604 |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median (CORRECTED).                      |           500 |   0.774 | 0.456    |   4.878   | 0.00407387  |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean (CORRECTED).                        |           500 |   0.768 | 0.457667 |   4.79167 | 0.00304092  |
| Apply PCA (36 components, 70% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean.  |           500 |   0.84  | 0.505667 |   4.657   | 0.00318982  |
| Apply PCA (60 components, 80% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean.  |           500 |   0.836 | 0.508333 |   4.74733 | 0.00139243  |
| Apply PCA (101 components, 90% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean. |           500 |   0.774 | 0.462    |   5.027   | 0.00165924  |
| Apply PCA (22 components, 60% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean.  |           500 |   0.768 | 0.451667 |   5.04667 | 0.00098491  |

## Method Details

### Downsample and then flatten the standard arrays (no volume).

```python
def f2(sd):
    """Downsample and then flatten the standard arrays (no volume)."""
    C_ds = sd["C"][:, ::43]
    S_D = make_shingles(C_ds, 20, 180)
    sh = S_D.shape
    return S_D.reshape(sh[0], sh[1] * sh[2])

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz.

```python
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

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median.

```python
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

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean.

```python
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

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median (CORRECTED).

```python
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

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean (CORRECTED).

```python
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

```

### Apply PCA (36 components, 70% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean.

```python
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

```

### Apply PCA (60 components, 80% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean.

```python
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

```

### Apply PCA (101 components, 90% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean.

```python
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

```

### Apply PCA (22 components, 60% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean.

```python
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

```

## Run Details

### Downsample and then flatten the standard arrays (no volume). (500)

fraction found: **0.668**	average first match: **0.359**	average average distance: **5.14**	average time: **0.00176**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | stravinsky  |      42 |
| stravinsky | spring-finale | ozawa       |      40 |
| mahler     | symphony3-3   | zander      |      20 |
| mahler     | symphony3-3   | rattle      |      17 |
| mahler     | symphony6-1   | zander      |      16 |
| beethoven  | symphony7-1   | karajan     |      15 |
| mahler     | symphony6-1   | barbirolli  |       7 |
| beethoven  | symphony7-1   | zander      |       3 |
| beethoven  | symphony7-1   | gardner     |       3 |
| poulenc    | flutesonata-1 | unkown1     |       2 |
| poulenc    | flutesonata-1 | unkown2     |       1 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz. (500)

fraction found: **0.636**	average first match: **0.344**	average average distance: **5.46**	average time: **0.00199**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| mahler     | symphony3-3   | zander      |      46 |
| stravinsky | spring-finale | stravinsky  |      36 |
| mahler     | symphony3-3   | rattle      |      33 |
| mahler     | symphony6-1   | zander      |      26 |
| stravinsky | spring-finale | ozawa       |      26 |
| beethoven  | symphony7-1   | karajan     |       9 |
| beethoven  | symphony7-1   | gardner     |       4 |
| beethoven  | symphony7-1   | zander      |       1 |
| poulenc    | flutesonata-1 | unkown2     |       1 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median. (500)

fraction found: **0.214**	average first match: **0.181**	average average distance: **6.79**	average time: **0.00059**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | stravinsky  |     206 |
| stravinsky | spring-finale | ozawa       |     156 |
| mahler     | symphony3-3   | rattle      |       9 |
| mahler     | symphony3-3   | zander      |       8 |
| mahler     | symphony6-1   | zander      |       7 |
| poulenc    | flutesonata-1 | unkown3     |       3 |
| poulenc    | flutesonata-1 | unkown2     |       2 |
| prokofiev  | flute         | adorjan     |       2 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean. (500)

fraction found: **0.228**	average first match: **0.188**	average average distance: **6.82**	average time: **0.000678**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | stravinsky  |     203 |
| stravinsky | spring-finale | ozawa       |     155 |
| mahler     | symphony3-3   | rattle      |      12 |
| mahler     | symphony6-1   | zander      |       9 |
| mahler     | symphony3-3   | zander      |       4 |
| poulenc    | flutesonata-1 | unkown3     |       2 |
| poulenc    | flutesonata-1 | unkown2     |       1 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median (CORRECTED). (500)

fraction found: **0.774**	average first match: **0.456**	average average distance: **4.88**	average time: **0.00407**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | ozawa       |      32 |
| mahler     | symphony3-3   | zander      |      32 |
| mahler     | symphony3-3   | rattle      |      17 |
| stravinsky | spring-finale | stravinsky  |      16 |
| mahler     | symphony6-1   | zander      |       9 |
| poulenc    | flutesonata-1 | unkown2     |       3 |
| beethoven  | symphony7-1   | karajan     |       2 |
| prokofiev  | flute         | adorjan     |       2 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean (CORRECTED). (500)

fraction found: **0.768**	average first match: **0.458**	average average distance: **4.79**	average time: **0.00304**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| mahler     | symphony3-3   | zander      |      42 |
| stravinsky | spring-finale | stravinsky  |      32 |
| stravinsky | spring-finale | ozawa       |      16 |
| mahler     | symphony3-3   | rattle      |      14 |
| mahler     | symphony6-1   | zander      |      10 |
| beethoven  | symphony7-1   | karajan     |       2 |

### Apply PCA (36 components, 70% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean. (500)

fraction found: **0.84**	average first match: **0.506**	average average distance: **4.66**	average time: **0.00319**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece           | performer   |   count |
|------------|-----------------|-------------|---------|
| stravinsky | spring-finale   | ozawa       |      19 |
| stravinsky | spring-finale   | stravinsky  |      16 |
| mahler     | symphony3-3     | zander      |       8 |
| mahler     | symphony6-1     | zander      |       7 |
| mahler     | symphony6-1     | barbirolli  |       6 |
| mahler     | symphony3-3     | rattle      |       6 |
| poulenc    | flutesonata-1   | unkown1     |       3 |
| beethoven  | symphony7-1     | gardner     |       3 |
| poulenc    | flutesonata-1   | unkown2     |       3 |
| beethoven  | symphony7-1     | karajan     |       2 |
| beethoven  | pianosonata32-1 | kempff      |       2 |
| poulenc    | flutesonata-1   | unkown3     |       2 |
| beethoven  | symphony7-1     | zander      |       1 |
| beethoven  | pianosonata32-2 | kempff      |       1 |
| prokofiev  | flute           | grot        |       1 |

### Apply PCA (60 components, 80% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean. (500)

fraction found: **0.836**	average first match: **0.508**	average average distance: **4.75**	average time: **0.00139**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | stravinsky  |      21 |
| mahler     | symphony3-3   | zander      |      20 |
| mahler     | symphony3-3   | rattle      |      12 |
| stravinsky | spring-finale | ozawa       |      10 |
| mahler     | symphony6-1   | zander      |       5 |
| mahler     | symphony6-1   | barbirolli  |       3 |
| poulenc    | flutesonata-1 | unkown3     |       3 |
| poulenc    | flutesonata-1 | unkown1     |       3 |
| poulenc    | flutesonata-1 | unkown2     |       2 |
| beethoven  | symphony7-1   | karajan     |       1 |
| prokofiev  | flute         | adorjan     |       1 |
| prokofiev  | flute         | grot        |       1 |

### Apply PCA (101 components, 90% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean. (500)

fraction found: **0.774**	average first match: **0.462**	average average distance: **5.03**	average time: **0.00166**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| mahler     | symphony3-3   | zander      |      37 |
| stravinsky | spring-finale | stravinsky  |      22 |
| stravinsky | spring-finale | ozawa       |      20 |
| mahler     | symphony3-3   | rattle      |      12 |
| mahler     | symphony6-1   | zander      |       9 |
| poulenc    | flutesonata-1 | unkown2     |       4 |
| prokofiev  | flute         | adorjan     |       3 |
| poulenc    | flutesonata-1 | unkown1     |       3 |
| poulenc    | flutesonata-1 | unkown3     |       2 |
| beethoven  | symphony7-1   | karajan     |       1 |

### Apply PCA (22 components, 60% variance explained) transform to the volume-agumented chroma downsampled to ~1 Hz by mean. (500)

fraction found: **0.768**	average first match: **0.452**	average average distance: **5.05**	average time: **0.000985**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece           | performer   |   count |
|------------|-----------------|-------------|---------|
| stravinsky | spring-finale   | stravinsky  |      22 |
| stravinsky | spring-finale   | ozawa       |      16 |
| poulenc    | flutesonata-1   | unkown1     |      12 |
| mahler     | symphony3-3     | zander      |      12 |
| poulenc    | flutesonata-1   | unkown3     |      10 |
| prokofiev  | flute           | grot        |       8 |
| mahler     | symphony3-3     | rattle      |       6 |
| mahler     | symphony6-1     | barbirolli  |       5 |
| poulenc    | flutesonata-1   | unkown2     |       4 |
| beethoven  | symphony7-1     | karajan     |       4 |
| beethoven  | symphony7-1     | gardner     |       4 |
| beethoven  | pianosonata32-1 | kempff      |       3 |
| prokofiev  | flute           | adorjan     |       3 |
| beethoven  | pianosonata32-2 | kempff      |       3 |
| beethoven  | pianosonata32-1 | pollini     |       2 |
| mahler     | symphony6-1     | zander      |       2 |