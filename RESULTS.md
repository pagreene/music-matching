# Results

## Summary

Summarize the results for each method of representing the shingles. `P_f` is the fraction of trials in which the top result found a match (the closer to 1 the better), `[n]` is the fraction of possible matches at the top of the list (the larger the better), and last but not least, `[[d]]` is an average over the average distances of matching results from the top of the list.

| Method Description                                                                                   |   Sample Size |   `P_f` |   `[n]` |   `[[d]]` |
|------------------------------------------------------------------------------------------------------|---------------|---------|---------|-----------|
| Simply flatten the arrays, at full resolution.                                                       |            10 |    0.5  |   0.5   |     4.6   |
| Add volume as a channel, then simply flatten the arrays, at full resolution.                         |            10 |    0.7  |   0.7   |     1.65  |
| Downsample and then flatten the standard arrays (no volume).                                         |            10 |    0.7  |   0.7   |     1.7   |
| Downsample and then flatten the standard arrays (no volume).                                         |           100 |    0.57 |   0.52  |     3.535 |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz.                       |           100 |    0.66 |   0.625 |     2.51  |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median.             |           100 |    0.26 |   0.255 |     4.985 |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean.               |           100 |    0.27 |   0.26  |     5.165 |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median (CORRECTED). |           100 |    0.76 |   0.725 |     1.09  |
| Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean (CORRECTED).   |           100 |    0.82 |   0.8   |     0.985 |

## Method Details

### Simply flatten the arrays, at full resolution.

```python
def f(sd):
    """Simply flatten the arrays, at full resolution."""
    sh = sd['S_D'].shape
    return sd['S_D'].reshape(sh[0], sh[1]*sh[2])

```

### Add volume as a channel, then simply flatten the arrays, at full resolution.

```python
def f(sd):
    """Add volume as a channel, then simply flatten the arrays, at full resolution."""
    
    # Add the volume to the top of the chroma array.
    F = np.block([[sd['volume']], [sd['C']]])
    
    # Re-partition it into documents
    N = F.shape[1]
    L = 20
    LL = int(L*N/180)
    F_D  = np.stack([F[:, i:(i + LL)] for i in range(N - LL + 1)])
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1]*sh[2])

```

### Downsample and then flatten the standard arrays (no volume).

```python
def f(sd):
    """Downsample and then flatten the standard arrays (no volume)."""
    C_ds = sd['C'][:, ::43]
    S_D = make_shingles(C_ds, 20, 180)
    sh = S_D.shape
    return S_D.reshape(sh[0], sh[1]*sh[2])

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz.

```python
def f(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz."""
    
    # Add the volume to the top of the chroma array.
    F = np.block([[sd['volume']], [sd['C']]])
        
    # Downsample features.
    F_ds = F[:, ::43]
    
    # Re-partition it into documents
    F_D  = make_shingles(F_ds, 20, 180)
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1]*sh[2])

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median.

```python
def f(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median."""
    
    # Add the volume to the top of the chroma array.
    F = np.block([[sd['volume']], [sd['C']]])
        
    # Downsample features.
    F_ds = np.stack([np.median(F_bit, axis=1) for F_bit in make_shingles(F, 1, 180, 1)])
    
    # Re-partition it into documents
    F_D  = make_shingles(F_ds, 20, 180)
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1]*sh[2])

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean.

```python
def f(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean."""
    
    # Add the volume to the top of the chroma array.
    F = np.block([[sd['volume']], [sd['C']]])
        
    # Downsample features.
    F_ds = np.stack([np.mean(F_bit, axis=1) for F_bit in make_shingles(F, 1, 180, 1)])
    
    # Re-partition it into documents
    F_D  = make_shingles(F_ds, 20, 180)
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1]*sh[2])

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median (CORRECTED).

```python
def f(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median (CORRECTED)."""
    
    # Add the volume to the top of the chroma array.
    F = np.block([[sd['volume']], [sd['C']]])
    
    # Downsample features.
    F_ds = np.stack([np.median(F_bit, axis=1) for F_bit in make_shingles(F, 1.5, 180, 1)]).T
    
    # Re-partition it into documents
    F_D  = make_shingles(F_ds, 20, 180)
    
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1]*sh[2])

```

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean (CORRECTED).

```python
def f(sd):
    """Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean (CORRECTED)."""
    
    # Add the volume to the top of the chroma array.
    F = np.block([[sd['volume']], [sd['C']]])
    
    # Downsample features.
    F_ds = np.stack([np.mean(F_bit, axis=1) for F_bit in make_shingles(F, 1.5, 180, 1)]).T
    
    # Re-partition it into documents
    F_D  = make_shingles(F_ds, 20, 180)
    
    sh = F_D.shape
    return F_D.reshape(sh[0], sh[1]*sh[2])

```

## Run Details

### Simply flatten the arrays, at full resolution. (10)

fraction found: **0.5**	average first match: **0.5**	average average distance: **4.6**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| mahler     | symphony6-1   | zander      |       2 |
| stravinsky | spring-finale | stravinsky  |       2 |
| beethoven  | symphony7-1   | karajan     |       1 |

### Add volume as a channel, then simply flatten the arrays, at full resolution. (10)

fraction found: **0.7**	average first match: **0.7**	average average distance: **1.65**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | ozawa       |       2 |
| mahler     | symphony3-3   | zander      |       1 |

### Downsample and then flatten the standard arrays (no volume). (10)

fraction found: **0.7**	average first match: **0.7**	average average distance: **1.7**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | ozawa       |       1 |
| mahler     | symphony6-1   | zander      |       1 |
| stravinsky | spring-finale | stravinsky  |       1 |

### Downsample and then flatten the standard arrays (no volume). (100)

fraction found: **0.57**	average first match: **0.52**	average average distance: **3.535**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | stravinsky  |      10 |
| stravinsky | spring-finale | ozawa       |       9 |
| beethoven  | symphony7-1   | karajan     |       7 |
| mahler     | symphony3-3   | zander      |       6 |
| mahler     | symphony3-3   | rattle      |       6 |
| mahler     | symphony6-1   | zander      |       2 |
| beethoven  | symphony7-1   | zander      |       2 |
| beethoven  | symphony7-1   | gardner     |       1 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz. (100)

fraction found: **0.66**	average first match: **0.625**	average average distance: **2.51**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| mahler     | symphony3-3   | zander      |      11 |
| mahler     | symphony6-1   | zander      |       7 |
| stravinsky | spring-finale | stravinsky  |       7 |
| mahler     | symphony3-3   | rattle      |       6 |
| stravinsky | spring-finale | ozawa       |       2 |
| beethoven  | symphony7-1   | karajan     |       1 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median. (100)

fraction found: **0.26**	average first match: **0.255**	average average distance: **4.985**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | ozawa       |      35 |
| stravinsky | spring-finale | stravinsky  |      34 |
| mahler     | symphony3-3   | rattle      |       2 |
| poulenc    | flutesonata-1 | unkown2     |       1 |
| prokofiev  | flute         | adorjan     |       1 |
| mahler     | symphony3-3   | zander      |       1 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean. (100)

fraction found: **0.27**	average first match: **0.26**	average average distance: **5.165**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| stravinsky | spring-finale | stravinsky  |      41 |
| stravinsky | spring-finale | ozawa       |      27 |
| mahler     | symphony3-3   | rattle      |       3 |
| mahler     | symphony6-1   | zander      |       1 |
| mahler     | symphony3-3   | zander      |       1 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by median (CORRECTED). (100)

fraction found: **0.76**	average first match: **0.725**	average average distance: **1.09**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| mahler     | symphony3-3   | zander      |      10 |
| stravinsky | spring-finale | ozawa       |       5 |
| stravinsky | spring-finale | stravinsky  |       3 |
| mahler     | symphony3-3   | rattle      |       2 |
| poulenc    | flutesonata-1 | unkown1     |       2 |
| mahler     | symphony6-1   | zander      |       1 |
| beethoven  | symphony7-1   | karajan     |       1 |

### Add volume as a channel, then simply flatten the arrays, downsampled to ~1 Hz by mean (CORRECTED). (100)

fraction found: **0.82**	average first match: **0.8**	average average distance: **0.985**

The number of misses caused by each entry. In other words, how often did each entry score best but was an incorrect match.

| composer   | piece         | performer   |   count |
|------------|---------------|-------------|---------|
| mahler     | symphony3-3   | zander      |      13 |
| stravinsky | spring-finale | stravinsky  |       2 |
| stravinsky | spring-finale | ozawa       |       1 |
| beethoven  | symphony7-1   | gardner     |       1 |
| prokofiev  | flute         | adorjan     |       1 |