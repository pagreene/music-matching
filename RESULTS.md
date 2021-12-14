# Results

## Summary

| Method Description                                                           |   P_f |   <n> |   <<d>> |
|------------------------------------------------------------------------------|-------|-------|---------|
| Simply flatten the arrays, at full resolution.                               |  0.5  |  0.5  |   4.6   |
| Add volume as a channel, then simply flatten the arrays, at full resolution. |  0.7  |  0.7  |   1.65  |
| Downsample and then flatten the standard arrays (no volume).                 |  0.7  |  0.7  |   1.7   |
| Downsample and then flatten the standard arrays (no volume).                 |  0.57 |  0.52 |   3.535 |

## Run Details

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

### Downsample and then flatten the standard arrays (no volume).

```python
def f(sd):
    """Downsample and then flatten the standard arrays (no volume)."""
    C_ds = sd['C'][:, ::43]
    S_D = make_shingles(C_ds, 20, 180)
    sh = S_D.shape
    return S_D.reshape(sh[0], sh[1]*sh[2])

```
