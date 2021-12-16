# Music Matching

An exploration of techiniques for identifying different performances of a piece of music. Much of the data was derived
from my own music collection and thus the data itself is not included here.

You can explore the results [here](RESULTS.md).

## Report

### The Problem

I listen to a lot of Western Classical music (everything from Josquin Des Prez to Kaija Saariaho). In Western classical
music, the same piece of music, once written down (composed) will be played by many different individuals/groups.

The goal is, given a corpus of recordings of (presumably) Western Classical music, can you take a roughly 20s snippet
from one performance and identify the others.

### The data

There are many datasets available for this problem, but I ended up using a selection of music from my own collection. I
have made a pickle of the derived `song_data` features that I extracted from the files, and which are used in all my
code. This data includes the names of the original wav files, which can be easily parsed with given functions into 
standardized composer, piece, and performer labels, the chroma "C", and the shingles "S_D", as well as the "rms" and
the "volume".

### My Approaches

My focus was on different embedding techniques, taking the chroma for that first three minutes, breaking it into 20s
chunks, and down-sampling it by various methods (listed in RESULTS.md), and applying PCA. I also used UMAP to evaluate
the distances between different pieces of music, observing fascinating ropes that form from neighboring shingles, making
each piece into a few strings, with performances often overlapping (when the embedding was a good one, anyways).

A significant part of my approach was implementing everything in a way that would allow me to rapidly test new ideas.
This was a front-heavy process which means I had limited time to really explore different ideas, but given more time
I could now rapidly test a host of approaches, including many I had hoped to touch on, such as using neural nets to
learn better embeddings.

### Results

Overall, without PCA, down-sampling with the mean was the top performer, and significantly reduced the time required.
Applying PCA reduced the effectiveness of the matching slightly, but also made the searches considerably faster. This
is a trade-off that would have to be decided in the application.

### My Contributions

I leveraged `librosa` for parsing the wav files and generating the chroma's. I used the standard `umap-learn` package
for UMAP and `sklearn` for PCA. Everything else is my own work.

## Sources

- Zalkow, F.; Müller, M. Learning Low-Dimensional Embeddings of Audio Shingles for Cross-Version Retrieval of Classical Music. Appl. Sci. 2020, 10, 19. https://doi.org/10.3390/app10010019
- https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html
