# Saved trajectories

The short examples run in the notebooks. Longer trajectories are saved here
so that you can spend your time analysing them and changing the examples.
The saved figures are included in the repository: compiling the book does
not require downloading any trajectories.

Small data files, the water and energy-exchange animations, and starting
structures are included in Git. The larger files are distributed as 17
ordinary ZIP files on the repository's `data-v1` release. The complete
download is about 9.2 GiB. Most of it belongs to Chapter 13's independent
lithium trajectories; Chapters 0–11 need no download.

## Get the data

Run these commands from the repository root. The helper needs only Python's
standard library.

```sh
python scripts/manage_data.py list
python scripts/manage_data.py fetch
python scripts/manage_data.py check
```

`fetch` reads the GitHub repository from your Git `origin`. If you obtained
the book as a ZIP, supply the repository explicitly:

```sh
python scripts/manage_data.py fetch --repo OWNER/REPOSITORY
```

Replace `OWNER/REPOSITORY` with this book's GitHub repository. The maintainer
must publish the data release before downloads are available. The script
does not create a release or upload anything.

To download less, select the data folders needed by a notebook:

| Notebook | Command after `python scripts/manage_data.py` |
| --- | --- |
| 00–11 | No download needed |
| 12 | `fetch 12` |
| 13 | `fetch 12 13` |
| 14 | `fetch 13 14` |
| 15 | `fetch 15` |
| 16 | `fetch 15 16` |
| 17 | `fetch 12 17` |
| 18 | `fetch 18` |
| F | `fetch 12 13` |

A number selects the whole chapter's data folder, including data for its
figure scripts and alternative runs. Selection does not automatically add
earlier chapters. The table includes the cross-chapter inputs used by the
notebooks. Use `check` with the same numbers to verify a selection.

You can also copy the release archives from another machine and fetch
without a network connection:

```sh
python scripts/manage_data.py fetch 18 --archive-dir /path/to/downloads
```

Downloads are cached in `data/.downloads/`, then unpacked into the chapter
folders. Allow space for both the archives and the extracted files: roughly
19 GiB for the complete collection. Once `check` succeeds, you can remove
the cached ZIP files to recover that space. Existing matching files are
left alone. A changed data file is preserved; move it elsewhere before
fetching again, or use `--replace` when you intend to restore the published
version. Files supplied by Git must be restored from Git.

## What is recorded

`manifest.json` records the path, byte count and SHA-256 of every distributed
file, plus the corresponding information for each archive. Fetching checks
the archive and its members before replacing an individual file. The ZIP
files retain the original arrays: no trajectories are shortened, rounded or
resampled for distribution. NumPy archives are already compressed, so the
outer ZIP files simply store them.

The `scripts/chNN_*` directories contain the calculations that produced the
data. Repeating these calculations is optional and can be expensive.
Chapter 17's external-calculator runs also need their own software and
model weights; neither is downloaded by this helper. Machine-dependent
timings describe the original runs, not the computer reading the files.

Generated LAMMPS work files, FHI-aims working directories and run logs are
not part of the data collection. The Chapter 17 training snapshots are
included with that chapter for further experiments.

## Prepare a data release

With the original data present and matching the manifest:

```sh
python scripts/manage_data.py check
python scripts/manage_data.py pack
```

`pack` writes deterministic archives to the ignored `release-assets/`
directory, checking their hashes against the manifest. Each archive is
under 1 GiB. Publish those ZIP files as assets of a GitHub release tagged
`data-v1`; keep them out of Git commits. Publishing is a separate manual
step. The repository can be cloned, the book compiled, and Chapters 0–11
used before a data release has been published.

For a later data edition, review the changed source files and regenerate
the manifest and archives together, then use a new release tag. Do not
replace the bytes of a published archive under an existing tag.
