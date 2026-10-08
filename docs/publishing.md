# Publishing the book and saved data

The Git repository contains the text, source code, notebooks, figure PDFs and
small data files. Large trajectories belong to the `data-v1` release. The
release helper checks their bytes against `data/manifest.json`.

## First upload

Create an empty GitHub repository under the intended account, then run these
commands from this project's root, replacing `OWNER/REPOSITORY`:

```sh
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

The initial local commit is already prepared. If the repository already has
an `origin`, check `git remote -v` and use that remote. Select the intended
visibility when creating the remote repository.

The LaTeX class in `book/style/uosthesis.cls` retains its original author and
copyright notices. No project-wide licence has yet been selected; choose the
terms for the book, code and data before offering permission to redistribute
or adapt them, and preserve third-party notices.

## Publish the trajectories

Check the original data, then prepare the archives:

```sh
python scripts/manage_data.py check
python scripts/manage_data.py pack
```

`release-assets/` now contains the ZIP files to upload. It is ignored by Git.
Each archive is under 1 GiB; the collection is about 9.2 GiB. Upload all the
ZIP files to a GitHub release tagged `data-v1`. This can be done in GitHub's
release editor, or with an authenticated GitHub CLI:

```sh
gh release create data-v1 --target main --title "Saved trajectories, version 1" --notes "Original saved trajectories for the textbook; checksums are recorded in data/manifest.json."
gh release upload data-v1 release-assets/*.zip
```

Wait until every asset has uploaded before announcing the data release. In a
fresh clone, test a small download and its checksum:

```sh
python scripts/manage_data.py fetch 18
python scripts/manage_data.py check 18
```

The book and notebooks through Chapter 11 work before the release is
published. The later notebooks require the groups listed in the
[data guide](../data/README.md).

## Later changes

Run `python scripts/setup_git.py` once per clone to install the local notebook
clean filter. Executed output remains in the working files, while the staged
notebooks contain their source and exercise metadata.

```sh
make test
make book
git status --short
git diff --stat
```

Review changes before committing. `data/.gitignore` lists the small inputs
included in Git; newly generated data stay untracked until deliberately
added to that list. Do not force-add the large archives or trajectories.

A changed data edition needs a reviewed manifest, matching archives and a new
release tag. Keep existing published assets unchanged so that older checkouts
can still fetch the data recorded in their manifests. The reader helper
accepts `--tag` for another release.
