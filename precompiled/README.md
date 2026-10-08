# Precompiled book

[md_notes.pdf](md_notes.pdf) is a compiled copy of *Molecular Dynamics and
Learned Propagators*. Open or download it to read the book without installing
LaTeX.

The editable source is in [`book/`](../book/). This PDF is a saved snapshot;
after changing the book, refresh it from the repository root:

```sh
make book
cp book/build/book.pdf precompiled/md_notes.pdf
```

The main [README](../README.md#compile-the-book) lists the compilation tools.
