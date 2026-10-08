"""Write a LaTeX driver for a single chapter, preserving its numbering."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chapter", help="two-digit chapter number, for example 04"
    )
    args = parser.parse_args()
    if (
        len(args.chapter) != 2
        or not args.chapter.isascii()
        or not args.chapter.isdigit()
    ):
        parser.error("use a two-digit chapter number, for example 04")

    book = Path(__file__).resolve().parents[1] / "book"
    matches = sorted((book / "chapters").glob(f"ch{args.chapter}_*.tex"))
    if len(matches) != 1:
        parser.error(f"no unique chapter source found for {args.chapter}")

    preamble, separator, _ = (
        (book / "book.tex").read_text().partition(r"\begin{document}")
    )
    if not separator:
        parser.error("book.tex has no document environment")
    chapter = matches[0].relative_to(book).with_suffix("").as_posix()
    driver = (
        preamble
        + "\\begin{document}\n\\mainmatter\n"
        + f"\\setcounter{{chapter}}{{{int(args.chapter) - 1}}}\n"
        + f"\\input{{{chapter}}}\n"
        + "\\bibliographystyle{unsrtnat}\n\\bibliography{references}\n"
        + "\\end{document}\n"
    )
    destination = book / "build" / f"ch{args.chapter}.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(driver)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
