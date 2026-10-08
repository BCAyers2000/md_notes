#=============================================================================!
#                                                                             !
#  BOOK BUILD                                                                 !
#                                                                             !
#  make            full build into build/book.pdf                             !
#  make chapter    one chapter alone, e.g. make chapter CH=04                 !
#  make figures    rerun every figure script under ../scripts                 !
#  make clean      remove build artefacts, keep the PDF                       !
#                                                                             !
#=============================================================================!

BUILD   := build
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error
PYTHON  := python

.PHONY: all chapter figures clean

all:
	$(LATEXMK) book.tex
	@grep -E '^(LaTeX|Package).*Warning' $(BUILD)/book.log | \
	   grep -v 'Font shape' || true

#-----------------------------------------------------------------------------!
#  SINGLE CHAPTER                                                             !
#  The driver is written into build/ and thrown away. Cross-references out    !
#  of the chapter print as ??; use the full build to check those.             !
#-----------------------------------------------------------------------------!
chapter:
	$(PYTHON) ../scripts/build_chapter.py "$(CH)"
	$(LATEXMK) -jobname=ch$(CH) $(BUILD)/ch$(CH).tex

figures:
	@for f in ../scripts/ch*/fig_*.py; do echo "$$f"; $(PYTHON) "$$f" || exit 1; done

clean:
	latexmk -c -outdir=$(BUILD) book.tex
