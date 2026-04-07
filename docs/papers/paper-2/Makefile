SHELL := /bin/bash

PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
PDFLATEX_FLAGS ?= -interaction=nonstopmode -halt-on-error -file-line-error

MAIN := manuscript

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN).tex
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf \
	      $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz
