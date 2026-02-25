SHELL := /bin/bash

PDFLATEX ?= pdflatex
PDFLATEX_FLAGS ?= -interaction=nonstopmode -halt-on-error -file-line-error

TEXS := $(wildcard *.tex)
PDFS := $(TEXS:.tex=.pdf)

.PHONY: all clean

all: $(PDFS)

%.pdf: %.tex
	$(PDFLATEX) $(PDFLATEX_FLAGS) $<
	$(PDFLATEX) $(PDFLATEX_FLAGS) $<

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb *.bbl *.blg
