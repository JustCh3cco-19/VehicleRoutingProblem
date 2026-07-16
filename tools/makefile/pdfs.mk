# Centralized LaTeX document build rules.
LATEX ?= pdflatex
LATEX_FLAGS ?= -interaction=nonstopmode -halt-on-error

PAPER_TEX := docs/paper/main.tex
PAPER_SECTIONS := $(wildcard docs/paper/sections/*.tex)
PAPER_FIGURES := \
	docs/paper/figures/scaling/strong_scaling_runtime.pdf \
	docs/paper/figures/scaling/strong_scaling_speedup.pdf \
	docs/paper/figures/scaling/strong_scaling_efficiency.pdf \
	docs/paper/figures/scaling/weak_scaling_runtime.pdf \
	docs/paper/figures/scaling/weak_scaling_efficiency.pdf \
	docs/paper/figures/comparison/backend_runtime.pdf \
	docs/paper/figures/comparison/backend_cost.pdf
PAPER_PDF := report.pdf

PDF_TARGETS := $(PAPER_PDF)

pdfs: $(PDF_TARGETS)

$(PAPER_PDF): $(PAPER_TEX) $(PAPER_SECTIONS) $(PAPER_FIGURES)
	@set -eu; \
	build_dir=$$(mktemp -d); \
	trap 'rm -rf "$$build_dir"' EXIT; \
	cd docs/paper; \
	$(LATEX) $(LATEX_FLAGS) -jobname=report -output-directory="$$build_dir" main.tex; \
	$(LATEX) $(LATEX_FLAGS) -jobname=report -output-directory="$$build_dir" main.tex; \
	cp "$$build_dir/report.pdf" ../../report.pdf

clean_pdfs:
	rm -f $(PDF_TARGETS)
	rm -f docs/paper/*.aux docs/paper/*.log docs/paper/*.out docs/paper/*.toc docs/paper/*.lof \
		docs/paper/*.lot docs/paper/*.fdb_latexmk docs/paper/*.fls docs/paper/*.synctex.gz
