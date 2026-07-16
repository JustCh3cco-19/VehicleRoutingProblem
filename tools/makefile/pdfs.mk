# Centralized LaTeX document build rules.
LATEX ?= pdflatex
LATEX_FLAGS ?= -interaction=nonstopmode -halt-on-error

BACKEND_DOC_TEX := docs/backends/spiegazione_backend.tex
BACKEND_DOC_PDF := docs/backends/spiegazione_backend.pdf

PAPER_TEX := paper/main.tex
PAPER_SECTIONS := $(wildcard paper/sections/*.tex)
PAPER_FIGURES := \
	paper/figures/scaling/strong_scaling_runtime.pdf \
	paper/figures/scaling/strong_scaling_speedup.pdf \
	paper/figures/scaling/strong_scaling_efficiency.pdf \
	paper/figures/scaling/weak_scaling_runtime.pdf \
	paper/figures/scaling/weak_scaling_efficiency.pdf \
	paper/figures/comparison/backend_runtime.pdf \
	paper/figures/comparison/backend_cost.pdf
PAPER_PDF := report.pdf

PDF_TARGETS := $(BACKEND_DOC_PDF) $(PAPER_PDF)

pdfs: $(PDF_TARGETS)

$(BACKEND_DOC_PDF): $(BACKEND_DOC_TEX)
	@set -eu; \
	build_dir=$$(mktemp -d); \
	trap 'rm -rf "$$build_dir"' EXIT; \
	cd docs/backends; \
	$(LATEX) $(LATEX_FLAGS) -output-directory="$$build_dir" spiegazione_backend.tex; \
	$(LATEX) $(LATEX_FLAGS) -output-directory="$$build_dir" spiegazione_backend.tex; \
	cp "$$build_dir/spiegazione_backend.pdf" spiegazione_backend.pdf

$(PAPER_PDF): $(PAPER_TEX) $(PAPER_SECTIONS) $(PAPER_FIGURES)
	@set -eu; \
	build_dir=$$(mktemp -d); \
	trap 'rm -rf "$$build_dir"' EXIT; \
	cd paper; \
	$(LATEX) $(LATEX_FLAGS) -jobname=report -output-directory="$$build_dir" main.tex; \
	$(LATEX) $(LATEX_FLAGS) -jobname=report -output-directory="$$build_dir" main.tex; \
	cp "$$build_dir/report.pdf" ../report.pdf

clean_pdfs:
	rm -f $(PDF_TARGETS)
	rm -f docs/backends/*.aux docs/backends/*.log docs/backends/*.out \
		docs/backends/*.toc docs/backends/*.fdb_latexmk \
		docs/backends/*.fls docs/backends/*.synctex.gz
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.lof \
		paper/*.lot paper/*.fdb_latexmk paper/*.fls paper/*.synctex.gz
