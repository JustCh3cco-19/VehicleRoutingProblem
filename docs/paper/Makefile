LATEX := pdflatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error

SOURCES := main.tex $(wildcard sections/*.tex)
FIGURES := \
	figures/scaling/strong_scaling_runtime.pdf \
	figures/scaling/strong_scaling_speedup.pdf \
	figures/scaling/strong_scaling_efficiency.pdf \
	figures/scaling/weak_scaling_runtime.pdf \
	figures/scaling/weak_scaling_efficiency.pdf \
	figures/comparison/backend_runtime.pdf \
	figures/comparison/backend_cost.pdf

.PHONY: all plots clean

all: ../../report.pdf

../../report.pdf: $(SOURCES) $(FIGURES)
	@set -eu; \
	build_dir=$$(mktemp -d); \
	trap 'rm -rf "$$build_dir"' EXIT; \
	$(LATEX) $(LATEXFLAGS) -jobname=report -output-directory="$$build_dir" main.tex; \
	$(LATEX) $(LATEXFLAGS) -jobname=report -output-directory="$$build_dir" main.tex; \
	cp "$$build_dir/report.pdf" "$@"

plots:
	cd ../.. && python3 docs/paper/scripts/analyze_results.py --root .

clean:
	rm -f main.aux main.log main.out main.toc main.lof main.lot main.pdf report.pdf ../../report.pdf
