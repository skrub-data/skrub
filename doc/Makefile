# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = skrub
SOURCEDIR     = .
BUILDDIR      = _build

ifneq ($(EXAMPLES_PATTERN),)
    EXAMPLES_PATTERN_OPTS := -D sphinx_gallery_conf.filename_pattern="$(EXAMPLES_PATTERN)"
endif

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS)\
    $(EXAMPLES_PATTERN_OPTS) .
# Options without doctrees (for html-split which uses custom doctree paths)
SPHINXOPTS_NO_DOCTREES = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(EXAMPLES_PATTERN_OPTS) .


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


html:
	# These two lines make the build a bit more lengthy, and the
	# the embedding of images more robust
	rm -rf $(BUILDDIR)/html/_images
	#rm -rf _build/doctrees/
	SKB_TABLE_REPORT_VERBOSITY=0 $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-noplot:
	SKB_TABLE_REPORT_VERBOSITY=0 $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-split:
	# Split build to reduce peak memory usage by separating example execution from doc processing.
	# Step 1 runs examples and generates outputs, step 2 builds complete docs with all cross-references.
	# The final build from step 2 has proper cross-references since sphinx-gallery still processes
	# the gallery structure even with plot_gallery=0 (it just skips code execution).
	@echo "Step 1: Building examples gallery only..."
	rm -rf $(BUILDDIR)/html-examples $(BUILDDIR)/doctrees-examples
	SKB_TABLE_REPORT_VERBOSITY=0 $(SPHINXBUILD) -d $(BUILDDIR)/doctrees-examples -D plot_gallery=1 -D exclude_patterns="['reference/*','modules/*']" \
		-b html $(SPHINXOPTS_NO_DOCTREES) $(BUILDDIR)/html-examples
	@echo
	@echo "Step 2: Building full docs with pre-built examples..."
	rm -rf $(BUILDDIR)/html $(BUILDDIR)/doctrees
	SKB_TABLE_REPORT_VERBOSITY=0 $(SPHINXBUILD) -d $(BUILDDIR)/doctrees -D plot_gallery=0 -b html $(SPHINXOPTS_NO_DOCTREES) $(BUILDDIR)/html
	@echo
	@echo "Step 3: Copying gallery outputs to final build..."
	cp -r $(BUILDDIR)/html-examples/auto_examples $(BUILDDIR)/html/ 2>/dev/null || true
	cp -r $(BUILDDIR)/html-examples/_images $(BUILDDIR)/html/ 2>/dev/null || true
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Linkcheck finished. Results are in $(BUILDDIR)/linkcheck."

linkcheck-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck-noplot
	@echo
	@echo "Linkcheck (no plot) finished. Results are in $(BUILDDIR)/linkcheck-noplot."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf _build/ reference/generated/
	rm -rf _build/ generated_for_index
	rm -f reference/*.rst
