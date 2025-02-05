
MLCOMP?=mlkit

.PHONY: all
all: cad

.PHONY: cad
cad:
	$(MLCOMP) -output $@ cad.mlb

.PHONY: test
test:
	@echo "-------T E S T --- R E P O R T-------"
	@echo ""
	@$(MAKE) -C test/output
	@$(MAKE) -C test/input

.PHONY: clean
clean:
	rm -rf *~ MLB *.exe run cad
	# $(MAKE) -C tests clean
	# $(MAKE) -C examples clean
	$(MAKE) -C test/input clean
	$(MAKE) -C test/output clean

.PHONY: realclean
realclean: clean
	rm -rf lib/github.com/diku-dk/sml-parse/MLB
	rm -rf lib/github.com/diku-dk/sml-uref/MLB
