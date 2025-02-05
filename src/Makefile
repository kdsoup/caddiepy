
MLCOMP?=mlkit

.PHONY: all
all: cad

.PHONY: cad
cad:
	$(MLCOMP) -output $@ cad.mlb

.PHONY: test
test:
	@echo "-------T E S T --- R E P O R T-------"
	-@$(MAKE) -C test/input
	-@$(MAKE) -C test/output

.PHONY: clean
clean:
	rm -rf *~ MLB *.exe run cad
	$(MAKE) -C test/input clean
	$(MAKE) -C test/output clean

.PHONY: realclean
realclean: clean
	rm -rf lib/github.com/diku-dk/sml-parse/MLB
	rm -rf lib/github.com/diku-dk/sml-uref/MLB
