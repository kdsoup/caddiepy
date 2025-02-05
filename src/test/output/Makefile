
# CADDIEPY

# ### PYTHON COMPILE TESTS ###
# Testing if python input file differentiates to python output code
# and if the python output code can compile according to python syntax.
# If python code comiles using py_compile the .res files will be empty.
# If there's an error, the error will be written to the .res files.

TESTS=add cos exp ln mul neg pow proj simple1 simple2 simple3 sin sub

PYFILES=$(TESTS:%=%_out.py)
RESFILES=$(TESTS:%=%_out.res)
LOGFILE = complog.txt

CAD?=../../cad

.PHONY: all
all: test

# .PHONY: tst
# tst:
# 	echo $(RESFILES)

# Write python files using Caddiepy with input files from '../input'
%_out.py: ../input/%.py $(CAD)
	@echo "from math import *" >> $@
	@echo "from numpy import dot, cross \n" >> $@
	@$(CAD) --Pdiffu $< >> $@ 2>&1; echo "done" >> /dev/null
	@$(CAD) -r --Pdiffu $< >> $@ 2>&1; echo "done" >> /dev/null

# Check that *_out.py files can compile and write result to complog.txt file
%_out.res: %_out.py
	@(python -m py_compile $< > $@ 2>&1; \
		if [ $$? -eq 0 ]; then \
			echo "Test $*: OK" >> $(LOGFILE) \
		; else \
			echo "Test $*: *** ERR: file $< does not compile to python ***" >> $(LOGFILE) \
		; fi)

# Test results
.PHONY: test
test: $(PYFILES) $(RESFILES)
	@echo "------------O U T P U T--------------"
	@$(CAD) --version
	@echo ""
	@cat $(RESFILES)
	@echo ""
	@echo "Tests succeeded:   `grep "OK" $(LOGFILE) | wc -l` /`grep "Test" $(LOGFILE) | wc -l`"
	@echo "Test errors:       `grep "ERR" $(LOGFILE) | wc -l` /`grep "Test" $(LOGFILE) | wc -l`"
	@echo "See complog.txt"
	@echo "-------------------------------------"
	@exit `grep "ERR" $(LOGFILE) | wc -l`

# Cleanup files
.PHONY: clean
clean:
	rm -rf *~ *.exe *.res complog.txt *.py  __pycache__
