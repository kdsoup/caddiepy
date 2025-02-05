
# CADDIEPY

# ### PYTHON COMPILE TESTS ###
# Testing if python input file differentiates to python output code
# and if the python output code can compile according to python syntax.
# If python code comiles using py_compile the .res files will be empty.
# If there's an error, the error will be written to the .res files.

TESTS=add cos exp ln mul neg pow proj simple1 simple2 simple3 sin sub

OUTFILES=$(TESTS:%=%_out.py)
RESFILES=$(TESTS:%=%_out.res)
LOGFILE = complog.txt

RUNFILES=$(TESTS:%=%_run.py)

CAD?=../../cad

.PHONY: all
all: test

# Write python files using Caddiepy with input files from '../input'
%_out.py: ../input/%.py $(CAD)
	@$(CAD) --Pdiffu $< >> $@ 2>&1; echo "done" >> /dev/null
	@$(CAD) -r --Pdiffu $< >> $@ 2>&1; echo "done" >> /dev/null

# Check that *_out.py files can compile and write result to complog.txt file
# %_out.res: %_out.py
# 	@(python -m py_compile $< > $@ 2>&1; \
# 		if [ $$? -eq 0 ]; then \
# 			echo "Test $*: OK" >> $(LOGFILE) \
# 		; else \
# 			echo "Test $*: *** ERR: file $< does not compile to python ***" >> $(LOGFILE) \
# 		; fi)

# Check generated _out.py files against the expected _out.ok files and write to log
%_out.res: %_out.py %_out.ok
	@(diff -aq $< $*_out.ok > /dev/null 2>&1; \
         if [ $$? -eq 0 ]; then \
             echo "Test $*: OK" > $@ \
         ; else \
             if [ -e $*_out.ok ]; then \
                echo "Test $*: *** ERR: file $< differs from $*_out.ok ***" > $@ \
             ; else \
                echo "Test $*: *** ERR: file $*_out.ok does not exist ***" > $@ \
             ; fi \
         ; fi)
	@cat $@ > /dev/null
	@cat $@ >> complog.txt

# Test results
.PHONY: test
test: $(OUTFILES) $(RESFILES)
	@echo ""
	@echo "------------ O U T P U T --------------"
	@$(CAD) --version
	@echo ""
	@cat $(RESFILES)
	@echo ""
	@echo "Tests succeeded:   `grep "OK" $(LOGFILE) | wc -l` /`grep "Test" $(LOGFILE) | wc -l`"
	@echo "Test errors:       `grep "ERR" $(LOGFILE) | wc -l` /`grep "Test" $(LOGFILE) | wc -l`"
	@echo "See complog.txt"
	@echo "---------------------------------------"
	@exit `grep "ERR" $(LOGFILE) | wc -l`



# Make run files with import numpy and math library to easily import the files in python interpreter
.PHONY: runfiles
runfiles: $(RUNFILES)

# Write python files using Caddiepy with input files from '../input'
# These files have the math and numpy libraries imported
%_run.py: ../input/%.py $(CAD)
	@echo "from math import *" >> $@
	@echo "from numpy import dot, cross \n" >> $@
	@$(CAD) --Pdiffu $< >> $@ 2>&1; echo "done" >> /dev/null
	@$(CAD) -r --Pdiffu $< >> $@ 2>&1; echo "done" >> /dev/null



# Cleanup files
.PHONY: clean
clean:
	rm -rf *~ *.exe *.res complog.txt *.py __pycache__
