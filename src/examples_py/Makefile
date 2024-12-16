
# CADDIEPY

# ### PYTHON COMPILE TESTS ###
# Testing if input function differentiates to python output code
# and if the python output code can compile according to python syntax.
# If python code comiles using py_compile the .res files will be empty.
# If there's an error, the error will be written to the .res files.

TESTS=add ba ln num simple smul test0 test1 test2 test3 zero

TESTFILES=$(TESTS:%=%.cad)
RESFILES=$(TESTFILES:%.cad=%.res)
PYFILES=$(TESTFILES:%.cad=%.py)
LOGFILE = complog.txt

CAD?=../cad

.PHONY: all
all: test

.PHONY: tst
tst:
	echo $(RESFILES)

# Write python files using CaddiePy
%.py: %.cad ../cad
	@$(CAD) --Pdiffu $< > $@ 2>&1; echo "done" >> /dev/null
	@$(CAD) -r --Pdiffu $< >> $@ 2>&1; echo "done" >> /dev/null

# Check that .py files can compile and write result to complog.txt file
%.res: %.py
	@(python -m py_compile $< > $@ 2>&1; \
		if [ $$? -eq 0 ]; then \
			echo "Test $*: OK" >> $(LOGFILE) \
		; else \
			echo "Test $*: *** ERR: file $< does not compile to python ***" >> $(LOGFILE) \
		; fi)

.PHONY: test
test: $(PYFILES) $(RESFILES)
	@echo "-------T E S T --- R E P O R T-------"
	@../cad --version
	@echo "Tests succeeded:   `grep "OK" $(LOGFILE) | wc -l` /`grep "Test" $(LOGFILE) | wc -l`"
	@echo "Test errors:       `grep "ERR" $(LOGFILE) | wc -l` /`grep "Test" $(LOGFILE) | wc -l`"
	@echo "See complog.txt"
	@echo "-------------------------------------"
	@exit `grep "ERR" $(LOGFILE) | wc -l`

../cad:
	$(MAKE) -C .. cad

.PHONY: clean
clean:
	rm -rf *~ *.exe *.res *.out complog.txt *.py *.txt __pycache__
