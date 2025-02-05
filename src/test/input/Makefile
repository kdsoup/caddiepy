# COMPILE .PY INPUT FILES

# Testing examples 
# .py input files and compile to type inference (--Ptyped) and internal expression (--Pexp)
# Using original makefile test from caddie

TESTS=add cos exp ln mul neg pow proj simple1 simple2 simple3 sin sub

TESTFILES=$(TESTS:%=%.py)
RESFILES=$(TESTFILES:%.py=%.res)
OUTFILES=$(TESTFILES:%.py=%.out)

CAD?=../../cad

.PHONY: all
all: test out

.PHONY: tst
tst:
	echo $(RESFILES)

.PHONY: out
out: $(OUTFILES)

%.out: %.py $(CAD)
	@$(CAD) --Ptyped --Pexp $< > $@ 2>&1; echo "done" >> /dev/null

%.res: %.out %.out.ok
	@(diff -aq $< $*.out.ok > /dev/null 2>&1; \
         if [ $$? -eq 0 ]; then \
             echo "Test $*: OK" > $@ \
         ; else \
             if [ -e $*.out.ok ]; then \
                echo "Test $*: *** ERR: file $< differs from $*.out.ok ***" > $@ \
             ; else \
                echo "Test $*: *** ERR: file $*.out.ok does not exist ***" > $@ \
             ; fi \
         ; fi)
	@cat $@ > /dev/null
	@cat $@ >> complog.txt

.PHONY: test
test: $(RESFILES)
	@echo ""
	@echo "-------------- I N P U T --------------"
	@$(CAD) --version
	@echo ""
	@cat $(RESFILES)
	@echo ""
	@echo "Tests succeeded:   `grep "OK" $(RESFILES) | wc -l` /`grep "Test" $(RESFILES) | wc -l`"
	@echo "Test errors:       `grep "ERR" $(RESFILES) | wc -l` /`grep "Test" $(RESFILES) | wc -l`"
	@echo "See complog.txt"
	@echo "---------------------------------------"
	@exit `grep "ERR" $(RESFILES) | wc -l`

../cad:
	$(MAKE) -C .. cad

.PHONE: cad
cad: 
	$(MAKE) -C ../.. cad

.PHONY: clean
clean:
	rm -rf *~ *.exe *.res *.out complog.txt __pycache__
