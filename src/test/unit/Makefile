.PHONY: all
all: test -i

.PHONY: test
test:
	@echo ""
	@echo "-------------- U N I T ----------------"
	- @python -m unittest -v > log.txt 2>&1
	@python -m unittest -v

.PHONY: clean
clean:
	rm -rf log.txt __pycache__

