.PHONY: all
all: test -i

.PHONY: test
test:
	@python -m unittest -v > log.txt 2>&1

.PHONY: clean
clean:
	rm -rf log.txt __pycache__

