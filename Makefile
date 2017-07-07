HOST=127.0.0.1
TEST_PATH=./

all:

test: unit_test

unit_test:
	python -m unittest discover --pattern="*_test.py" -v

version:
	python -c "import miniflow; print(miniflow)"

smoke: version
	./tests/smoke_tests.py
