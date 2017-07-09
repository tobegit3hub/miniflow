HOST=127.0.0.1
TEST_PATH=./

all:

test: unittest

unittest:
	python -m unittest discover --pattern="*_test.py" -v

pytest:
	pytest --showlocals --durations=1 --pyargs

version:
	python -c "import miniflow; print(miniflow)"

smoke: version
	./tests/smoke_tests.py
