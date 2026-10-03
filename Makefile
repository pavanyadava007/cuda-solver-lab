all:
	./scripts/build.sh

test: all
	cd build && ctest --output-on-failure

bench:
	./run_all.sh

report:
	.venv/bin/python scripts/make_report.py

clean:
	rm -rf build

.PHONY: all test bench report clean
