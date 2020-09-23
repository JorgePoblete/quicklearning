PIP=pip3.8
PYTHON=python3.8

clean:
	@rm -f dist/*

build: clean
	@${PYTHON} setup.py bdist_wheel

upload: build
	@${PYTHON} -m twine upload dist/*

upgrade:
	@${PIP} install --quiet --upgrade quicklearning
	@${PIP} install --quiet --upgrade quicklearning
	@${PIP} show quicklearning