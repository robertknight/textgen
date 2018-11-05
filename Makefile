.pydeps:
	pip install -r requirements.txt
	touch .pydeps

train: .pydeps
	python textgen train data/oliver-twist.txt

generate: .pydeps
	python textgen generate data/oliver-twist.txt

format:
	black codegen/*.py

lint:
	flake8 codegen
