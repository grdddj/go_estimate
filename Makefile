style:
	isort --profile black *.py
	black *.py

check:
	flake8 --ignore=E501 *.py
	mypy *.py
