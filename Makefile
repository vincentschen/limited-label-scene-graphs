check: 
	isort -rc -c . --skip .env/ --skip .ipynb_checkpoints
	black --check . --exclude .ipynb_checkpoints/ --exclude .env/
	flake8 . --exclude .ipynb_checkpoints/ --exclude .env/


fix:
	isort -rc -c . --skip .env/ --skip .ipynb_checkpoints/
	black . --exclude .ipynb_checkpoints/ --exclude .env/

