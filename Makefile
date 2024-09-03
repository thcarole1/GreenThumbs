run:
	@python GreenThumbs_package_folder/main.py

run_uvicorn:
	@uvicorn GreenThumbs_package_folder.api:app --reload

install:
	@pip install -e .

test:
	@pytest -v tests
