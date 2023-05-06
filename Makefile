create_profiling:
	@echo "Creating profiling in reports folder..."
	@python scripts/create_profiling.py

process_data:
	@echo "Processing data..."
	@python scripts/process_data.py
	@echo "File saved in data folder!"
