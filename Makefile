create_profiling:
	@echo "Creating profiling in reports folder..."
	@python scripts/create_profiling.py

process_data:
	@echo "Processing data..."
	@python scripts/process_data.py
	@echo "File saved in data folder!"

split_data:
	@echo "Splitting data..."
	@python scripts/split_data.py
	@echo "Files saved in data folder!"