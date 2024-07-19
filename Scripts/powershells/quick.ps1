# Stage the deleted, modified, and new files
git add Documents/Journal/data_fetch_tab\ \(1.0\)\ .png Documents/Journal/data_fetch_tab\ \(2.0\)\ .png \
Documents/Journal/week\ 1\ entry\ 1\ 07\ -\ 3\ -\ 2024 Documents/Journal/week\ 1\ entry\ 2\ 07\ -\ 6\ -\ 2024 \
Documents/Journal/week\ 1\ entry\ 3\ 07\ -\ 7\ -\ 2024 Documents/Journal/week\ 1\ entry\ 4\ 07\ -\ 8\ -\ 2024 \
Documents/Journal/week\ 1\ entry\ 5\ 07\ -\ 12\ -\ 2024\ no\ power\ start \
Documents/Journal/week\ 1\ entry\ 6\ 07\ -\ 14\ -\ 2024\ no\ power \
Documents/Journal/week\ 1\ entry\ 7\ 07\ -\ 15\ -\ 2024\ no\ power \
Documents/Journal/week\ 1\ entry\ 8\ 07\ -\ 16\ -\ 2024 \
Documents/Journal/week\ 2\ entry\ 9\ 07\ -\ 17\ -\ 2024 \
Documents/Journal/week\ 2\ entry\ 9\ 07\ -\ 17\ -\ 2024\ B \
Tests/Data_Processing/ Tests/Utilities/test_DataLakeHandler.py \
Tests/Utilities/test_data_processing_utils.py Scripts/Data_Processing/Technical_indicators/custom_indicators.py \
Scripts/GUI/data_fetch_tab.py Scripts/Utilities/data_processing_utils.py \
Scripts/powershells/quick.ps1 data/alpha_vantage/tsla_data.csv \
data/trading_data.db logs/alpha_vantage.log logs/data_store.log

# Remove the deleted files from git tracking
git rm Documents/Journal/data\ fetch\ tab\ \(preview\).png \
Documents/Journal/entry\ 1\ 07\ -\ 3\ -\ 2024 \
Documents/Journal/entry\ 2\ 07\ -\ 6\ -\ 2024 \
Documents/Journal/entry\ 3\ 07\ -\ 7\ -\ 2024 \
Documents/Journal/entry\ 4\ 07\ -\ 8\ -\ 2024 \
Documents/Journal/entry\ 5\ 07\ -\ 12\ -\ 2024\ no\ power\ start \
Documents/Journal/entry\ 6\ 07\ -\ 14\ -\ 2024\ no\ power \
Documents/Journal/entry\ 7\ 07\ -\ 15\ -\ 2024\ no\ power \
Documents/Journal/entry\ 8\ 07\ -\ 16\ -\ 2024

# Commit the changes with a detailed message
git commit -m "Major updates and cleanup on July 17, 2024:
- Addressed and resolved ModuleNotFoundError issues.
- Enhanced the GUI with Azure theme and added tabs for AlphaVantage, Nasdaq, and Polygon data fetchers.
- Implemented threading for asynchronous data fetching in FetcherGUI.
- Improved test scripts, configured environment variables, and ensured correct project path setup.
- Developed and tested methods for constructing API URLs and extracting results from API responses.
- Enhanced error handling and logging across modules.
- Began integration with CI/CD pipeline for automated testing and deployment.
- Updated project documentation and cleaned up extra CSV files.
- Created databank to organize data more effectively.
- Deleted outdated journal entries and preview images.
- Added new journal entries for detailed documentation."

# Push the changes to the repository
git push origin master
