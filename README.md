# FPL MLOps

# Summary
* Project starts with legacy data from github found here [github](https://github.com/vaastav/Fantasy-Premier-League)
    * Only game week data per player is used for now
* To recreate intial latest data, run:
    * src/pipeline/01_extract_players_from_legacy_github_data.py
    * src/pipeline/02_update_legacy_data_with_latest_data.py

* Following this, the script can autoupdate the baseline data with:
    * src/pipeline/03_update_raw_player_data_from_API.py

* To transform the data, run the following scripts. This will be changed to a pipeline in a later version.