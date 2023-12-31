import pandas as pd
import shutil
import glob
import os
import re


def pline():
    print("\n")
    print("=" * 30)
    print("\n")


def create_df_from_json(data, main_keys):
    df = pd.DataFrame()

    for main_key in main_keys:
        df_temp = pd.DataFrame()
        vals = []
        try:
            for key, value in data[main_key][0].items():
                vals.append(f"{key}: {value}")
            vals.append(f"======length: {len(data[main_key])}")
        except:
            try:
                for key, value in data[main_key].items():
                    vals.append(f"{key}: {value}")
                vals.append(f"======length: 1")
            except:
                vals.append(data[main_key])
            vals.append(f"======length: int")

        df_temp[main_key] = vals
        df = pd.concat([df, df_temp], axis=1)

    return df


def extract_copy_csv(year, source_dir, dest_dir):

    print(f"{year} processing started")
    players = set()

    for filepath in glob.glob(os.path.join(source_dir, "*/gw.csv")):
        player_name_with_tag = os.path.basename(os.path.dirname(filepath))
        player_name = re.sub(
            r"_\d+", "", player_name_with_tag
        )  # Remove the numeric tag

        if player_name in players:
            print(f"Duplicate player found: {player_name}")
            continue

        players.add(player_name)
        new_filename = f"{player_name}_20{year}.csv"
        shutil.copy(filepath, os.path.join(dest_dir, new_filename))

    print(f"{year} processing complete.\n")
