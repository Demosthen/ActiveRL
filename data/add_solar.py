# %%
import pandas as pd
import os
import sys


# %%
folder_names = [".old/Test_cold_Texas/", ".old/Test_dry_Cali/", ".old/Test_hot_new_york/",".old/Test_snowy_Cali_winter/"]
output_folders = ["Test_cold_Texas", "Test_dry_Cali", "Test_hot_new_york", "Test_snowy_Cali_winter"]
for folder_name, output_folder in zip(folder_names, output_folders):
    # %%
    buildings = []
    building_paths = []
    solar = None
    weather = None
    weather_path = None
    for file_name in os.listdir(folder_name):
        file = os.path.join(folder_name, file_name)
        out_file = os.path.join(output_folder, file_name)
        if "Building" in file_name:
            buildings.append(pd.read_csv(file))
            building_paths.append(file_name)
        elif "solar" in file_name:
            solar = pd.read_csv(file)
            solar = solar.drop('Unnamed: 0', axis=1)
            solar.to_csv(out_file, index=False)
        elif "weather" in file_name:
            weather = pd.read_csv(file)
            weather = weather.drop('Unnamed: 0', axis=1)
            weather.to_csv(out_file, index=False) 

    # %%
    for i in range(len(buildings)):
        buildings[i] = buildings[i].drop('Unnamed: 0', axis=1)
        buildings[i]["Heating Load [kWh]"] = 0
        buildings[i]["Solar Generation [W/kW]"] = solar["Hourly Data: AC inverter power (W)"]

    # %%
    for building, building_path in zip(buildings, building_paths):
        full_building_path = os.path.join(output_folder, building_path)
        building.to_csv(full_building_path, index=False)

    # %%



