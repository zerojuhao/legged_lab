import joblib
import os

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
motion_file = os.path.join(LEGGED_LAB_ROOT_DIR, "data", "g1", "my_walk.pkl")

assert os.path.exists(motion_file), f"Motion file {motion_file} does not exist, please check the file."

motion_data_dict = joblib.load(motion_file)
motion_names = list(motion_data_dict.keys())
print("Motion names:")
for name in motion_names:
    print(f'"{name}"')
    