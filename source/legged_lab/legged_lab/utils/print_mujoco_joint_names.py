import os
import mujoco
import argparse

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Print MuJoCo joint names.")
parser.add_argument(
    "--scene_path",
    type=str,
    default=os.path.join("g1/mjcf_retargeted/g1.xml"),
    help="Path to the MuJoCo scene file, under `assets` directory.",
)
args = parser.parse_args()

scene_path = os.path.join(LEGGED_LAB_ROOT_DIR, "assets", args.scene_path)
if not os.path.exists(scene_path):
    raise FileNotFoundError(f"Scene file not found: {scene_path}")

model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
]

print("Joint names:")
print(joint_names[1:])
print(len(joint_names[1:])) 