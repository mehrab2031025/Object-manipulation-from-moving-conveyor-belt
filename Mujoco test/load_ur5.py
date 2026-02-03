import mujoco
import mujoco.viewer
import time
import os

# Get the folder where this Python file is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to UR5 XML (adjust if your structure is different)
UR5_XML_PATH = os.path.join(CURRENT_DIR, "mujoco_menagerie-main", "universal_robots_ur5e", "ur5e.xml")


def main():
    print(f"Looking for UR5 at: {UR5_XML_PATH}")

    # Check if file exists
    if not os.path.exists(UR5_XML_PATH):
        print("ERROR: File not found!")
        print("Please check the path and make sure you downloaded the files.")
        return

    print("File found! Loading...")

    # Load the UR5 model
    model = mujoco.MjModel.from_xml_path(UR5_XML_PATH)
    data = mujoco.MjData(model)

    print("UR5 loaded successfully!")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")

    # Print joint names
    print("\nJoints in this robot:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()