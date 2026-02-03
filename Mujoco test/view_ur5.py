import mujoco
import mujoco.viewer
import time

# Same path as before
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UR5_XML_PATH = os.path.join(CURRENT_DIR, "mujoco_menagerie-main", "universal_robots_ur5e", "ur5e.xml")


def main():
    # Load model
    model = mujoco.MjModel.from_xml_path(UR5_XML_PATH)
    data = mujoco.MjData(model)

    print("Opening viewer... Close the window to exit.")

    # Launch the viewer (this opens a 3D window)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Keep the window open for 30 seconds
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < 60:
            # Step the physics (nothing moves yet, just keeps simulation alive)
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Small delay
            time.sleep(0.001)

    print("Viewer closed.")


if __name__ == "__main__":
    main()