import mujoco
import mujoco.viewer
import numpy as np
import time
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UR5_XML_PATH = os.path.join(CURRENT_DIR, "ur5e_robot", "ur5e.xml")


def main():
    model = mujoco.MjModel.from_xml_path(UR5_XML_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < 100:

            t = time.time() - start_time

            # Move all 6 joints with different speeds (wave motion)
            for i in range(6):
                data.ctrl[i] = np.sin(t * 2 + i) * 0.5  # Each joint offset by i

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()