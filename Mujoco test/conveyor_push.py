import mujoco
import mujoco.viewer
import numpy as np
import time
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "ur5e_robot", "ur5e_with_conveyor.xml")


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("Conveyor pushing object!")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Good viewing angle
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0

        # Find conveyor actuator (should be index 6, after 6 UR5 joints)
        conveyor_act_id = 6  # Adjust if needed

        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < 15:

            t = time.time() - start_time

            # Move robot slowly
            for i in range(6):
                data.ctrl[i] = np.sin(t * 0.3) * 0.2

            # Move conveyor at constant speed (0.1 m/s)
            data.ctrl[conveyor_act_id] = 0.1

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    print("Done!")


if __name__ == "__main__":
    main()