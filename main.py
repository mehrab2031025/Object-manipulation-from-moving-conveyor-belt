# load_ur5.py
import mujoco
import mujoco.viewer
import time

# Change this path to where you downloaded the UR5
# Example: "C:/Users/YourName/Downloads/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
UR5_XML_PATH = "PASTE_YOUR_PATH_HERE/ur5e.xml"


def main():
    # Load the UR5 model
    model = mujoco.MjModel.from_xml_path(UR5_XML_PATH)
    data = mujoco.MjData(model)

    print("UR5 loaded successfully!")
    print(f"Robot has {model.nq} position coordinates (joints + anything else)")
    print(f"Robot has {model.nv} velocity coordinates")
    print(f"Robot has {model.nu} actuators (controllable joints)")

    # Print joint names to understand the robot structure
    print("\nJoint names:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  Joint {i}: {name}")

    # Print actuator names
    print("\nActuator names:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  Actuator {i}: {name}")

    # Run simulation with viewer (like the simulate app, but from Python)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer opened! Close the window to exit.")

        # Run for 10 seconds
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < 10:
            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Small delay to not max out CPU
            time.sleep(0.001)


if __name__ == "__main__":
    main()