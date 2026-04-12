from sense_hat import SenseHat
import time
import math

sense = SenseHat()

# Make sure IMU parts are enabled
sense.set_imu_config(True, True, True)

print("Rotate the Pi slowly and then quickly. Press Ctrl+C to stop.\n")

while True:
    try:
        gyro_raw = sense.get_gyroscope_raw()
        ori_rad = sense.get_orientation_radians()

        print("gyro_raw dict:", gyro_raw, "| orientation yaw:", round(float(ori_rad.get("yaw", 0.0)), 4))
        time.sleep(0.2)
    except KeyboardInterrupt:
        break