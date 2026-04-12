from sense_hat import SenseHat
import time
import numpy as np

# Create Sense HAT object
sense = SenseHat()

# Turn on all IMU sensors
sense.set_imu_config(True, True, True)

def read_imu_data():
    # Read accelerometer data
    accel = sense.get_accelerometer_raw()

    # Read gyroscope data
    gyro = sense.get_gyroscope_raw()

    # Save values into lists first (not really needed, but easier to understand)
    accel_list = [accel['x'], accel['y'], accel['z']]
    gyro_list = [gyro['x'], gyro['y'], gyro['z']]

    # Convert lists to numpy arrays
    accel_array = np.array(accel_list)
    gyro_array = np.array(gyro_list)

    # Read orientation (not used in the file, but might be useful later)
    orientation = sense.get_orientation()
    yaw = orientation['yaw']
    pitch = orientation['pitch']
    roll = orientation['roll']

    # Get current time
    current_time = time.time()

    return accel_array, gyro_array, current_time, (yaw, pitch, roll)


# Start logging data
if __name__ == "__main__":
    try:
        # Open CSV file
        file = open("imu_log.csv", "w")

        # Write header
        file.write("timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")

        # Loop for fixed number of samples
        for i in range(1000):
            accel_data, gyro_data, timestamp, orientation = read_imu_data()

            # Write values one by one (not the most efficient way)
            file.write(str(timestamp) + ",")
            file.write(str(accel_data[0]) + ",")
            file.write(str(accel_data[1]) + ",")
            file.write(str(accel_data[2]) + ",")
            file.write(str(gyro_data[0]) + ",")
            file.write(str(gyro_data[1]) + ",")
            file.write(str(gyro_data[2]) + "\n")

            # Small delay between samples
            time.sleep(0.01)

        # Close file manually
        file.close()

    except KeyboardInterrupt:
        print("Program stopped.")
