import airsim
import time
import random
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
# Arm the drone
client.enableApiControl(True)
client.armDisarm(True)


while True:
    client.moveToPositionAsync(5487.372, 3550.54, -71.77,1).join()

    quad_state = client.getMultirotorState().kinematics_estimated.position
    print(quad_state.x_val, quad_state.y_val, quad_state.z_val)
    time.sleep(5)


print("Take off")
client.takeoffAsync().join()  # Wait for takeoff to complete
time.sleep(5)
print("GO")
# Move the drone (roll_rate, pitch_rate, yaw_rate, throttle, duration, vehicle_name='')


# yaw_angle = random.randrange(-1, 2)
client.moveByAngleRatesThrottleAsync(0, 0.2, 1, 0.61, 1).join() # Wait for movement to complete

time.sleep(0.2)
quad_state = client.getMultirotorState().kinematics_estimated.position
print(quad_state.z_val)
client.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -1.78, 3).join()
print("Stop")
# Take off to a certain altitude

# Hover for a few seconds
# airsim.time.sleep(1)

# Land the drone
# client.landAsync().join()

time.sleep(10)
# # Disarm the drone
# client.armDisarm(False)

# # Release the connection
# client.enableApiControl(False)