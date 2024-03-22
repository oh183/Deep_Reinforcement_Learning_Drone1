import airsim
import inputs
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Define functions to handle controller input
def get_controller_state():
    left_thumb_y = 0
    left_thumb_x = 0
    right_thumb_x = 0
    right_trigger = 0

    events = inputs.get_gamepad()
    for event in events:
        if event.ev_type == 'Absolute' and event.code == 'ABS_Y':
            left_thumb_y = event.state / 32767.0  # Normalize to range [-1, 1]
        elif event.ev_type == 'Absolute' and event.code == 'ABS_X':
            left_thumb_x = event.state / 32767.0  # Normalize to range [-1, 1]
        elif event.ev_type == 'Absolute' and event.code == 'ABS_RX':
            right_thumb_x = event.state / 32767.0  # Normalize to range [-1, 1]
        elif event.ev_type == 'Absolute' and event.code == 'ABS_Z':
            right_trigger = event.state / 255.0  # Normalize to range [0, 1]
    
    return left_thumb_y, left_thumb_x, right_thumb_x, right_trigger


while True:
    # Get controller input
    pitch, roll, yaw_rate, throttle = get_controller_state()

    # Move the drone based on controller input
    client.moveByRollPitchYawrateThrottleAsync(roll, pitch, yaw_rate, throttle, duration=0.1).join()

    # Get the drone's current position
    drone_state = client.getMultirotorState()
    drone_position = drone_state.kinematics_estimated.position

    # Print the drone's location
    print("Drone position: ", drone_position)

    # Add a small delay to prevent overwhelming the system
    time.sleep(0.1)