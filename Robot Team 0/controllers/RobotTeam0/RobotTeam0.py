"""RobotTeam0 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor, Camera


# create the Robot instance.
robot = Robot()

WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1


# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
camera = robot.getCamera('camera1')
camera.enable(timestep)
camera.recognitionEnable(timestep)

leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))


    

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    # img_arr = camera.getImageArray()
    # for x in range(0, camera.getWidth()):
        # for y in range(0, camera.getHeight()):
            # red = img_arr[x][y][0]
            # green = img_arr[x][y][1]
            # blue = img_arr[x][y][2]
            # print("red="+str(red)+" blue="+str(blue)+" green="+str(green))
    # print(img_arr)
    
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
