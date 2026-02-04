# SMPL
weblink: https://www.cnblogs.com/sariel-sakura/p/14321818.html
pose parameters: 24*3, 24 body joints, 3-D axis-angle representation(a 3D vector to represent rotation, the direction of the vector is the rotation axis, and the norm of the vector is the rotation angle)


# Sophia facial control
app Arkit live link target: 10.0.0.10:1111
now run demo_llf_control.py, Sophia's face should be corresponde with your face on the app.

# Sophia body control
Already have smpl_visualizer on local computer, added tcp code for running on Sophia's computer.
Run the tcp bridge code first and check connectivity: ss -lntp | grep 5005.

Then run visualizer.
To end the session, quit tcp bridge code and run "lsof -i :5005" to find the PID of the process, then do "kill -9 PID" to kill the port.

The smpl_visualizer.py is not well-structed, it used some virtual(helper) indices that inside the range of 0-num_joints. However, it's a program only for body control, so it can run with problem.

The startup pose of the robot and the web-end pose looks similar, but when performing some combination of rotations, they may differ quite a lot, maybe due to the joints position and DOF difference. E.g, Sophia's elbow has only one DOF, which is to contrl the angle between upper and lower arm, the other rotation DOF is achieved by a motor locates on the upper arm, while the SMPLX model has a well-defined elbow.

LeftShoulderYaw, robot moving speed significantly slower than web-end
