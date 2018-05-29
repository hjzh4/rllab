"""
Sawyer Interface
"""

from intera_core_msgs.msg import JointLimits
import intera_interface
import numpy as np
import rospy

from rllab.spaces import Box

from contrib.ros.robots.robot import Robot


class Sawyer(Robot):
    def __init__(self, initial_joint_pos, control_mode):
        """
        :param initial_joint_pos: {str: float}
                            {'joint_name': position_value}, the joints in 'joint_name' list
                            are what we are trying to control.
        """
        Robot.__init__(self)
        self._limb = intera_interface.Limb('right')
        self._gripper = intera_interface.Gripper()
        self._initial_joint_pos = initial_joint_pos
        self._control_mode = control_mode

    @property
    def enabled(self):
        return intera_interface.RobotEnable(
            intera_interface.CHECK_VERSION).state().enabled

    def _set_limb_joint_positions(self, joint_angle_cmds):
        self._limb.set_joint_positions(joint_angle_cmds)

    def _set_limb_joint_velocities(self, joint_angle_cmds):
        self._limb.set_joint_velocities(joint_angle_cmds)

    def _set_limb_joint_torques(self, joint_angle_cmds):
        self._limb.set_joint_torques(joint_angle_cmds)

    def _set_gripper_position(self, position):
        self._gripper.set_position(position)

    def move_to_start_position(self):
        if rospy.is_shutdown():
            return
        self._limb.move_to_joint_positions(
            self._initial_joint_pos, timeout=5.0)
        self._gripper.open()
        rospy.sleep(1.0)

    def get_obs(self):
        # gripper's information
        gripper_pos = np.array(self._limb.endpoint_pose()['position'])
        gripper_ori = np.array(self._limb.endpoint_pose()['orientation'])
        gripper_lvel = np.array(self._limb.endpoint_velocity()['linear'])
        gripper_avel = np.array(self._limb.endpoint_velocity()['angular'])
        gripper_force = np.array(self._limb.endpoint_effort()['force'])
        gripper_torque = np.array(self._limb.endpoint_effort()['torque'])

        # robot joints angles
        robot_joint_angles = np.array(list(self._limb.joint_angles().values()))

        obs = np.concatenate(
            (gripper_pos, gripper_ori, gripper_lvel, gripper_avel,
             gripper_force, gripper_torque, robot_joint_angles))
        return obs

    def set_command(self, commands):
        """
        :param commands: [float]
                    list of command for different joints and gripper
        """
        # the order is the same with the order in initial_joint_pos
        joint_commands = {
            'right_j0': commands[0],
            'right_j1': commands[1],
            'right_j2': commands[2],
            'right_j3': commands[3],
            'right_j4': commands[4],
            'right_j5': commands[5],
            'right_j6': commands[6]
        }
        if self._control_mode == 'position':
            self._set_limb_joint_positions(joint_commands)
        elif self._control_mode == 'velocity':
            self._set_limb_joint_velocities(joint_commands)
        elif self._control_mode == 'effort':
            self._set_limb_joint_torques(joint_commands)

        self._set_gripper_position(commands[7])

    @property
    def gripper_pose(self):
        return self._limb.endpoint_pose()

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        # For sawyer's joints
        # [head_pan, right_gripper_l_finger_joint, right_gripper_r_finger_joint, right_j0, right_j1,
        # right_j2, right_j3, right_j4, right_j5, right_j6]
        # For our use
        # joints[3:10]
        # set the action space depending on different control modes
        joint_limits = rospy.wait_for_message('/robot/joint_limits',
                                              JointLimits)
        lower_bounds = np.array([])
        upper_bounds = np.array([])
        for joint in self._initial_joint_pos:
            joint_idx = joint_limits.joint_names.index(joint)
            if self._control_mode == 'position':
                lower_bounds = np.concatenate((
                    lower_bounds,
                    np.array(
                        joint_limits.position_lower[joint_idx:joint_idx + 1])))
                upper_bounds = np.concatenate((
                    upper_bounds,
                    np.array(
                        joint_limits.position_upper[joint_idx:joint_idx + 1])))
            elif self._control_mode == 'velocity':
                lower_bounds = np.concatenate((lower_bounds, np.zeros(1)))
                upper_bounds = np.concatenate(
                    (upper_bounds,
                     np.array(joint_limits.velocity[joint_idx:joint_idx + 1])))
            elif self._control_mode == 'effort':
                lower_bounds = np.concatenate((lower_bounds, np.zeros(1)))
                upper_bounds = np.concatenate(
                    (upper_bounds,
                     np.array(joint_limits.effort[joint_idx:joint_idx + 1])))
            else:
                raise ValueError(
                    'Control mode %s is not known!' % self._control_mode)
        return Box(
            np.concatenate((lower_bounds, np.array([0]))),
            np.concatenate((upper_bounds, np.array([100]))))
