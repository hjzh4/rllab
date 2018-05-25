from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped
import numpy as np
import rospy


class TaskObject(object):
    def __init__(self, name, initial_pos, random_delta_range, resource=None):
        """
        Task Object interface
        :param name: str
        :param initial_pos: geometry_msgs.msg.Point
                object's original position
        :param random_delta_range: [float, float, float]
                positive, the range that would be used in sampling object' new
                start position for every episode. Set it as 0, if you want to keep the
                object's initial_pos for every episode.
        :param resource: str
                the model path(str) for simulation training or ros topic name for real robot training
        """
        self._name = name
        self._resource = resource
        self._initial_pos = initial_pos
        self._random_delta_range = random_delta_range

    @property
    def random_delta_range(self):
        return self._random_delta_range

    @property
    def name(self):
        return self._name

    @property
    def resource(self):
        return self._resource

    @property
    def initial_pos(self):
        return self._initial_pos

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = value


class TaskObjectManager(object):
    def __init__(self):
        """
        Users can use this interface to manage the objects in a specific task
        """
        self._manipulateds = []
        self._targets = []
        self._commons = []

    def add_target(self, task_object):
        self._targets.append(task_object)

    def add_manipulated(self, task_object):
        self._manipulateds.append(task_object)

    def add_common(self, task_object):
        self._commons.append(task_object)

    @property
    def objects(self):
        return self._manipulateds + self._targets + self._commons

    @property
    def manipulateds(self):
        return self._manipulateds

    @property
    def targets(self):
        return self._targets

    def sub_gazebo(self):
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self._gazebo_update_manipulated_states)

    def _gazebo_update_manipulated_states(self, data):
        model_states = data
        model_names = model_states.name

        for manipulated in self._manipulateds:
            manipulated_idx = model_names.index(manipulated.name)
            manipulated_pose = model_states.pose[manipulated_idx]
            manipulated.position = manipulated_pose.position
            manipulated.orientation = manipulated_pose.orientation

    def sub_vicon(self):
        for manipulated in self._manipulateds:
            rospy.Subscriber(manipulated.resource, TransformStamped,
                             self._vicon_update_manipulated_states)

    def _vicon_update_manipulated_states(self, data):
        translation = data.transform.translation
        rotation = data.transform.rotation
        child_frame_id = data.child_frame_id

        for manipulated in self._manipulateds:
            if manipulated.resource == child_frame_id:
                manipulated.position = translation
                manipulated.orientation = rotation

    def get_manipulateds_obs(self):
        manipulateds_pos = np.array([])
        manipulateds_ori = np.array([])

        for manipulated in self._manipulateds:
            pos = np.array([
                manipulated.position.x, manipulated.position.y,
                manipulated.position.z
            ])
            ori = np.array([
                manipulated.orientation.x, manipulated.orientation.y,
                manipulated.orientation.z, manipulated.orientation.w
            ])
            manipulateds_pos = np.concatenate((manipulateds_pos, pos))
            manipulateds_ori = np.concatenate((manipulateds_ori, ori))

        achieved_goal = np.squeeze(manipulateds_pos)

        obs = np.concatenate((manipulateds_pos, manipulateds_ori))

        return {'obs': obs, 'achieved_goal': achieved_goal}
