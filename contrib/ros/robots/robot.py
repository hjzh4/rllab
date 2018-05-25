"""
robot Interface
"""


class Robot(object):
    def move_to_start_position(self):
        raise NotImplementedError

    def set_command(self, commands):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError