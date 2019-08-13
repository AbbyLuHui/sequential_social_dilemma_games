import os
import unittest

from rollout import Controller


class TestRollout(unittest.TestCase):
    def setUp(self):
        self.controller_explore = Controller(env_name="explore")
        self.controller = Controller()

    def test_rollouts(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.controller_explore.render_rollout(horizon=50, path=path)
        self.controller.render_rollout(horizon=80, path=path)

        # cleanup
        if os.path.exists("trajectory.mp4"):
            os.remove("trajectory.mp4")


if __name__ == '__main__':
    unittest.main()
