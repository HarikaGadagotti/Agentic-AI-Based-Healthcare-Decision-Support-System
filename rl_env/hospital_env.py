import gym
from gym import spaces
import numpy as np

class HospitalEnv(gym.Env):
    def __init__(self):
        super(HospitalEnv, self).__init__()

        # State: [available_icu_beds, available_staff, incoming_patients]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(3,), dtype=np.int32
        )

        # Actions:
        # 0 = Do nothing
        # 1 = Allocate ICU bed
        # 2 = Call extra staff
        self.action_space = spaces.Discrete(3)

        self.state = None

    def reset(self):
        self.state = np.array([
            np.random.randint(5, 20),   # ICU beds
            np.random.randint(5, 20),   # Staff
            np.random.randint(1, 10)    # Incoming patients
        ])
        return self.state

    def step(self, action):
        icu, staff, patients = self.state

        reward = 0

        if action == 1 and icu > 0:
            icu -= 1
            reward = 10
        elif action == 2:
            staff += 1
            reward = 5
        else:
            reward = -1

        patients = max(0, patients - 1)

        self.state = np.array([icu, staff, patients])
        done = False

        return self.state, reward, done, {}
