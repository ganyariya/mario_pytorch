import gym


class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super(CustomRewardEnv, self).__init__(env)
        self._current_x = 0
        self._max_x = 0
        self.reward = 0

    def reset(self, **kwargs):
        self._current_x = 0
        self._max_x = 0
        self.reward = 0
        return self.env.reset(**kwargs)

    # ステップ
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        print(reward, info)

        if info["x_pos"] > self._current_x:
            self.reward += 1
        else:
            self.reward -= 1

        self.reward /= 1000
        self._current_x = info["x_pos"]

        if info["life"] <= 1:
            self.reward -= 0.3

        if info["life"] == 1:
            done = True

        # エピソード完了の変更
        if info["flag_get"]:
            self.reward += 2
            done = True

        print(self.reward)
        return state, self.reward, done, info
