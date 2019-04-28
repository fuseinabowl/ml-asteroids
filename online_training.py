import pyglet

from asteroids import env as environment
from asteroids.renderer import renderer
from asteroids.game import update_result
from asteroids.agents import dqn_agent

class OnlineTraining():

    def main(self):
        self.env = environment.Env()
        self.last_seen_observation = self.env.reset()
        self.agent = dqn_agent.DQNAgent(20, 6)

        self.games_remaining = 10
        
        def update_game():
            player_actions_as_single_value = self.agent.act(self.last_seen_observation)
            turn_input = player_actions_as_single_value % 3
            thrust_input = player_actions_as_single_value // 3
            next_observation, reward, is_done, _ = self.env.step((turn_input, thrust_input))
            self.agent.remember(self.last_seen_observation, player_actions_as_single_value, reward, next_observation, is_done)
            self.last_seen_observation = next_observation

            if is_done:
                self.last_seen_observation = self.env.reset()
                self.games_remaining = self.games_remaining - 1
                if self.games_remaining <= 0:
                    return update_result.UpdateResult.GAME_COMPLETED

            return update_result.UpdateResult.CONTINUE_GAME

        def get_world():
            return self.env.world

        renderer_instance = renderer.Renderer(update_game, get_world)
        
        renderer_instance.run()


if __name__ == '__main__':
    trainer = OnlineTraining()
    trainer.main()