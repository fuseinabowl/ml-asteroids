import pyglet

from asteroids import env as environment
from asteroids.renderer import renderer
from asteroids.game import update_result

def main():
    env = environment.Env()
    last_seen_observation = env.reset()
    
    def update_game():
        player_actions = env.action_space.sample()
        last_seen_observation, _, is_done = env.step(player_actions)

        result = update_result.UpdateResult.GAME_COMPLETED if is_done else update_result.UpdateResult.CONTINUE_GAME
        return result

    def get_world():
        return env.world

    renderer_instance = renderer.Renderer(update_game, get_world)
    
    renderer_instance.run()


if __name__ == '__main__':
    main()