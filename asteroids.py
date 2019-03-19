import pyglet

from game import player, world
from renderer import renderer

def main():
    instance_world = world.World()
    players = [player.Player()]

    def update_game():
        for player in players:
            player_perceived_world_state = player.gather_player_perceived_world_state(instance_world)
            player.calculate_and_store_actions(player_perceived_world_state)
        instance_world.update()

    def get_world():
        return instance_world

    renderer_instance = renderer.Renderer(update_game, get_world)

    renderer_instance.run()

if __name__ == '__main__':
    main()
