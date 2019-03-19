import pyglet

from game import human_player, world
from renderer import renderer

def main():
    instance_world = world.World()

    def update_game():
        player_perceived_world_state = player.gather_player_perceived_world_state(instance_world)
        player_actions = player.get_this_frame_actions(player_perceived_world_state)
        instance_world.update(player_actions)

    def get_world():
        return instance_world

    renderer_instance = renderer.Renderer(update_game, get_world)
    
    player = human_player.HumanPlayer(renderer_instance.game_window)
    instance_world.add_player(player)

    renderer_instance.run()

if __name__ == '__main__':
    main()
