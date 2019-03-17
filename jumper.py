import pyglet

from objects import player, world
from game import background, resources

def load_background(background_batch):
    return background.load_background(background_batch, resources.background)

def main():
    game_window = pyglet.window.Window(800, 600)

    background_batch = pyglet.graphics.Batch()
    main_batch = pyglet.graphics.Batch()

    background_sprite = load_background(background_batch)

    instance_world = world.World()
    players = [player.Player()]

    def update(dt):
        for player in players:
            player_perceived_world_state = player.gather_player_perceived_world_state(instance_world)
            player.calculate_and_store_actions(player_perceived_world_state)
        instance_world.update()

    @game_window.event
    def on_draw():
        game_window.clear()
        background_batch.draw()
        # main_batch.draw()

    pyglet.clock.schedule_interval(update, 1 / 120.0)
    pyglet.app.run()

if __name__ == '__main__':
    main()
