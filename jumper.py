import pyglet

from objects import player, world

window = pyglet.window.Window()
image = pyglet.resource.image('a-pecks-thin.png')

instance_world = world.World()
players = [player.Player()]

@window.event
def on_update():
    for player in players:
        player_perceived_world_state = player.gather_player_perceived_world_state(instance_world)
        player.calculate_actions(player_perceived_world_state)
    instance_world.update()

@window.event
def on_draw():
    window.clear()
    image.blit(0,0)

pyglet.app.run()