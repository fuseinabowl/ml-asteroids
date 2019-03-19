import pyglet

pyglet.resource.path = ['renderer/resources']
pyglet.resource.reindex()

background = pyglet.resource.image("purple.png")
player = pyglet.resource.image("player_ship.png")