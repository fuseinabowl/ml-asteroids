import pyglet

pyglet.resource.path = ['renderer/resources']
pyglet.resource.reindex()

background = pyglet.resource.image("purple.png")