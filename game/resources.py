import pyglet

pyglet.resource.path = ['resources']
pyglet.resource.reindex()

background = pyglet.resource.image("purple.png")