import pyglet

pyglet.resource.path = ['asteroids/renderer/resources']
pyglet.resource.reindex()

def center_image(image : pyglet.resource.image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width / 2
    image.anchor_y = image.height / 2

background = pyglet.resource.image("purple.png")

def load_player_ship_asset(file_name : str):
    image = pyglet.resource.image(file_name)
    player_scale = 0.5
    image.height = image.height * player_scale
    image.width = image.width * player_scale
    center_image(image)
    return image

player = load_player_ship_asset("player_ship.png")
player_damage = [load_player_ship_asset(damage_filename) for damage_filename in [
    "playerShip1_damage1.png",
    "playerShip1_damage2.png",
    "playerShip1_damage3.png"]]

asteroid = pyglet.resource.image("asteroid.png")
center_image(asteroid)