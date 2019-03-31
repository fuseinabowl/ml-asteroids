from . import physics_object

class PlayerShip(physics_object.PhysicsObject):
    def __init__(self, *args, **kwargs):
        super(PlayerShip, self).__init__(*args, **kwargs)

        self.max_health = 3
        self.current_health = self.max_health