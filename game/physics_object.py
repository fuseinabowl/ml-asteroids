import math

class PhysicsObject():
    def __init__(self, 
            x:float=0,y:float=0,
            x_velocity:float=0,y_velocity:float=0,
            rotation:float=0,
            rotational_velocity:float=0,
            friction:float=0.01,
            rotational_friction:float=0.01
    ):
        self.position = [x,y]
        self.velocity = [x_velocity,y_velocity]
        self.rotation = rotation
        self.rotational_velocity =rotational_velocity
        self.friction = friction
        self.rotational_friction = rotational_friction

    def update(self):
        self.position = [position + velocity for position, velocity in zip(self.position, self.velocity)]
        self.velocity = [velocity * (1 - self.friction) for velocity in self.velocity]

        new_rotation = self.rotation + self.rotational_velocity
        self.rotation = new_rotation % (2 * math.pi)

        self.rotational_velocity = self.rotational_velocity * self.rotational_friction
