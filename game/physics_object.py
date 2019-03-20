class PhysicsObject():
    def __init__(self, 
            x:float=0,y:float=0,
            x_velocity:float=0,y_velocity:float=0,
            rotation:float=0,
            rotational_velocity:float=0
    ):
        self.position = [x,y]
        self.velocity = [x_velocity,y_velocity]
        self.rotation = rotation
        self.rotational_velocity =rotational_velocity

    def update(self):
        self.position = [position + velocity for position, velocity in zip(self.position, self.velocity)]
