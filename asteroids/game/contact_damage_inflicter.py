from Box2D import b2ContactListener, b2Contact
        
class ContactDamageInflicter(b2ContactListener):
    def __init__(self, world):
        b2ContactListener.__init__(self)
        self._world = world

    def BeginContact(self, contact):
        pass
    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact : b2Contact, impulse):
        asteroid = None
        if contact.fixtureA.body == self._world.player:
            asteroid = contact.fixtureB.body
        elif contact.fixtureB.body == self._world.player:
            asteroid = contact.fixtureA.body

        if asteroid is not None:
            normalImpulses = sum(impulse.normalImpulses)
            tangentImpulses = sum(impulse.tangentImpulses)
            self._world.player_impact(normalImpulses, tangentImpulses, asteroid)