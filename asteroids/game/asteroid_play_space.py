import Box2D

PLAYSPACE_PADDING = 100

def add_asteroid_play_space(world : Box2D.b2World, left_border : float, right_border : float, bottom_border : float, top_border : float):
    fixture_shape = Box2D.b2PolygonShape()
    width = right_border - left_border + PLAYSPACE_PADDING
    height = top_border - bottom_border + PLAYSPACE_PADDING
    fixture_shape.SetAsBox(width, height, Box2D.b2Vec2(left_border + width / 2, bottom_border + height / 2), 0)

    fixture_def = Box2D.b2FixtureDef()
    fixture_def.shape = fixture_shape
    fixture_def.isSensor = True

    play_space_body_def = Box2D.b2BodyDef()
    play_space_body_def.fixtures = [fixture_def]
    play_space_body_def.type = Box2D.b2_staticBody
    play_space_body_def.position = (0, 0)

    return world.CreateBody(play_space_body_def)

def report_objects_in_play_space(play_space_body : Box2D.b2Body):
    objects = []
    for contact in play_space_body.contacts:
        objects.append(contact.other)
    return objects