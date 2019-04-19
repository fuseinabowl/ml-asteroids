
import Box2D

from .collision_filter_categories import CollisionFilterCategory

def add_borders(world : Box2D.b2World, left_border : float, right_border : float, bottom_border : float, top_border : float):
    fixture_defs = _create_fixtures(left_border, right_border, bottom_border, top_border)

    border_body_def = Box2D.b2BodyDef()
    border_body_def.fixtures = fixture_defs
    border_body_def.type = Box2D.b2_staticBody
    border_body_def.position = (0, 0)

    return world.CreateBody(border_body_def)
    
def _create_fixtures(left_border : float, right_border : float, bottom_border : float, top_border : float):
    # create in clockwise rotation so the edges face inwards
    vertices = (
        (left_border, bottom_border),
        (left_border, top_border),
        (right_border, top_border),
        (right_border, bottom_border)
    )
    fixture_defs = []
    for i, _ in enumerate(vertices):
        next_i = (i + 1) % 4

        side_border_shape = Box2D.b2EdgeShape()
        side_border_shape.vertex1 = vertices[i]
        side_border_shape.vertex2 = vertices[next_i]

        border_fixture_def = Box2D.b2FixtureDef()
        border_fixture_def.shape = side_border_shape
        border_fixture_def.friction = 0
        border_fixture_def.density = 0
        border_fixture_def.filter.categoryBits = CollisionFilterCategory.BORDER
        fixture_defs.append(border_fixture_def)
    
    return fixture_defs