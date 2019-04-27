from . import single_frame_actions

def convert_gym_actions_to_world_actions(actions):
    assert(len(actions) == 2)
    assert(actions[0] in range(3))
    assert(actions[1] in range(2))
    
    return single_frame_actions.SingleFrameActions(
        bullets_firing = 0,
        turn_speed = actions[0] - 1,
        thrust = actions[1]
    )