

class World():
    def __init__(self):
        self._player_ship = (50,50)
        self._player_controller = None

    def update(self, player_actions):
        self._player_ship = (self._player_ship[0], self._player_ship[1] + 1)

    def add_player(self, player_controller):
        self._player_controller = player_controller

    @property
    def player(self):
        return self._player_ship