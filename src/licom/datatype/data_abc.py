from mesh.g2l import Global2Local

class DataABC():
    def __init__(self) -> None:
        Global2Local.allocate(self, self._field)
