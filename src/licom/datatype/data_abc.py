
from backend.calculation.field import Field

class DataABC():
    def __init__(self) -> None:
        Field.allocate(self, self._field)
