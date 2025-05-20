from src.Models.PropertyChange import PropertyChange


class ObjectChange:
    def __init__(self, object_id: str, changes: list[PropertyChange], object_disappeared: bool = False):
        self.object_id = object_id
        self.object_changes = changes
        self.object_disappeared = object_disappeared

    def has_changes(self) -> bool:
        return self.object_disappeared or any(self.object_changes)
