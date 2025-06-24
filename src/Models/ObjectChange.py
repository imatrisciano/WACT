from src.Models.PropertyChange import PropertyChange


class ObjectChange:
    def __init__(self, object_id: str, changes: list[PropertyChange], object_disappeared: bool = False):
        self.object_id = object_id
        self.object_changes = changes
        self.object_disappeared = object_disappeared

    def has_changes(self) -> bool:
        return self.object_disappeared or any(self.object_changes)

    def number_of_changes(self) -> int:
        return len(self.object_changes)

    def __str__(self):
        if self.object_disappeared:
            return f"['{self.object_id}']: disappeared"
        else:
            return f"['{self.object_id}']: {len(self.object_changes)} changes"
