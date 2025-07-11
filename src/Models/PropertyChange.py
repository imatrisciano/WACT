from typing import Optional


class PropertyChange:
    def __init__(self,
                 property_path: Optional[list[str]] = None,
                 before_value: Optional = None,
                 after_value: Optional = None):

        self.property_path: Optional[list[str]] = property_path
        self.before_value: Optional = before_value
        self.after_value: Optional = after_value

    def property_path_str(self) -> str:
        return str.join('.', self.property_path) if self.property_path is not None else ""

    def __str__(self):
        return f"{self.property_path_str()}: '{self.before_value}' -> '{self.after_value}'"
