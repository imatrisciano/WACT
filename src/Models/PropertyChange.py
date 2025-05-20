class PropertyChange:
    def __init__(self):
        self.property_path: list = None
        self.before_value = None
        self.after_value = None

    def __init__(self, property_path: list, before_value, after_value):
        self.property_path = property_path
        self.before_value = before_value
        self.after_value = after_value
