import sys

class ObjectEncoder:
    def __init__(self):
        self.object_encoding_size = 37

        self.INDEX_TO_TEMPERATURE_MAP = {
            0: "Unknown",
            1: "Cold",
            2: "RoomTemp",
            3: "Hot"
        }
        self.TEMPERATURE_TO_INDEX_MAP = {v: k for k, v in self.INDEX_TO_TEMPERATURE_MAP.items()}

        self.INDEX_TO_MATERIALS_MAP = {
            0: "Unknown",
            1: "Food",
            2: "Paper",
            3: "Glass",
            4: "Ceramic",
            5: "Metal",
            6: "Plastic",
            7: "Sponge",
            8: "Wood",
            9: "Organic",
            10: "Stone",
            11: "Fabric",
            12: "Rubber",
            13: "Leather",
            14: "Wax",
            15: "Soap",
        }
        self.MATERIALS_TO_INDEX_MAP = {v: k for k, v in self.INDEX_TO_MATERIALS_MAP.items()}

        self.INDEX_TO_OBJECT_TYPES_MAP = {
            0: "Unknown",
            1: "Apple",
            2: "Book",
            3: "Bottle",
            4: "Bowl",
            5: "Bread",
            6: "ButterKnife",
            7: "Cabinet",
            8: "CoffeeMachine",
            9: "CounterTop",
            10: "CreditCard",
            11: "Cup",
            12: "DishSponge",
            13: "Drawer",
            14: "Egg",
            15: "Faucet",
            16: "Floor",
            17: "Fork",
            18: "Fridge",
            19: "GarbageCan",
            20: "HousePlant",
            21: "Kettle",
            22: "Knife",
            23: "Lettuce",
            24: "LightSwitch",
            25: "Microwave",
            26: "Mug",
            27: "Pan",
            28: "PaperTowelRoll",
            29: "PepperShaker",
            30: "Plate",
            31: "Pot",
            32: "Potato",
            33: "SaltShaker",
            34: "Shelf",
            35: "ShelvingUnit",
            36: "Sink",
            37: "SinkBasin",
            38: "SoapBottle",
            39: "Spatula",
            40: "Spoon",
            41: "Statue",
            42: "Stool",
            43: "StoveBurner",
            44: "StoveKnob",
            45: "Toaster",
            46: "Tomato",
            47: "Vase",
            48: "Window",
            49: "WineBottle",
            50: "PotatoSliced",
            51: "BreadSliced",
            52: "LettuceSliced",
            53: "CellPhone",
            54: "Chair",
            55: "Ladle",
            56: "EggCracked",
            57: "SideTable",
            58: "AppleSliced",
            59: "DiningTable",
            60: "Pen",
            61: "SprayBottle",
            62: "TomatoSliced",
            63: "Curtains",
            64: "Pencil",
            65: "Blinds",
            66: "GarbageBag",
            67: "Safe",
            68: "AluminumFoil",
            69: "Mirror",
            70: "ArmChair",
            71: "Box",
            72: "CoffeeTable",
            73: "DeskLamp",
            74: "FloorLamp",
            75: "KeyChain",
            76: "Laptop",
            77: "Newspaper",
            78: "Painting",
            79: "Pillow",
            80: "RemoteControl",
            81: "Sofa",
            82: "Television",
            83: "TissueBox",
            84: "TVStand",
            85: "Watch",
            86: "Boots",
            87: "Ottoman",
            88: "WateringCan",
            89: "Desk",
            90: "Dresser",
            91: "DogBed",
            92: "Candle",
            93: "RoomDecor",
            94: "AlarmClock",
            95: "BaseballBat",
            96: "BasketBall",
            97: "Bed",
            98: "CD",
            99: "TeddyBear",
            100: "TennisRacket",
            101: "Cloth",
            102: "Dumbbell",
            103: "Poster",
            104: "LaundryHamper",
            105: "TableTopDecor",
            106: "Desktop",
            107: "Footstool",
            108: "VacuumCleaner",
            109: "Bathtub",
            110: "BathtubBasin",
            111: "HandTowel",
            112: "HandTowelHolder",
            113: "Plunger",
            114: "ScrubBrush",
            115: "ShowerCurtain",
            116: "ShowerHead",
            117: "SoapBar",
            118: "Toilet",
            119: "ToiletPaper",
            120: "ToiletPaperHanger",
            121: "Towel",
            122: "TowelHolder",
            123: "ShowerDoor",
            124: "ShowerGlass"
        }
        self.OBJECT_TYPES_TO_INDEX_MAP = {v: k for k,v in self.INDEX_TO_OBJECT_TYPES_MAP.items()}

        self.INDEX_TO_FILL_LIQUID_MAP = {
            0: "None",
            1: "coffee",
            2: "wine",
            3: "water",
        }
        self.FILL_LIQUID_TO_INDEX_MAP = {v: k for k, v in self.INDEX_TO_FILL_LIQUID_MAP.items()}

        self._FEATURE_ORDER = [
            "position.x", "position.y", "position.z",
            "rotation.x", "rotation.y", "rotation.z",
            "visible", "isInteractable", "receptacle", "toggleable", "isToggled",
            "breakable", "isBroken", "canFillWithLiquid", "isFilledWithLiquid",
            "dirtyable", "isDirty", "canBeUsedUp", "isUsedUp",
            "cookable", "isCooked", "isHeatSource", "isColdSource",
            "sliceable", "isSliced", "openable", "isOpen", "openness",
            "pickupable", "isPickedUp", "moveable", "mass", "distance", "isMoving"
        ]
        self._FEATURES_THAT_ARE_BOOLEANS = [
            "visible", "isInteractable", "receptacle", "toggleable", "isToggled",
            "breakable", "isBroken", "canFillWithLiquid", "isFilledWithLiquid",
            "dirtyable", "isDirty", "canBeUsedUp", "isUsedUp",
            "cookable", "isCooked", "isHeatSource", "isColdSource",
            "sliceable", "isSliced", "openable", "isOpen",
            "pickupable", "isPickedUp", "moveable", "isMoving"
        ]


    @staticmethod
    def _get_value(obj, key_path):
        """Helper to navigate nested dicts via dotted path like 'position.x'."""
        keys = key_path.split('.')
        value = obj
        for k in keys:
            value = value.get(k, None)
            if value is None:
                break
        if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
            return float(value)
        else:
            print(f"Unknown data type for property {key_path}", file=sys.stderr)
            return 0.0  # Default for None or unexpected types

    @staticmethod
    def _get_one_hot_vector(item_to_index_map: dict, item: str):
        """
         returns a list with the same length as `item_to_index_map` which contains all 0 expect
         for the position `i` where `item == item_array[i]`
        """

        # Allocate list of zeroes
        one_hot_vector = [0] * len(item_to_index_map)

        # get the item's index or defaults to 0 if not found
        item_index = item_to_index_map.get(item, 0)
        one_hot_vector[item_index] = 1

        return one_hot_vector

    @staticmethod
    def _read_one_hot_vector(index_to_item_map: dict, one_hot_vector: list):
        """ Reads a one hot vector and returns the value of `index_to_item_map` corresponding to the
        index of the greatest value in the one hot vector"""

        # Get the greatest value
        max_one_hot_value = max(one_hot_vector)

        # If it's 0, return the first element in the map (Unknown), else get the index of the greatest value
        argmax = 0 if max_one_hot_value == 0 else one_hot_vector.index(max_one_hot_value)

        # Access the map and return the value corresponding to the index of the greatest value in the one hot vector
        return index_to_item_map.get(argmax)

    @staticmethod
    def format_float(value: float) -> str:
        """
        Formats a float into a string with sign, two digits, comma, two decimal places
        e.g.
         1.12837125475 -> '+01,13'
         -15.12903 -> '-15,13'
         1530.12837125475 -> '+1530,13'
        """
        return f"{'+' if value >= 0 else '-'}{abs(value):0>5.2f}".replace('.', ',')

    def _get_object_id(self, object_type: str, position_x: float, position_y: float, position_z: float) -> str:
        return f"{object_type}|{self.format_float(position_x)}|{self.format_float(position_y)}|{self.format_float(position_z)}"

    def _get_object_id_by_object(self, obj: dict) -> str:
        return self._get_object_id(
            obj["objectType"],
            obj["position"]["x"],
            obj["position"]["y"],
            obj["position"]["z"])


    def _set_dict_key(self, obj: dict, property_path: str, value):
        """
        Navigates the `obj` dictionary using the `property_path` information to set that value equal to `value`
        """

        # Splits the property_path into it's nodes, then navigates into the object
        parts = property_path.split('.')
        current = obj
        for p in parts[:-1]:
            current = current.setdefault(p, {}) # navigate into property. If it doesn't exist returns {}
        leaf_key = parts[-1]

        # Parse value
        value = bool(value) \
            if leaf_key in self._FEATURES_THAT_ARE_BOOLEANS \
            else float(value)

        current[leaf_key] = value

    def encode(self, obj: dict) -> list:
        # if obj is None, return all zeroes
        if obj is None:
            return [0] * self.object_encoding_size

        # Encode bools, ints and floats:
        features = [self._get_value(obj, key) for key in self._FEATURE_ORDER]

        # Add special features
        features.append(self.FILL_LIQUID_TO_INDEX_MAP.get(obj["fillLiquid"], 0))
        features.append(self.TEMPERATURE_TO_INDEX_MAP.get(obj["temperature"], 0))
        features.append(self.OBJECT_TYPES_TO_INDEX_MAP.get(obj["objectType"], 0))

        # We will ignore "salientMaterials" (too much data for too little information) as well as "name" (not relevant),
        # "receptacleObjectIds" (has more to do with the world status), "objectId" (can be obtained by object type + coordinates)

        #return torch.tensor(features, dtype=torch.float32).to(self.device)
        return features

    def decode(self, features: list) -> dict:
        output_object = {}

        for feature_path, feature_value in zip(self._FEATURE_ORDER, features):
            self._set_dict_key(output_object, feature_path, feature_value)

        # Parse special features
        output_object["fillLiquid"] = self.INDEX_TO_FILL_LIQUID_MAP.get(features[len(self._FEATURE_ORDER) + 0], None)
        output_object["temperature"] = self.INDEX_TO_TEMPERATURE_MAP.get(features[len(self._FEATURE_ORDER) + 1], None)
        output_object["objectType"] = self.INDEX_TO_OBJECT_TYPES_MAP.get(features[len(self._FEATURE_ORDER) + 2], None)

        # Reconstruct derived properties for convenience
        output_object["objectId"] = self._get_object_id_by_object(output_object)
        return output_object


"""
{
    "name": "CounterTop_2fe78146", 
    "position": {
     "x": 0.6930000185966492,
     "y": 0.9462000131607056,
     "z": -2.4839999675750732
    },
    "rotation": {
     "x": -0.0,
     "y": 0.0,
     "z": 0.0
    },
    "visible": false,
    "isInteractable": false,
    "receptacle": true,
    "toggleable": false,
    "isToggled": false,
    "breakable": false,
    "isBroken": false,
    "canFillWithLiquid": false,
    "isFilledWithLiquid": false,
    "fillLiquid": null,
    "dirtyable": false,
    "isDirty": false,
    "canBeUsedUp": false,
    "isUsedUp": false,
    "cookable": false,
    "isCooked": false,
    "temperature": "RoomTemp",
    "isHeatSource": false,
    "isColdSource": false,
    "sliceable": false,
    "isSliced": false,
    "openable": false,
    "isOpen": false,
    "openness": 0.0,
    "pickupable": false,
    "isPickedUp": false,
    "moveable": false,
    "mass": 0.0,
    "salientMaterials": null,
    "receptacleObjectIds": [
     "Pan|+00,72|+00,90|-02,42",
     "Kettle|+01,04|+00,90|-02,60",
     "Spatula|+00,38|+00,91|-02,33",
     "PepperShaker|+00,30|+00,90|-02,47",
     "SaltShaker|+00,35|+00,90|-02,57"
    ],
    "distance": 3.873828649520874,
    "objectType": "CounterTop",
    "objectId": "CounterTop|+00,69|+00,95|-02,48",
    "assetId": "",
    "parentReceptacles": null,
    "controlledObjects": null,
    "isMoving": false
}
"""