import unittest
from src.DataPreprocessing.ObjectEncoder import ObjectEncoder

class ObjectEncoderTests(unittest.TestCase):
    def test_encoded_and_decoded_object_remains_the_same(self):
        object_encoder = ObjectEncoder()
        example_object = {'position': {'x': 0.6930000185966492, 'y': 0.9462000131607056, 'z': -2.4839999675750732}, 'rotation': {'x': -0.0, 'y': 0.0, 'z': 0.0}, 'visible': False, 'isInteractable': False, 'receptacle': True, 'toggleable': False, 'isToggled': False, 'breakable': False, 'isBroken': False, 'canFillWithLiquid': False, 'isFilledWithLiquid': False, 'dirtyable': False, 'isDirty': False, 'canBeUsedUp': False, 'isUsedUp': False, 'cookable': False, 'isCooked': False, 'isHeatSource': False, 'isColdSource': False, 'sliceable': False, 'isSliced': False, 'openable': False, 'isOpen': False, 'openness': 0.0, 'pickupable': False, 'isPickedUp': False, 'moveable': False, 'mass': 0.0, 'distance': 3.873828649520874, 'isMoving': False, 'fillLiquid': 'None', 'temperature': 'RoomTemp', 'objectType': 'CounterTop', 'objectId': 'CounterTop|+00,69|+00,95|-02,48'}


        encoded_object = object_encoder.encode(example_object)
        decoded_object = object_encoder.decode(encoded_object)

        self.assertDictEqual(example_object, decoded_object, "Decoded object dictionary is different from its original state")

if __name__ == '__main__':
    unittest.main()
