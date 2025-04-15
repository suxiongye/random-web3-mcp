import unittest
import time
import asyncio
from unittest.mock import patch

from utils import (
    generate_request_id,
    get_random_str,
    basic_random_generator,
    random_array_generator,
    weighted_random_selector,
    random_feature_allocator,
    distribution_generator,
    random_event_trigger,
    random_seed_generator,
    random_shuffler,
    coordinate_generator
)


class TestUtils(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Setup work before testing"""
        print("\nStarting tests...")

    def tearDown(self):
        """Cleanup work after testing"""
        print("Tests completed\n")

    # Test request ID generation
    def test_generate_request_id(self):
        print("\nTesting generate_request_id function")
        
        # Define test cases
        test_cases = [
            {
                "name": "8-digit random number",
                "input": "12345678",
                "min_length": 21,
                "max_length": 23,
                "prefix_length": 8
            },
            {
                "name": "Random number longer than 8 digits",
                "input": "1234567890",
                "min_length": 21,
                "max_length": 23,
                "prefix_length": 8
            },
            {
                "name": "7-digit number",
                "input": "1234567",
                "min_length": 20,
                "max_length": 22,
                "prefix_length": 7
            },
            {
                "name": "Empty string",
                "input": "",
                "min_length": 13,
                "max_length": 15,
                "prefix_length": 0
            },
            {
                "name": "Contains letters",
                "input": "abc12345",
                "min_length": 21,
                "max_length": 23,
                "prefix_length": 8
            },
            {
                "name": "Contains symbols",
                "input": "!@#$%^&*",
                "min_length": 21,
                "max_length": 23,
                "prefix_length": 8
            }
        ]
        
        # Execute test cases
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            print(f"Input: {case['input']}")
            
            # Record time before generating request ID
            start_time = int(time.time() * 1000)
            
            request_id = generate_request_id(case["input"])
            
            # Record time after generating request ID
            end_time = int(time.time() * 1000)
            
            print(f"Generated request ID: {request_id}")
            print(f"Generation time: {(end_time - start_time):.2f}ms")
            
            # Verify request ID format
            id_length = len(request_id)
            assert case["min_length"] <= id_length <= case["max_length"], \
                f"Request ID length should be between {case['min_length']} and {case['max_length']}, got {id_length}"
            
            # Verify prefix and hyphen
            prefix = request_id[:case["prefix_length"]]
            expected_prefix = case["input"][:8] if len(case["input"]) > 0 else ""
            assert prefix == expected_prefix[:case["prefix_length"]], \
                f"Request ID should start with the first {case['prefix_length']} characters of the input"
            assert request_id[case["prefix_length"]] == "-" if case["prefix_length"] > 0 else True, \
                f"Request ID should contain a hyphen after the prefix"
            
            # Verify timestamp part is between start_time and end_time
            if case["prefix_length"] > 0:
                timestamp_part = int(request_id[case["prefix_length"] + 1:])
            else:
                # 处理空字符串输入的情况，去掉负号
                timestamp_str = request_id[1:] if request_id.startswith('-') else request_id
                timestamp_part = int(timestamp_str)
            
            assert start_time <= timestamp_part <= end_time + 1000, \
                f"Timestamp {timestamp_part} should be between {start_time} and {end_time + 1000}"

    # Test getting random string
    async def test_get_random_str(self):
        print("\nTesting get_random_str function")
        
        # Test normal case
        random_str = await get_random_str()
        print(f"Generated random string: {random_str}")
        
        # Verify string format
        self.assertEqual(len(random_str), 64, "Random string should be 64 characters long")
        self.assertTrue(all(c in "0123456789abcdef" for c in random_str), 
                       "Random string should only contain hexadecimal characters")

    # Test basic random number generation
    async def test_basic_random_generator(self):
        print("\nTesting basic_random_generator function")
        
        test_cases = [
            {
                "name": "Default range",
                "min_value": 0,
                "max_value": 1000000,
                "salt": ""
            },
            {
                "name": "Custom range",
                "min_value": 1,
                "max_value": 100,
                "salt": "test123"
            },
            {
                "name": "Single value range",
                "min_value": 42,
                "max_value": 42,
                "salt": "fixed"
            },
            {
                "name": "Negative range",
                "min_value": -100,
                "max_value": 100,
                "salt": "neg"
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            result = await basic_random_generator(
                min_value=case["min_value"],
                max_value=case["max_value"],
                salt=case["salt"]
            )
            
            print(f"Result: {result}")
            
            # Verify result format
            self.assertIn("requestId", result, "Result should contain requestId")
            self.assertIn("randomValue", result, "Result should contain randomValue")
            
            # Verify value range
            random_value = result["randomValue"]
            self.assertGreaterEqual(random_value, case["min_value"], 
                                  "Random value should not be less than minimum")
            self.assertLessEqual(random_value, case["max_value"], 
                               "Random value should not be greater than maximum")

    # Test random array generation
    async def test_random_array_generator(self):
        print("\nTesting random_array_generator function")
        
        test_cases = [
            {
                "name": "Default case",
                "array_length": 5,
                "min_value": 0,
                "max_value": 100,
                "allow_duplicates": True,
                "salt": ""
            },
            {
                "name": "No duplicates",
                "array_length": 10,
                "min_value": 1,
                "max_value": 20,
                "allow_duplicates": False,
                "salt": "unique"
            },
            {
                "name": "Single value range",
                "array_length": 3,
                "min_value": 42,
                "max_value": 42,
                "allow_duplicates": True,
                "salt": "fixed"
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            result = await random_array_generator(
                array_length=case["array_length"],
                min_value=case["min_value"],
                max_value=case["max_value"],
                allow_duplicates=case["allow_duplicates"],
                salt=case["salt"]
            )
            
            print(f"Result: {result}")
            
            # Verify result format
            self.assertIn("requestId", result, "Result should contain requestId")
            self.assertIn("randomArray", result, "Result should contain randomArray")
            
            # Verify array properties
            random_array = result["randomArray"]
            self.assertEqual(len(random_array), case["array_length"], 
                           "Array length should match specified length")
            
            # Verify value ranges
            for value in random_array:
                self.assertGreaterEqual(value, case["min_value"], 
                                      "Values should not be less than minimum")
                self.assertLessEqual(value, case["max_value"], 
                                   "Values should not be greater than maximum")
            
            # Verify uniqueness if required
            if not case["allow_duplicates"]:
                self.assertEqual(len(set(random_array)), len(random_array),
                               "Array should not contain duplicates")

    # Test weighted random selection
    async def test_weighted_random_selector(self):
        print("\nTesting weighted_random_selector function")
        
        options = ["Rare", "Epic", "Legendary", "Common"]
        weights = [200, 100, 50, 650]  # 20%, 10%, 5%, 65%
        
        # Test multiple selections
        selection_counts = {option: 0 for option in options}
        total_trials = 10
        
        for _ in range(total_trials):
            result = await weighted_random_selector(options, weights)
            selection = result["selectedOption"]
            selection_counts[selection] += 1
        
        print("\nSelection distribution after 10 trials:")
        for option, count in selection_counts.items():
            percentage = (count / total_trials) * 100
            print(f"{option}: {percentage:.1f}% ({count} times)")

    async def test_random_feature_allocator(self):
        print("\nTesting random_feature_allocator function")
        
        test_cases = [
            {
                "name": "Character stats",
                "feature_count": 3,
                "feature_max_values": [100, 100, 100],  # Strength, Agility, Intelligence
                "salt": "character"
            },
            {
                "name": "Item properties",
                "feature_count": 4,
                "feature_max_values": [1000, 50, 20, 5],  # Damage, Defense, Speed, Rarity
                "salt": "item"
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            result = await random_feature_allocator(
                feature_count=case["feature_count"],
                feature_max_values=case["feature_max_values"],
                salt=case["salt"]
            )
            
            print(f"Result: {result}")
            
            # Verify result format
            self.assertIn("requestId", result, "Result should contain requestId")
            self.assertIn("features", result, "Result should contain features array")
            self.assertIn("featureBitmap", result, "Result should contain feature bitmap")
            
            # Verify feature count and ranges
            features = result["features"]
            self.assertEqual(len(features), case["feature_count"],
                           "Feature count should match specified count")
            
            for i, value in enumerate(features):
                self.assertGreaterEqual(value, 0,
                                      f"Feature {i} should not be negative")
                self.assertLessEqual(value, case["feature_max_values"][i],
                                   f"Feature {i} should not exceed maximum value")

    # Test probability distribution generator
    async def test_distribution_generator(self):
        print("\nTesting distribution_generator function")
        
        test_cases = [
            {
                "name": "Uniform distribution",
                "distribution_type": 1,
                "distribution_parameters": [0, 100],
                "salt": "uniform"
            },
            {
                "name": "Normal distribution",
                "distribution_type": 2,
                "distribution_parameters": [50, 10],  # mean=50, std=10
                "salt": "normal"
            },
            {
                "name": "Exponential distribution",
                "distribution_type": 3,
                "distribution_parameters": [2.0],  # lambda=2.0
                "salt": "exp"
            },
            {
                "name": "Binomial distribution",
                "distribution_type": 4,
                "distribution_parameters": [10, 0.5],  # n=10, p=0.5
                "salt": "binomial"
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            result = await distribution_generator(
                distribution_type=case["distribution_type"],
                distribution_parameters=case["distribution_parameters"],
                salt=case["salt"]
            )
            
            print(f"Result: {result}")
            
            # Verify result format
            self.assertIn("requestId", result, "Result should contain requestId")
            self.assertIn("randomValue", result, "Result should contain random value")
            self.assertIn("distributionMetadata", result, "Result should contain distribution metadata")
            
            # Verify distribution metadata
            metadata = result["distributionMetadata"]
            self.assertEqual(metadata["distributionType"], case["distribution_type"],
                           "Distribution type should match specified type")
            
            # Verify value ranges based on distribution type
            random_value = result["randomValue"]
            if case["distribution_type"] == 1:  # Uniform
                self.assertGreaterEqual(random_value, case["distribution_parameters"][0],
                                      "Value should not be less than minimum")
                self.assertLessEqual(random_value, case["distribution_parameters"][1],
                                   "Value should not be greater than maximum")
            elif case["distribution_type"] == 4:  # Binomial
                n = case["distribution_parameters"][0]
                self.assertGreaterEqual(random_value, 0,
                                      "Binomial value should not be negative")
                self.assertLessEqual(random_value, n,
                                   f"Binomial value should not exceed {n}")

    # Test random event trigger
    async def test_random_event_trigger(self):
        print("\nTesting random_event_trigger function")
        
        test_cases = [
            {
                "name": "Combat events",
                "event_count": 3,
                "event_probabilities": [500, 300, 200],  # 50%, 30%, 20%
                "salt": "combat"
            },
            {
                "name": "Treasure drops",
                "event_count": 4,
                "event_probabilities": [800, 150, 40, 10],  # 80%, 15%, 4%, 1%
                "salt": "loot"
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            result = await random_event_trigger(
                event_count=case["event_count"],
                event_probabilities=case["event_probabilities"],
                salt=case["salt"]
            )
            
            print(f"Result: {result}")
            
            # Verify result format
            self.assertIn("requestId", result, "Result should contain requestId")
            self.assertIn("triggeredEvents", result, "Result should contain triggered events bitmap")
            self.assertIn("eventResults", result, "Result should contain event results")
            
            # Verify event count
            event_results = result["eventResults"]
            self.assertEqual(len(event_results), case["event_count"],
                           "Number of events should match specified count")
            
            # Verify event probabilities and results
            for i, event in enumerate(event_results):
                self.assertIn("eventId", event, "Event should have an ID")
                self.assertIn("probability", event, "Event should have a probability")
                self.assertIn("triggered", event, "Event should have triggered status")
                self.assertIn("randomValue", event, "Event should have a random value")
                
                self.assertEqual(event["probability"], case["event_probabilities"][i]/10,
                               f"Event {i} probability should match specified value")
                self.assertGreaterEqual(event["randomValue"], 0,
                                      "Random value should not be negative")
                self.assertLessEqual(event["randomValue"], 1000,
                                   "Random value should not exceed 1000")

    async def test_random_seed_generator(self):
        """Test random seed generator"""
        print("\nTesting random seed generator...")
        
        test_cases = [
            {
                "name": "Short seed",
                "seed_length": 16,
                "salt": ""
            },
            {
                "name": "Long seed",
                "seed_length": 64,
                "salt": "test"
            },
            {
                "name": "With salt",
                "seed_length": 32,
                "salt": "custom_salt"
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            result = await random_seed_generator(
                seed_length=case["seed_length"],
                salt=case["salt"]
            )
            
            print(f"Result: {result}")
            
            # Basic verification
            self.assertIsInstance(result, dict)
            self.assertIn("requestId", result)
            self.assertIn("randomSeed", result)
            self.assertIn("entropy", result)
            
            # Verify seed length
            self.assertEqual(len(bytes.fromhex(result["randomSeed"])), case["seed_length"])

    async def test_random_shuffler(self):
        """Test random shuffler"""
        print("\nTesting random shuffler...")
        
        test_cases = [
            {
                "name": "Number array",
                "input": [1, 2, 3, 4, 5],
                "salt": ""
            },
            {
                "name": "String array",
                "input": ["A", "B", "C", "D"],
                "salt": "test"
            },
            {
                "name": "Single element array",
                "input": [1],
                "salt": ""
            },
            {
                "name": "Empty array",
                "input": [],
                "salt": ""
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            result = await random_shuffler(
                input_array=case["input"],
                salt=case["salt"]
            )
            
            print(f"Result: {result}")
            
            # Basic verification
            self.assertIsInstance(result, dict)
            self.assertIn("requestId", result)
            self.assertIn("shuffledArray", result)
            
            # Verify length is the same
            self.assertEqual(len(result["shuffledArray"]), len(case["input"]))
            
            # Verify elements are the same (just in different order)
            self.assertEqual(sorted(result["shuffledArray"]), sorted(case["input"]))

    # Test random coordinate generator
    async def test_coordinate_generator(self):
        print("\nTesting coordinate_generator function")
        
        test_cases = [
            {
                "name": "2D coordinates",
                "dimensions": 2,
                "min_values": [-10.0, -10.0],
                "max_values": [10.0, 10.0],
                "coordinate_count": 5,
                "salt": "2d"
            },
            {
                "name": "3D coordinates",
                "dimensions": 3,
                "min_values": [0.0, 0.0, 0.0],
                "max_values": [100.0, 100.0, 100.0],
                "coordinate_count": 3,
                "salt": "3d"
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            result = await coordinate_generator(
                dimensions=case["dimensions"],
                min_values=case["min_values"],
                max_values=case["max_values"],
                coordinate_count=case["coordinate_count"],
                salt=case["salt"]
            )
            
            print(f"Result: {result}")
            
            # Verify result format
            self.assertIn("requestId", result, "Result should contain requestId")
            self.assertIn("coordinates", result, "Result should contain coordinates")
            
            # Verify coordinate count and dimensions
            coordinates = result["coordinates"]
            self.assertEqual(len(coordinates), case["coordinate_count"],
                           "Number of coordinates should match specified count")
            
            for coord in coordinates:
                self.assertEqual(len(coord), case["dimensions"],
                               "Coordinate dimensions should match specified dimensions")
                
                # Verify coordinate ranges
                for i, value in enumerate(coord):
                    self.assertGreaterEqual(value, case["min_values"][i],
                                          f"Coordinate {i} should not be less than minimum")
                    self.assertLessEqual(value, case["max_values"][i],
                                       f"Coordinate {i} should not exceed maximum")

def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

if __name__ == '__main__':
    print("Starting tests...")
    TestUtils.test_basic_random_generator = async_test(TestUtils.test_basic_random_generator)
    TestUtils.test_random_array_generator = async_test(TestUtils.test_random_array_generator)
    TestUtils.test_weighted_random_selector = async_test(TestUtils.test_weighted_random_selector)
    TestUtils.test_random_feature_allocator = async_test(TestUtils.test_random_feature_allocator)
    TestUtils.test_distribution_generator = async_test(TestUtils.test_distribution_generator)
    TestUtils.test_random_event_trigger = async_test(TestUtils.test_random_event_trigger)
    TestUtils.test_random_seed_generator = async_test(TestUtils.test_random_seed_generator)
    TestUtils.test_random_shuffler = async_test(TestUtils.test_random_shuffler)
    TestUtils.test_coordinate_generator = async_test(TestUtils.test_coordinate_generator)
    
    unittest.main() 