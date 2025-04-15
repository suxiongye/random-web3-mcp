"""
Multi-chain Random Number Generation MCP Service

This service provides verifiable random number generation functionality based on block hash,
supporting various random number generation scenarios:
1. Basic Random Numbers: Generate random numbers within a specified range
2. Random Arrays: Generate random arrays of specified length
3. Weighted Random: Random selection based on weights
4. Feature Assignment: Randomly generate feature value combinations
5. Probability Distribution: Support random numbers with various probability distributions
6. Event Trigger: Probability-based event trigger
7. Random Seed: Generate high-entropy random seed
8. Array Shuffle: Randomly shuffle array order
9. Coordinate Generation: Generate random coordinate points

All random number generation is based on the combination of block hash of the EVM chain,
ensuring higher randomness and verifiability.
"""

import os
import sys
from typing import List, Optional

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server.fastmcp import FastMCP
from utils import (
    basic_random_generator,
    random_array_generator,
    weighted_random_selector,
    random_feature_allocator,
    distribution_generator,
    random_event_trigger,
    random_seed_generator,
    random_shuffler,
    coordinate_generator,
    rarity_distributor
)

# Initialize FastMCP server
mcp = FastMCP("randomweb3mcp")


@mcp.tool()
async def generate_basic_random(salt: str = "", min_value: int = 0, max_value: int = 1000000) -> str:
    """Basic Random Number Generator
    
    Generate a random integer within the specified range
    
    Args:
        salt (str, optional): Random number salt value for increased randomness. Defaults to "".
        min_value (int, optional): Minimum value (inclusive). Defaults to 0.
        max_value (int, optional): Maximum value (inclusive). Defaults to 1000000.
    
    Returns:
        str: JSON string containing the random number result
    
    Application Scenarios:
    1. Lottery systems
    2. Game random numbers
    3. Random ID generation
    4. Test data generation
    """
    return await basic_random_generator(min_value=min_value, max_value=max_value, salt=salt)

@mcp.tool()
async def generate_random_array(salt: str = "", array_length: int = 1, min_value: int = 0, 
                              max_value: int = 1000000, allow_duplicates: bool = True) -> str:
    """Random Array Generator
    
    Generate a random array of specified length
    
    Args:
        salt (str, optional): Random number salt value. Defaults to "".
        array_length (int, optional): Array length. Defaults to 1.
        min_value (int, optional): Minimum value. Defaults to 0.
        max_value (int, optional): Maximum value. Defaults to 1000000.
        allow_duplicates (bool, optional): Allow duplicate values. Defaults to True.
    
    Returns:
        str: JSON string containing the random array
    
    Application Scenarios:
    1. Batch random number generation
    2. Random sampling
    3. Test dataset generation
    4. Random task assignment
    """
    return await random_array_generator(salt, array_length, min_value, max_value, allow_duplicates)

@mcp.tool()
async def generate_random_weighted(options: List[str], weights: List[int], salt: str = "") -> str:
    """Weighted Random Selector
    
    Randomly select an option based on weights
    
    Args:
        options (List[str]): List of options
        weights (List[int]): Corresponding weight list (0-1000)
        salt (str, optional): Random number salt value. Defaults to "".
    
    Returns:
        str: JSON string containing the selection result
    
    Application Scenarios:
    1. Lottery systems (prizes with different probabilities)
    2. Random drops (weighted item drops)
    3. Task assignment (based on priority)
    4. A/B testing (experiment groups with different ratios)
    """
    return await weighted_random_selector(options, weights, salt)

@mcp.tool()
async def generate_random_feature(feature_count: int, feature_max_values: List[int], salt: str = "") -> str:
    """Random Feature Allocator

    Generate a set of random feature values for objects, each feature value within its specified range.
    Feature values are encoded into a bitmap, with each feature occupying 8 bits.

    Args:
        feature_count (int): Number of features to generate
        feature_max_values (List[int]): List of maximum values for each feature, length must equal feature_count
        salt (str, optional): Random number salt value for increased randomness. Defaults to "".

    Returns:
        str: JSON string containing feature values and bitmap, formatted as:
        {
            "requestId": "Generated request ID",
            "features": [List of feature values],
            "featureBitmap": Feature bitmap value
        }

    Application Scenarios:
    1. Game character attribute generation (strength, agility, intelligence, etc.)
    2. Equipment attribute randomization (attack, defense, speed, etc.)
    3. Biological trait simulation (genes, traits, etc.)
    4. Random scene generation (terrain, weather, environment, etc.)
    """
    return await random_feature_allocator(feature_count, feature_max_values, salt)

@mcp.tool()
async def generate_distribution(distribution_type: int, distribution_parameters: List[float], salt: str = "") -> str:
    """Probability Distribution Random Generator

    Generate random numbers according to specified probability distribution type and parameters.
    Supports various common probability distributions.

    Args:
        distribution_type (int): Distribution type:
            1 = Uniform distribution (parameters: [min_value, max_value])
            2 = Normal distribution (parameters: [mean, standard_deviation])
            3 = Exponential distribution (parameters: [scale_parameter])
            4 = Binomial distribution (parameters: [trials, success_probability])
        distribution_parameters (List[float]): Distribution parameter list
        salt (str, optional): Random number salt value for increased randomness. Defaults to "".

    Returns:
        str: JSON string containing random value and distribution information, formatted as:
        {
            "requestId": "Generated request ID",
            "randomValue": Generated random value,
            "distributionMetadata": {
                "distributionType": Distribution type,
                ...Distribution parameters
            }
        }

    Application Scenarios:
    1. Financial market simulation (return distribution, risk analysis)
    2. Natural phenomena simulation (particle distribution, noise generation)
    3. Load testing (user behavior distribution)
    4. Statistical sampling (experimental data generation)
    """
    return await distribution_generator(distribution_type, distribution_parameters, salt)

@mcp.tool()
async def generate_random_event(event_count: int, event_probabilities: List[int], salt: str = "") -> str:
    """Random Event Trigger

    Trigger a series of events based on given probabilities, each event has an independent trigger probability.
    Uses bitmap to record trigger status for easy processing.

    Args:
        event_count (int): Total number of events
        event_probabilities (List[int]): Trigger probability for each event (0-1000, representing 0-100%)
        salt (str, optional): Random number salt value for increased randomness. Defaults to "".

    Returns:
        str: JSON string containing event trigger results, formatted as:
        {
            "requestId": "Generated request ID",
            "triggeredEvents": Event trigger bitmap,
            "eventResults": [
                {
                    "eventId": Event ID,
                    "probability": Trigger probability,
                    "triggered": Whether triggered,
                    "randomValue": Random value
                },
                ...
            ]
        }

    Application Scenarios:
    1. Game random events (trigger plot, drop items)
    2. Probability effect determination (skill trigger, combo determination)
    3. Risk event simulation (fault prediction, accident events)
    4. Multiple condition determination (combined probability events)
    """
    return await random_event_trigger(event_count, event_probabilities, salt)

@mcp.tool()
async def generate_random_seed(seed_length: int, salt: str = "") -> str:
    """Random Seed Generator

    Generate high-entropy random seed for encryption or other scenarios requiring high-quality random numbers.
    Uses blockchain hash as entropy source to ensure randomness.

    Args:
        seed_length (int): Length of seed to generate (in bytes)
        salt (str, optional): Random number salt value for increased randomness. Defaults to "".

    Returns:
        str: JSON string containing random seed, formatted as:
        {
            "requestId": "Generated request ID",
            "randomSeed": "Random seed in hexadecimal format",
            "entropy": Estimated entropy value
        }

    Application Scenarios:
    1. Key generation (encryption keys, signature seeds)
    2. Security tokens (session identifiers, authentication tokens)
    3. Random number initialization (PRNG seeds, simulation initial states)
    4. Unique identifier generation (UUID seeds, random identifiers)
    """
    return await random_seed_generator(seed_length, salt)

@mcp.tool()
async def shuffle_array(input_array: List, salt: str = "") -> str:
    """Random Array Shuffler

    Randomly shuffle the input array, ensuring each element has an equal probability of appearing in any position.
    Uses Fisher-Yates shuffle algorithm to ensure fairness.

    Args:
        input_array (List): Array to be shuffled, elements can be of any type
        salt (str, optional): Random number salt value for increased randomness. Defaults to "".

    Returns:
        str: JSON string containing the shuffled array, formatted as:
        {
            "requestId": "Generated request ID",
            "shuffledArray": [Shuffled array]
        }

    Application Scenarios:
    1. Game shuffling (playing cards, mahjong tiles)
    2. Random ordering (question order, playlist)
    3. Random grouping (team assignment, experiment grouping)
    4. Data shuffling (training dataset, test cases)
    """
    return await random_shuffler(input_array, salt)

@mcp.tool()
async def generate_coordinate(dimensions: int, min_values: List[float], max_values: List[float], 
                            coordinate_count: int, salt: str = "") -> str:
    """Random Coordinate Generator

    Generate random coordinate points in a specified dimensional space, each dimension has its own value range.
    Supports coordinate generation in any number of dimensions.

    Args:
        dimensions (int): Number of coordinate dimensions (1D, 2D, 3D, etc.)
        min_values (List[float]): List of minimum values for each dimension
        max_values (List[float]): List of maximum values for each dimension
        coordinate_count (int): Number of coordinate points to generate
        salt (str, optional): Random number salt value for increased randomness. Defaults to "".

    Returns:
        str: JSON string containing random coordinates, formatted as:
        {
            "requestId": "Generated request ID",
            "coordinates": [
                [x1, y1, z1, ...],  # First point coordinates
                [x2, y2, z2, ...],  # Second point coordinates
                ...
            ]
        }

    Application Scenarios:
    1. Game object positioning (NPC locations, item distribution)
    2. Particle systems (effect generation, particle distribution)
    3. Map generation (terrain height, resource distribution)
    4. Spatial sampling (3D modeling, spatial analysis)
    """
    return await coordinate_generator(dimensions, min_values, max_values, coordinate_count, salt)

@mcp.tool()
async def generate_rarity(item_count: int, rarity_tiers: List[str], rarity_percentages: List[float], guaranteed_minimums: Optional[List[int]] = None, salt: str="") -> str:
    """Rarity Distributor
    Args:
        item_count: Number of items
        rarity_tiers: Array of rarity tiers
        rarity_percentages: Probability percentage for each rarity tier
        guaranteed_minimums: Minimum guaranteed count for each rarity tier (optional)
        salt: Additional entropy source
    """
    return await rarity_distributor(item_count, rarity_tiers, rarity_percentages, guaranteed_minimums, salt)

if __name__ == "__main__":
    mcp.run()

