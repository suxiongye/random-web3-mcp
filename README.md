# RandomWeb3MCP - Web3 Random Element Generation Service

RandomWeb3MCP is a random element generation service based on EVM block hash. The service provides various random element generation tools that can be used in games, finance, testing, and other fields.

## Features

- **Verifiability**: All random numbers are generated based on blockchain hash, ensuring fairness and verifiability
- **Diversity**: Supports various random number generation scenarios, from basic random numbers to complex probability distributions
- **Reliability**: Uses blockchain as entropy source to ensure randomness quality
- **Usability**: Provides simple and intuitive API interfaces for easy integration

## Installation

```bash
git clone git@git.woa.com:zhixinlian/zxl-mcp-server.git
pip install -e .
```

## Quick Start

### Configuration in tico or Cursor

Add random-web3-mcp service configuration in Cursor settings:

```json
{
  "mcpServers": {
    "random-web3-mcp": {
      "command": "uv",
      "args": ["--directory", "local_repo_directory/zxl-mcp-server", "run", "main.py"]
    }
  }
}
```

## Tool List

### generate_basic_random
#### Name
Basic Random Number Generator
#### Function
Generate a random integer within the specified range
#### Parameters
- min_value (int, optional): Minimum value (inclusive). Defaults to 0.
- max_value (int, optional): Maximum value (inclusive). Defaults to 1000000.
- salt (str, optional): Random number salt value for increased randomness. Defaults to ''
#### Returns
JSON string containing the random number result

#### Application Scenarios
1. Lottery systems
2. Game random numbers
3. Random ID generation
4. Test data generation

### generate_random_array
#### Name
Random Array Generator
#### Function
Generate a random array of specified length
#### Parameters
- array_length (int, optional): Array length. Defaults to 1.
- min_value (int, optional): Minimum value. Defaults to 0.
- max_value (int, optional): Maximum value. Defaults to 1000000.
- salt (str, optional): Random number salt value. Defaults to ''
#### Returns
JSON string containing the random array

#### Application Scenarios
1. Batch random number generation
2. Random sampling
3. Test dataset generation
4. Random task assignment

### generate_random_weighted
#### Name
Weighted Random Selector
#### Function
Randomly select an option based on weights
#### Parameters
- options (List[str]): List of options
- weights (List[int]): Corresponding weight list (0-1000)
- salt (str, optional): Random number salt value. Defaults to ''
#### Returns
JSON string containing the selection result

#### Application Scenarios
1. Lottery systems (prizes with different probabilities)
2. Random drops (weighted item drops)
3. Task assignment (based on priority)
4. A/B testing (experiment groups with different ratios)

### generate_random_feature
#### Name
Random Feature Allocator
#### Function
Generate a set of random feature values for objects, each feature value within its specified range. Feature values are encoded into a bitmap, with each feature occupying 8 bits
#### Parameters
- feature_count (int): Number of features to generate
- feature_max_values (List[int]): List of maximum values for each feature, length must equal feature_count
- salt (str, optional): Random number salt value for increased randomness. Defaults to ''

#### Returns
JSON string containing feature values and bitmap, formatted as:
```json
{
    "requestId": "Generated request ID",
    "features": [List of feature values],
    "featureBitmap": Feature bitmap value
}
```

#### Application Scenarios
1. Game character attribute generation (strength, agility, intelligence, etc.)
2. Equipment attribute randomization (attack, defense, speed, etc.)
3. Biological trait simulation (genes, traits, etc.)
4. Random scene generation (terrain, weather, environment, etc.)

### generate_distribution
#### Name
Probability Distribution Random Generator
#### Function
Generate random numbers according to specified probability distribution type and parameters. Supports various common probability distributions.
#### Parameters
- distribution_type (int): Distribution type:
  - 1 = Uniform distribution (parameters: [min_value, max_value])
  - 2 = Normal distribution (parameters: [mean, standard_deviation])
  - 3 = Exponential distribution (parameters: [scale_parameter])
  - 4 = Binomial distribution (parameters: [trials, success_probability])
- distribution_parameters (List[float]): Distribution parameter list
- salt (str, optional): Random number salt value for increased randomness. Defaults to ''

#### Returns
JSON string containing random value and distribution information, formatted as:
```json
{
    "requestId": "Generated request ID",
    "randomValue": Generated random value,
    "distributionMetadata": {
        "distributionType": Distribution type,
        ...Distribution parameters
    }
}
```
#### Application Scenarios
1. Financial market simulation (return distribution, risk analysis)
2. Natural phenomena simulation (particle distribution, noise generation)
3. Load testing (user behavior distribution)
4. Statistical sampling (experimental data generation)

### generate_random_event
#### Name
Random Event Trigger
#### Function
Trigger a series of events based on given probabilities, each event has an independent trigger probability. Uses bitmap to record trigger status for easy processing.
#### Parameters
- event_count (int): Total number of events
- event_probabilities (List[int]): Trigger probability for each event (0-1000, representing 0-100%)
- salt (str, optional): Random number salt value for increased randomness. Defaults to ''

#### Returns
JSON string containing event trigger results, formatted as:
```json
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
```

#### Application Scenarios
1. Game random events (trigger plot, drop items)
2. Probability effect determination (skill trigger, combo determination)
3. Risk event simulation (fault prediction, accident events)
4. Multiple condition determination (combined probability events)

### generate_random_seed
#### Name
Random Seed Generator
#### Function
Generate high-entropy random seed for encryption or other scenarios requiring high-quality random numbers. Uses blockchain hash as entropy source to ensure randomness.
#### Parameters
- seed_length (int): Length of seed to generate (in bytes)
- salt (str, optional): Random number salt value for increased randomness. Defaults to ''

#### Returns
JSON string containing random seed, formatted as:
```json
{
    "requestId": "Generated request ID",
    "randomSeed": "Random seed in hexadecimal format",
    "entropy": Estimated entropy value
}
```

#### Application Scenarios
1. Key generation (encryption keys, signature seeds)
2. Security tokens (session identifiers, authentication tokens)
3. Random number initialization (PRNG seeds, simulation initial states)
4. Unique identifier generation (UUID seeds, random identifiers)

### shuffle_array
#### Name
Random Array Shuffler
#### Function
Randomly shuffle the input array, ensuring each element has an equal probability of appearing in any position. Uses Fisher-Yates shuffle algorithm to ensure fairness.
#### Parameters
- input_array (List): Array to be shuffled, elements can be of any type
- salt (str, optional): Random number salt value for increased randomness. Defaults to ''

#### Returns
JSON string containing the shuffled array, formatted as:
```json
{
    "requestId": "Generated request ID",
    "shuffledArray": [Shuffled array]
}
```

#### Application Scenarios
1. Game shuffling (playing cards, mahjong tiles)
2. Random ordering (question order, playlist)
3. Random grouping (team assignment, experiment grouping)
4. Data shuffling (training dataset, test cases)

### generate_coordinate
#### Name
Random Coordinate Generator
#### Function
Generate random coordinate points in a specified dimensional space, each dimension has its own value range. Supports coordinate generation in any number of dimensions.
#### Parameters
- dimensions (int): Number of coordinate dimensions (1D, 2D, 3D, etc.)
- min_values (List[float]): List of minimum values for each dimension
- max_values (List[float]): List of maximum values for each dimension
- coordinate_count (int): Number of coordinate points to generate
- salt (str, optional): Random number salt value for increased randomness. Defaults to ''

#### Returns
JSON string containing random coordinates, formatted as:
```json
{
    "requestId": "Generated request ID",
    "coordinates": [
        [x1, y1, z1, ...],  # First point coordinates
        [x2, y2, z2, ...],  # Second point coordinates
        ...
    ]
}
```

#### Application Scenarios
1. Game object positioning (NPC locations, item distribution)
2. Particle systems (effect generation, particle distribution)
3. Map generation (terrain height, resource distribution)
4. Spatial sampling (3D modeling, spatial analysis)

### generate_rarity
#### Name
Rarity Random Allocator
#### Function
Generate random coordinate points in a specified dimensional space, each dimension has its own value range. Supports coordinate generation in any number of dimensions.
#### Parameters
- item_count: Project quantity
- rarity_tiers: Rarity level array
- rarity_percentages: Probability percentage for each rarity level
- guaranteed_minimums: Guaranteed quantity for each rarity level (optional)
- salt (str, optional): Random number salt value for increased randomness. Defaults to ''

#### Returns
JSON string containing random rarity array, formatted as:
```json
{
    "requestId": "Generated request ID",
    "rarityDistribution": [Rarity allocation result]
}
```

#### Application Scenarios
1. Game item drops (different rarity equipment, items)
2. Lottery systems (prizes with different probabilities)
3. Resource allocation (different rarity resources, materials)
4. Random event trigger (different probability events)

## Application Scenarios

### Game Development
- Random item drops
- Character attribute generation
- Map random generation
- Probability event trigger

### Financial Application
- Risk simulation
- Investment portfolio analysis
- Market behavior simulation

### Test Data
- Random test case generation
- Load test data
- Performance test sample

### Scientific Calculation
- Monte Carlo simulation
- Particle system simulation
- Random sampling

## Notes

1. All random number generation depends on the blockchain hash of the Trust Chain, please ensure normal network connection
2. Weighted random selector weight value range is 0-1000, representing 0-100% probability
3. Probability distribution parameters need to provide correct parameter list according to specific distribution type
4. It is recommended to use salt parameter in production environment to increase randomness


## Error Handling

Service may return error types:

```python
{
    "error": "Error message",
    "code": "Error code",
    "requestId": "Request ID"
}
```

Common error codes:
- `INVALID_PARAMS`: Parameter error
- `NETWORK_ERROR`: Network connection error
- `CHAIN_ERROR`: Blockchain access error
- `INTERNAL_ERROR`: Internal service error

## Performance Consideration

- Each random number generation request needs to access the blockchain, which may have a certain delay
- It is recommended to cache frequently used random numbers
- Pay attention to request frequency when handling a large number of concurrent requests

## Contribution Guide

Welcome to submit Issue and Pull Request to help improve this project. Before submitting, please ensure:

1. Code conforms to PEP 8 specification
2. Appropriate test cases are added
3. Related documents are updated

## License

This project uses MIT License. See [LICENSE](LICENSE) file for details.
