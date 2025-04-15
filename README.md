# RandomWeb3MCP - Web3 Random Element Generation Service

RandomWeb3MCP is a random element generation service based on EVM block hash. The service provides various random element generation tools that can be used in games, finance, testing, and other fields.

## Features

- **Verifiability**: All random numbers are generated based on blockchain hash, ensuring fairness and verifiability
- **Diversity**: Supports various random number generation scenarios, from basic random numbers to complex probability distributions
- **Reliability**: Uses blockchain as entropy source to ensure randomness quality
- **Usability**: Provides simple and intuitive API interfaces for easy integration


## Quick Start

### Configure in Cursor

Add RandomWeb3MCP service configuration in Cursor settings:

```json
{
  "mcpServers": {
    "random-web3-mcp": {
      "command": "uv",
      "args": ["--directory", "random-web3-mcp", "run", "main.py"]
    }
  }
}
```

### Usage Example

After configuration, you can directly use RandomWeb3MCP's random number generation features in Cursor:

```python
# Generate basic random number
result = await mcp.randomweb3mcp.generate_basic_random(min_value=1, max_value=100)

# Generate weighted random selection
result = await mcp.randomweb3mcp.generate_random_weighted(
    options=["Rare", "Epic", "Legendary", "Common"],
    weights=[30, 15, 5, 50]
)
```

## Function List

### 1. Basic Random Number Generation
```python
result = await mcp.generate_basic_random(
    salt="",           # Optional: Random number salt value
    min_value=0,      # Optional: Minimum value
    max_value=1000000 # Optional: Maximum value
)
```

### 2. Random Array Generation
```python
result = await mcp.generate_random_array(
    salt="",           # Optional: Random number salt value
    array_length=1,    # Optional: Array length
    min_value=0,      # Optional: Minimum value
    max_value=1000000, # Optional: Maximum value
    allow_duplicates=True # Optional: Allow duplicate values
)
```

### 3. Weighted Random Selection
```python
result = await mcp.generate_random_weighted(
    options=["Option1", "Option2", "Option3"],  # List of options
    weights=[50, 30, 20],                # Weight list (sum should be 100)
    salt=""                              # Optional: Random number salt value
)
```

### 4. Random Feature Assignment
```python
result = await mcp.generate_random_feature(
    feature_count=3,                # Number of features
    feature_max_values=[10,20,30],  # List of maximum values for each feature
    salt=""                         # Optional: Random number salt value
)
```

### 5. Probability Distribution Random Number
```python
result = await mcp.generate_distribution(
    distribution_type=1,           # Distribution type (1=Uniform, 2=Normal, 3=Exponential, 4=Binomial)
    distribution_parameters=[0,1],  # Distribution parameter list
    salt=""                        # Optional: Random number salt value
)
```

### 6. Random Event Trigger
```python
result = await mcp.generate_random_event(
    event_count=3,                # Number of events
    event_probabilities=[500,300,200], # Trigger probabilities (0-1000)
    salt=""                       # Optional: Random number salt value
)
```

### 7. Random Seed Generation
```python
result = await mcp.generate_random_seed(
    seed_length=32,  # Seed length (bytes)
    salt=""         # Optional: Random number salt value
)
```

### 8. Array Random Shuffle
```python
result = await mcp.shuffle_array(
    input_array=[1,2,3,4,5],  # Array to be shuffled
    salt=""                   # Optional: Random number salt value
)
```

### 9. Random Coordinate Generation
```python
result = await mcp.generate_coordinate(
    dimensions=2,              # Coordinate dimensions
    min_values=[0,0],         # Minimum values for each dimension
    max_values=[100,100],     # Maximum values for each dimension
    coordinate_count=5,        # Number of coordinate points to generate
    salt=""                   # Optional: Random number salt value
)
```

## Application Scenarios

### Game Development
- Random item drops
- Character attribute generation
- Random map generation
- Probability event triggers

### Financial Applications
- Risk simulation
- Investment portfolio analysis
- Market behavior simulation

### Test Data
- Random test case generation
- Load test data
- Performance test samples

### Scientific Computing
- Monte Carlo simulation
- Particle system simulation
- Random sampling

## Notes

1. All random number generation depends on the block hash of the trust chain, please ensure network connection is normal
2. Weight values for weighted random selection range from 0-1000, representing 0-100% probability
3. Probability distribution parameters need to be provided correctly according to the specific distribution type
4. It is recommended to use the salt parameter in production environment to increase randomness

## Error Handling

Possible error types returned by the service:

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

## Performance Considerations

- Each random number generation request needs to access the blockchain, there may be some delay
- It is recommended to cache frequently used random numbers
- Pay attention to request rate control during high concurrent requests

## Contribution Guidelines

Issues and Pull Requests are welcome to help improve this project. Before submitting, please ensure:

1. Code complies with PEP 8 standards
2. Appropriate test cases have been added
3. Related documentation has been updated

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Examples by Scenarios

### Game Development Examples

```python
# 1. Random Item Drop System
result = await mcp.generate_random_weighted(
    options=["Legendary Sword", "Epic Armor", "Rare Potion", "Common Material"],
    weights=[5, 15, 30, 50],  # 0.5%, 1.5%, 3%, 5% drop rates
    salt="dungeon_boss_drop"
)

# 2. Character Attribute Generation
attributes = await mcp.generate_random_feature(
    feature_count=6,
    feature_max_values=[100, 100, 100, 100, 100, 100],  # STR, DEX, CON, INT, WIS, CHA
    salt="character_creation"
)

# 3. Random Map Generation
room_positions = await mcp.generate_coordinate(
    dimensions=2,
    min_values=[0, 0],
    max_values=[100, 100],
    coordinate_count=10,  # Generate 10 room positions
    salt="dungeon_layout"
)

# 4. Combat Critical Hit System
critical_hit = await mcp.generate_random_event(
    event_count=1,
    event_probabilities=[150],  # 15% critical hit chance
    salt="combat_roll"
)
```

### Financial Simulation Examples

```python
# 1. Stock Price Movement Simulation
price_changes = await mcp.generate_distribution(
    distribution_type=2,  # Normal distribution
    distribution_parameters=[0, 0.02],  # Mean=0%, StdDev=2%
    salt="stock_simulation"
)

# 2. Risk Assessment
risk_scenarios = await mcp.generate_random_array(
    array_length=1000,
    min_value=-100,
    max_value=100,
    salt="risk_analysis"
)

# 3. Portfolio Allocation
allocation = await mcp.generate_random_weighted(
    options=["Stocks", "Bonds", "Real Estate", "Crypto", "Cash"],
    weights=[400, 300, 150, 100, 50],  # 40%, 30%, 15%, 10%, 5%
    salt="portfolio_strategy"
)
```

### Testing Examples

```python
# 1. Load Test User Behavior
user_actions = await mcp.generate_random_weighted(
    options=["view", "click", "purchase", "share", "leave"],
    weights=[500, 250, 100, 100, 50],
    salt="user_behavior"
)

# 2. Performance Test Data Generation
test_data = await mcp.generate_random_array(
    array_length=100,
    min_value=1,
    max_value=1000000,
    allow_duplicates=False,
    salt="perf_test"
)

# 3. API Test Cases
api_parameters = await mcp.generate_random_feature(
    feature_count=4,
    feature_max_values=[100, 1000, 50, 10],  # page, size, status, type
    salt="api_test"
)
```

### Scientific Simulation Examples

```python
# 1. Particle System
particle_positions = await mcp.generate_coordinate(
    dimensions=3,
    min_values=[-10, -10, -10],
    max_values=[10, 10, 10],
    coordinate_count=100,
    salt="particle_system"
)

# 2. Genetic Algorithm
mutation_events = await mcp.generate_random_event(
    event_count=10,
    event_probabilities=[50, 50, 50, 50, 50, 50, 50, 50, 50, 50],  # 5% mutation chance each
    salt="genetic_mutation"
)

# 3. Monte Carlo Integration
sample_points = await mcp.generate_distribution(
    distribution_type=1,  # Uniform distribution
    distribution_parameters=[0, 1],  # Range [0,1]
    salt="monte_carlo"
)
```

### NFT and Blockchain Gaming Examples

```python
# 1. NFT Trait Generation
nft_traits = await mcp.generate_random_feature(
    feature_count=5,
    feature_max_values=[8, 12, 6, 15, 4],  # background, body, eyes, accessories, special
    salt="nft_mint"
)

# 2. Random Lootbox Contents
lootbox = await mcp.generate_random_weighted(
    options=["Mythical", "Legendary", "Epic", "Rare", "Common"],
    weights=[10, 40, 100, 250, 600],
    salt="lootbox_open"
)

# 3. Randomized Game Events
daily_events = await mcp.generate_random_event(
    event_count=5,
    event_probabilities=[200, 300, 400, 150, 100],  # Different event probabilities
    salt="daily_reset"
)
```
