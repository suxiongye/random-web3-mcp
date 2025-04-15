import time
import httpx
from typing import Any, Dict, Optional, Union, List
import hashlib
import numpy as np
import asyncio
import aiohttp

# Constants
RPC_ENDPOINTS = {
    "eth": "https://rpc.mevblocker.io",
    "bsc": "https://bsc-dataseed2.bnbchain.org",
    "base": "https://mainnet.base.org"
}

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
MAX_RETRIES = 3  # Maximum retry attempts
RETRY_DELAY = 1  # Retry delay (seconds)
TIMEOUT = 10  # Request timeout (seconds)
PRIVATE_KEY = ""

# Headers for RPC calls
RPC_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
    "content-type": "application/json",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "user-agent": USER_AGENT
}

async def get_latest_block_hash(chain: str = "eth") -> str:
    """
    Get the latest block hash for the specified chain
    
    Args:
        chain: Chain name, supports "eth" or "bsc"
        
    Returns:
        str: Block hash, returns empty string if failed
    """
    if chain not in RPC_ENDPOINTS:
        print(f"Unsupported chain type: {chain}")
        return ""
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": ["latest", False],
            "id": 1
        }
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                RPC_ENDPOINTS[chain],
                headers=RPC_HEADERS,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            if "result" not in data or not data["result"]:
                print(f"Failed to get {chain} block data")
                return ""
                
            block_hash = data["result"].get("hash", "")
            if not block_hash:
                print(f"{chain} block hash is empty")
                return ""
                
            return block_hash
                
    except Exception as e:
        print(f"Failed to get {chain} latest block hash: {str(e)}")
        return ""


# Request API
async def request_api(url: str, **kwargs) -> Optional[dict]:
    """
    Send HTTP request and return response data
    Includes retry mechanism and timeout settings

    Args:
        url: Request URL
        **kwargs: Additional request parameters

    Returns:
        Optional[dict]: Response data, returns None if request fails
    """
    headers = {
        "User-Agent": USER_AGENT
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=TIMEOUT, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise aiohttp.ClientError(f"HTTP {response.status}")
                        
        except asyncio.TimeoutError:
            print(f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                print("Maximum retry attempts reached, request failed")
                return None
                
        except aiohttp.ClientError as e:
            print(f"HTTP error (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                print("Maximum retry attempts reached, request failed")
                return None
                
        except Exception as e:
            print(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                print("Maximum retry attempts reached, request failed")
                return None

# Generate request ID
def generate_request_id(random_num: str) -> str:
    """Generate unique request ID"""
    request_counter = int(time.time() * 1000)
    return f"{random_num[:8]}-{request_counter}"

# Derive a new seed from the main random number
def _derive_seed(random_num: str, salt: str = "") -> int:
    """
    Derive a new seed from the main random number
    Args:
        random_num: Main random number
        salt: Optional additional entropy source
    Returns:
        Derived integer seed (in range 0 to 2^32-1)
    """
    source = f"{random_num}{salt}"
    derived = hashlib.sha256(source.encode()).hexdigest()
    # Take first 8 characters (32 bits) as seed, ensure within valid range
    return int(derived[:8], 16) % (2**32)

# Estimate entropy of data
def _estimate_entropy(data: bytearray) -> float:
    """
    Simple entropy estimation
    Args:
        data: Input data as bytearray
    Returns:
        Estimated entropy value
    """
    counts = {}
    for byte in data:
        counts[byte] = counts.get(byte, 0) + 1
        
    entropy = 0
    for count in counts.values():
        probability = count / len(data)
        entropy -= probability * np.log2(probability)
    return entropy

# Get random string (use all chain block hashes)
async def get_random_str() -> str:
    """
    Get random string
    Uses combination of latest block hashes from all configured chains to generate random string
    Requires at least 2/3 of chains to return valid block hashes
    
    Returns:
        str: Random string
    """
    print("\nStarting to get random string...")
    try:
        # Get latest block hash from all chains
        hash_tasks = [get_latest_block_hash(chain) for chain in RPC_ENDPOINTS.keys()]
        block_hashes = await asyncio.gather(*hash_tasks)
        
        # Filter out empty hashes
        valid_hashes = [h for h in block_hashes if h]
        total_chains = len(RPC_ENDPOINTS)
        valid_count = len(valid_hashes)
        required_count = (total_chains * 2) // 3  # Round down, require at least 2/3
        
        print(f"Valid block hashes: {valid_count}/{total_chains} (need at least {required_count})")
        
        if valid_count < required_count:
            print(f"Insufficient valid block hashes: {valid_count} < {required_count}")
            return ""
            
        # Combine all hashes to generate new random string
        combined = "".join(valid_hashes)
        result = hashlib.sha256(combined.encode()).hexdigest()
        
        print(f"Generated random string: {result}")
        return result
    except Exception as e:
        print(f"Failed to get random string: {str(e)}")
        return ""

# Basic random number generator
async def basic_random_generator(min_value: int = 0, max_value: int = 1000000, salt: str="") -> dict:
    """Basic random number generator"""
    print(f"\nStarting to generate random number, parameters: salt={salt}, min={min_value}, max={max_value}")
    try:
        random_num = await get_random_str()
        if not random_num:
            print("Failed to get random number")
            return {"error": "Failed to get random number"}
            
        request_id = generate_request_id(random_num)
        print(f"Generated request ID: {request_id}")
        
        seed = _derive_seed(request_id, salt)
        print(f"Derived seed: {seed}")
        
        np.random.seed(seed)
        random_value = int(np.random.randint(min_value, max_value + 1))
        print(f"Generated random value: {random_value}")
            
        result = {
            "requestId": request_id,
            "randomValue": random_value
        }
        print(f"Final result: {result}")
        return result
    except Exception as e:
        print(f"Failed to generate random number: {str(e)}")
        return {"error": f"Failed to generate random number: {str(e)}"}

# Random array generator
async def random_array_generator(array_length: int = 1, min_value: int = 0, max_value: int = 1000000, allow_duplicates: bool = True, salt: str="") -> Dict:
    """Random array generator"""
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
        
    request_id = generate_request_id(random_num)
    seed = _derive_seed(request_id, salt)
    np.random.seed(seed)
    
    if allow_duplicates:
        random_array = np.random.randint(min_value, max_value + 1, size=array_length).tolist()
    else:
        # Ensure no duplicates
        if (max_value - min_value + 1) < array_length:
            raise ValueError("Range is too small to generate non-duplicate values")
        
        random_array = np.random.choice(
            range(min_value, max_value + 1), 
            size=array_length, 
            replace=False
        ).tolist()
    
    result = {
        "requestId": request_id,
        "randomArray": random_array
    }
    
    return result

# Weighted random selector
async def weighted_random_selector(options: List[str], weights: List[int], salt: str = "") -> Dict:
    """
    Weighted random selector
    
    Randomly select an option based on weights
    
    Args:
        options: List of options to choose from
        weights: List of weights for each option (0-1000)
        salt: Optional salt value for additional randomness
        
    Returns:
        Dict containing selected option and selection metadata
    """
    if len(options) != len(weights):
        raise ValueError("Options and weights must have the same length")
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
        
    request_id = generate_request_id(random_num)
    seed = _derive_seed(request_id, salt)
    
    np.random.seed(seed)
    # Normalize weights
    weights_array = np.array(weights, dtype=float)
    weights_normalized = weights_array / np.sum(weights_array)
    
    # Select based on weights
    selection_index = np.random.choice(len(options), p=weights_normalized)
    selected_option = options[selection_index]
        
    result = {
        "requestId": request_id,
        "selectedOption": selected_option,
        "selectionIndex": int(selection_index)
    }
        
    return result

# Random feature allocator
async def random_feature_allocator(feature_count: int, feature_max_values: List[int], salt: str="") -> Dict:
    """
    Random feature allocator
    
    Generate random feature values for each feature within specified ranges
    
    Args:
        feature_count: Number of features to generate
        feature_max_values: Maximum value for each feature
        salt: Optional salt value for additional randomness
        
    Returns:
        Dict containing feature values and bitmap
    """
    if len(feature_max_values) != feature_count:
        raise ValueError("Feature count must match length of max values array")
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
    request_id = generate_request_id(random_num)
    seed = _derive_seed(request_id, salt)
    np.random.seed(seed)
    
    features = []
    feature_bitmap = 0
    
    for i in range(feature_count):
        max_val = feature_max_values[i]
        feature_val = np.random.randint(0, max_val + 1)
        features.append(int(feature_val))
        
        # Encode feature into bitmap (8 bits per feature)
        feature_bitmap |= (feature_val << (i * 8))
    
    result = {
        "requestId": request_id,
        "features": features,
        "featureBitmap": feature_bitmap
    }
    
    return result

# Distribution random generator
async def distribution_generator(distribution_type: int, distribution_parameters: List[float], salt: str="") -> Dict:
    """
    Distribution random generator
    
    Generate random numbers following specified probability distribution
    
    Args:
        distribution_type: Type of distribution:
            1 = Uniform distribution
            2 = Normal distribution
            3 = Exponential distribution
            4 = Binomial distribution
        distribution_parameters: Parameters for the distribution
        salt: Optional salt value for additional randomness
        
    Returns:
        Dict containing random value and distribution metadata
    """
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
    request_id = generate_request_id(random_num)
    seed = _derive_seed(request_id, salt)
    np.random.seed(seed)
    
    metadata = {"distributionType": distribution_type}
    
    if distribution_type == 1:  # Uniform distribution
        min_val, max_val = distribution_parameters
        random_value = float(np.random.uniform(min_val, max_val))
        metadata.update({"min": min_val, "max": max_val})
        
    elif distribution_type == 2:  # Normal distribution
        mean, std_dev = distribution_parameters
        random_value = float(np.random.normal(mean, std_dev))
        metadata.update({"mean": mean, "stdDev": std_dev})
        
    elif distribution_type == 3:  # Exponential distribution
        scale = distribution_parameters[0]
        random_value = float(np.random.exponential(scale))
        metadata.update({"scale": scale})
        
    elif distribution_type == 4:  # Binomial distribution
        n, p = distribution_parameters
        random_value = int(np.random.binomial(n, p))
        metadata.update({"trials": n, "probability": p})
        
    else:
        raise ValueError("Unsupported distribution type")
    
    result = {
        "requestId": request_id,
        "randomValue": random_value,
        "distributionMetadata": metadata
    }
    
    return result

# Random event trigger
async def random_event_trigger(event_count: int, event_probabilities: List[int], salt: str="") -> Dict:
    """
    Random event trigger
    
    Trigger events based on their individual probabilities
    
    Args:
        event_count: Number of events
        event_probabilities: Probability for each event (0-1000)
        salt: Optional salt value for additional randomness
        
    Returns:
        Dict containing triggered events and their results
    """
    if len(event_probabilities) != event_count:
        raise ValueError("Event count must match probabilities array length")
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
    request_id = generate_request_id(random_num)
    seed = _derive_seed(request_id, salt)
    np.random.seed(seed)
    
    triggered_events = 0
    event_results = []
    
    for i in range(event_count):
        probability = event_probabilities[i]
        if not 0 <= probability <= 1000:
            raise ValueError("Probability must be between 0 and 1000")
        
        # Generate random number between 0-999
        random_value = np.random.randint(0, 1000)
        is_triggered = random_value < probability
        
        event_results.append({
            "eventId": i,
            "probability": probability / 10,  # Convert to percentage
            "triggered": is_triggered,
            "randomValue": int(random_value)
        })
        
        # Encode trigger result into bitmap
        if is_triggered:
            triggered_events |= (1 << i)
    
    result = {
        "requestId": request_id,
        "triggeredEvents": triggered_events,
        "eventResults": event_results
    }
    
    return result
    
# Random seed generator
async def random_seed_generator(seed_length: int, salt: str="") -> Dict:
    """
    Random seed generator
    
    Generate high-entropy random seed
    
    Args:
        seed_length: Length of seed in bytes
        salt: Optional salt value for additional randomness
        
    Returns:
        Dict containing random seed and entropy estimation
    """
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
    request_id = generate_request_id(random_num)
    combined_source = f"{random_num}{request_id}{salt}"
    
    # Create initial hash using SHA-256
    initial_hash = hashlib.sha256(combined_source.encode()).digest()
    
    # Extend seed to required length
    seed_bytes = bytearray()
    while len(seed_bytes) < seed_length:
        # Use counter as additional entropy
        counter = len(seed_bytes).to_bytes(4, byteorder='little')
        next_hash = hashlib.sha256(initial_hash + counter).digest()
        seed_bytes.extend(next_hash)
    
    # Truncate to required length
    seed_bytes = seed_bytes[:seed_length]
    
    # Calculate approximate entropy
    entropy = _estimate_entropy(seed_bytes)
    
    result = {
        "requestId": request_id,
        "randomSeed": seed_bytes.hex(),
        "entropy": entropy
    }
    
    return result

# Shuffle algorithm
async def random_shuffler(input_array: List, salt: str="") -> Dict:
    """
    Random array shuffler
    
    Randomly shuffle the input array
    
    Args:
        input_array: Array to shuffle
        salt: Optional salt value for additional randomness
        
    Returns:
        Dict containing shuffled array
    """
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
    request_id = generate_request_id(random_num)
    seed = _derive_seed(request_id, salt)
    np.random.seed(seed)
    
    # Copy array to avoid modifying original
    shuffled_array = input_array.copy()
    np.random.shuffle(shuffled_array)
    
    result = {
        "requestId": request_id,
        "shuffledArray": shuffled_array
    }
    
    return result

# Random coordinate generator
async def coordinate_generator(dimensions: int, min_values: List[float], max_values: List[float], coordinate_count: int, salt: str="") -> Dict:
    """
    Random coordinate generator
    
    Generate random coordinates in specified dimensional space
    
    Args:
        dimensions: Number of dimensions
        min_values: Minimum values for each dimension
        max_values: Maximum values for each dimension
        coordinate_count: Number of coordinates to generate
        salt: Optional salt value for additional randomness
        
    Returns:
        Dict containing generated coordinates
    """
    if len(min_values) != dimensions or len(max_values) != dimensions:
        raise ValueError("Dimension arrays must match specified dimensions")
    
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
    request_id = generate_request_id(random_num)
    seed = _derive_seed(request_id, salt)
    np.random.seed(seed)
    
    coordinates = []
    
    for _ in range(coordinate_count):
        point = []
        for dim in range(dimensions):
            point.append(float(np.random.uniform(min_values[dim], max_values[dim])))
        coordinates.append(point)
    
    result = {
        "requestId": request_id,
        "coordinates": coordinates
    }
    
    return result

# Rarity distributor
async def rarity_distributor(item_count: int, rarity_tiers: List[str], rarity_percentages: List[float], guaranteed_minimums: Optional[List[int]] = None, salt: str="") -> Dict:
    """
    Rarity distributor
    
    Distribute items across rarity tiers based on specified percentages
    
    Args:
        item_count: Total number of items
        rarity_tiers: List of rarity tier names
        rarity_percentages: Percentage for each rarity tier
        guaranteed_minimums: Minimum guaranteed items per tier
        salt: Optional salt value for additional randomness
        
    Returns:
        Dict containing rarity assignments and tier counts
    """
    if len(rarity_tiers) != len(rarity_percentages):
        raise ValueError("Tiers and percentages must have the same length")
    
    if guaranteed_minimums and len(guaranteed_minimums) != len(rarity_tiers):
        raise ValueError("Guaranteed minimums must match tiers length if provided")
    
    random_num = await get_random_str()
    if not random_num:
        return {"error": "Failed to get random number"}
    request_id = generate_request_id(random_num)
    seed = _derive_seed(request_id, salt)
    np.random.seed(seed)
    
    # Initialize guaranteed minimums
    tier_counts = {tier: 0 for tier in rarity_tiers}
    assignments = [""] * item_count
    remaining_items = item_count
    
    # Handle guaranteed minimums
    if guaranteed_minimums:
        for i, tier in enumerate(rarity_tiers):
            min_count = guaranteed_minimums[i]
            if min_count > 0:
                # Randomly assign guaranteed minimums
                indices = np.random.choice(
                    [i for i, a in enumerate(assignments) if a == ""],
                    size=min(min_count, remaining_items),
                    replace=False
                )
                
                for idx in indices:
                    assignments[idx] = tier
                    tier_counts[tier] += 1
                    remaining_items -= 1
    
    # Handle remaining items
    if remaining_items > 0:
        # Normalize percentages
        percentages = np.array(rarity_percentages, dtype=float)
        percentages = percentages / np.sum(percentages)
        
        # Randomly assign remaining items
        remaining_indices = [i for i, a in enumerate(assignments) if a == ""]
        remaining_assignments = np.random.choice(
            rarity_tiers,
            size=remaining_items,
            p=percentages
        )
        
        for i, idx in enumerate(remaining_indices):
            tier = remaining_assignments[i]
            assignments[idx] = tier
            tier_counts[tier] += 1
    
    result = {
        "requestId": request_id,
        "rarityAssignments": assignments,
        "tierCounts": tier_counts
    }
    
    return result


if __name__ == "__main__":
    result = asyncio.run(get_random_str())
    print(result)
    result = asyncio.run(basic_random_generator(min_value=1, max_value=100, salt="random123"))
    print(result)
    