import numpy as np

_SENTINEL = np.array([1000.0, 1000.0])

def _ensure_array_2d_list(ps):
    arr = []
    for p in ps:
        if p is None:
            arr.append(_SENTINEL.copy())
        else:
            a = np.array(p, dtype=float)
            arr.append(a[:2])
    return arr

def calculate_euclidean_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.sqrt(np.sum((pos1 - pos2) ** 2))

def create_preference_lists(positions_a, positions_b):
    """Create preference lists based on Euclidean distances. Handles None by deprioritizing them."""
    a = _ensure_array_2d_list(positions_a)
    b = _ensure_array_2d_list(positions_b)

    preferences = {}
    num_positions = len(a)

    # Vectorized pairwise distances
    A = np.stack(a, axis=0)
    B = np.stack(b, axis=0)
    # expand dims to compute all pairwise distances
    diff = A[:, None, :] - B[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))

    for i in range(num_positions):
        order = np.argsort(dists[i])
        preferences[i] = order.tolist()

    return preferences

def stable_marriage(players_prefs, roles_prefs, num_players):
    """Implement Gale-Shapley algorithm for stable marriage problem."""
    # Initialize all players and roles as free
    free_players = list(range(num_players))
    matches = {i: None for i in range(num_players)}
    proposed = {i: [] for i in range(num_players)}
    
    while free_players:
        player = free_players[0]
        
        # Get player's preference list
        for role in players_prefs[player]:
            if role not in proposed[player]:
                proposed[player].append(role)
                current_match = matches[role]
                
                # If role is free, create match
                if current_match is None:
                    matches[role] = player
                    free_players.remove(player)
                    break
                
                # If role prefers new player to current match
                if roles_prefs[role].index(player) < roles_prefs[role].index(current_match):
                    matches[role] = player
                    free_players.remove(player)
                    free_players.append(current_match)
                    break
    
    return matches

def role_assignment(teammate_positions, formation_positions): 
    """
    Assign roles to players using the Stable Marriage algorithm.
    
    Args:
        teammate_positions: List of player positions (numpy arrays with [x,y])
        formation_positions: List of desired formation positions (numpy arrays with [x,y])
    
    Returns:
        Dictionary mapping player numbers (1-5) to assigned positions
    """
    # Create preference lists based on Euclidean distances
    players_prefs = create_preference_lists(teammate_positions, formation_positions)
    roles_prefs = create_preference_lists(formation_positions, teammate_positions)
    
    # Get stable matches using Gale-Shapley algorithm
    matches = stable_marriage(players_prefs, roles_prefs, len(teammate_positions))
    
    # Create final point_preferences dictionary with 1-based indexing
    point_preferences = {}
    for role, player in matches.items():
        # Convert to 1-based indexing for final output
        point_preferences[player + 1] = formation_positions[role]
    
    return point_preferences