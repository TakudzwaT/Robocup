import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _team_attacking_direction(strategy) -> int:
    # +1 means we attack to +x, -1 means we attack to -x
    # In this environment, when our side is left, we attack to +x
    return 1 if strategy.side == 0 else -1


def _field_limits():
    # Half field: x in [-15, 15], y in [-10, 10]
    return (-15.0, 15.0, -10.0, 10.0)


def _respect_role_bands(x_targets, attack_dir):
    """
    Enforce role-based position constraints to prevent players from over-committing.
    
    Roles: [striker, secondary_attacker, support_mid, holding_mid, defender]
    - Striker: can go deep
    - Secondary attacker: limited to halfway through opponent half (7.5m from center)
    - Support mid: slightly more conservative (~8m from center)
    - Holding mid: should not cross halfway (x <= 0 for left team)
    - Defender: never cross halfway (x <= -0.5 for left team)
    """
    striker_x, sec_att_x, support_x, hold_x, def_x = x_targets
    
    # Half of opponent half depth from halfway line (7.5m)
    HALFWAY_DEPTH = 7.5
    
    # Simplify: our_half_max and opp_half_min were both 0.0 (halfway line)
    if attack_dir == 1:  # Attacking toward +x (left team)
        sec_att_x = min(sec_att_x, HALFWAY_DEPTH)
        support_x = min(support_x, HALFWAY_DEPTH + 0.5)
        hold_x = min(hold_x, 0.0)  # Don't cross halfway
        def_x = min(def_x, -0.5)  # Stay in our half
    else:  # Attacking toward -x (right team)
        sec_att_x = max(sec_att_x, -HALFWAY_DEPTH)
        support_x = max(support_x, -HALFWAY_DEPTH - 0.5)
        hold_x = max(hold_x, 0.0)  # Don't cross halfway
        def_x = max(def_x, 0.5)   # Stay in our half

    return [striker_x, sec_att_x, support_x, hold_x, def_x]


def _opponent_pressure(opp_positions: np.ndarray, point: np.ndarray, radius: float = 3.0) -> int:
    """Count opponents within radius of given point."""
    if opp_positions.shape[0] == 0:
        return 0
    distances = np.linalg.norm(opp_positions - point, axis=1)
    return int(np.sum(distances < radius))


def _clamp_xy(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    """Clamp coordinates to field bounds with margin."""
    return np.array([
        clamp(x, xmin + 0.5, xmax - 0.5),
        clamp(y, ymin + 0.5, ymax - 0.5),
    ])


def GenerateBasicFormation(strategy) -> list:
    """
    Generate context-aware formation points for 5 players, influenced by ball and opponents.

    Returns: list of 5 np.ndarray([x,y]) absolute targets corresponding to role slots.
    The stable marriage in Assignment will map players to these.
    Role slots (conceptual):
      0 - striker, 1 - secondary attacker (limited), 2 - support mid (limited), 
      3 - holding mid (never crosses halfway), 4 - defender (never crosses halfway)
    """
    # Constants for position offsets
    FORWARD_OFFSET = 3.5
    SUPPORT_OFFSET = 2.0
    BEHIND_OFFSET = -2.0
    DEEP_OFFSET = 6.0
    
    PRESSURE_RADIUS = 4.0
    PRESSURE_MULTIPLIER = 0.8
    MAX_PULLBACK = 3.0
    
    ball = np.array(strategy.ball_2d[:2])
    attack_dir = _team_attacking_direction(strategy)
    xmin, xmax, ymin, ymax = _field_limits()

    # Filter and prepare opponent positions
    opp_positions = [p for p in strategy.opponent_positions if p is not None]
    opp_np = np.array(opp_positions) if len(opp_positions) > 0 else np.zeros((0, 2))

    # Baseline lanes in y to spread players; center around ball.y but clamp
    lane_offsets = np.array([0.0, 2.5, -2.5, 5.0, -5.0])
    base_y = clamp(ball[1], ymin + 2.0, ymax - 2.0)
    lanes_y = np.clip(base_y + lane_offsets, ymin + 1.0, ymax - 1.0)

    # Adaptive: pull back if many opponents ahead of ball in corridor
    corridor_probe = ball + np.array([attack_dir * PRESSURE_RADIUS, 0.0])
    pressure = _opponent_pressure(opp_np, corridor_probe, radius=PRESSURE_RADIUS)
    pressure_pullback = clamp(pressure * PRESSURE_MULTIPLIER, 0.0, MAX_PULLBACK)

    # Compute raw x targets per conceptual role
    striker_x = ball[0] + attack_dir * (DEEP_OFFSET - pressure_pullback)
    sec_att_x = ball[0] + attack_dir * (FORWARD_OFFSET - 0.5 - pressure_pullback * 0.5)
    support_x = ball[0] + attack_dir * (SUPPORT_OFFSET - pressure_pullback * 0.5)
    hold_x = ball[0] + attack_dir * BEHIND_OFFSET
    def_x = hold_x - attack_dir * 2.5

    # Enforce role band restrictions
    striker_x, sec_att_x, support_x, hold_x, def_x = _respect_role_bands(
        [striker_x, sec_att_x, support_x, hold_x, def_x], attack_dir
    )

    # Distribute lanes to roles to create width and avoid clustering
    points = [
        _clamp_xy(striker_x, lanes_y[0], xmin, xmax, ymin, ymax),   # striker central-ish
        _clamp_xy(sec_att_x, lanes_y[1], xmin, xmax, ymin, ymax),  # secondary attacker slightly high
        _clamp_xy(support_x, lanes_y[2], xmin, xmax, ymin, ymax),  # support slightly low
        _clamp_xy(hold_x, lanes_y[3], xmin, xmax, ymin, ymax),     # holding mid wider high
        _clamp_xy(def_x, lanes_y[4], xmin, xmax, ymin, ymax),       # defender wider low
    ]

    # If ball is in our half, pull defender and holding mid closer to goal side for safety
    SAFETY_SHIFT = 1.5
    ball_in_our_half = (attack_dir == 1 and ball[0] < 0.0) or (attack_dir == -1 and ball[0] > 0.0)
    if ball_in_our_half:
        safety_shift = attack_dir * -SAFETY_SHIFT
        points[3][0] += safety_shift
        points[4][0] += safety_shift * 1.5

    # Possession-aware restraint: if opponents closer to ball than our closest teammate,
    # keep advanced roles from over-committing
    POSSESSION_MARGIN = 0.2
    try:
        has_possession = (strategy.min_teammate_ball_dist + POSSESSION_MARGIN < 
                         strategy.min_opponent_ball_dist)
    except (AttributeError, TypeError):
        has_possession = False

    if not has_possession:
        # Limit striker and secondary into or near the halfway at most
        if attack_dir == 1:  # left team attacks +x
            points[0][0] = clamp(points[0][0], -2.0, 6.0)
            points[1][0] = clamp(points[1][0], -4.0, 2.0)
            points[2][0] = clamp(points[2][0], -4.0, 1.0)
        else:  # right team attacks -x
            points[0][0] = clamp(points[0][0], -6.0, 2.0)
            points[1][0] = clamp(points[1][0], -2.0, 4.0)
            points[2][0] = clamp(points[2][0], -1.0, 4.0)

    return points
