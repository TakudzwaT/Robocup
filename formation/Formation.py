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
    # Role order: 0..4 produce five spots for 5 players (assignment will match players to these)
    # We want two attackers, one advanced support, one holding mid (should not cross halfway), one defender (never crosses halfway).
    # Constrain along attack x direction.
    our_half_max = 0.0 if attack_dir == 1 else 0.0
    opp_half_min = 0.0 if attack_dir == 1 else 0.0
    # Half of opponent half depth from halfway line (7.5m)
    half_opp_half_depth = 7.5

    # Sort of roles: [striker, secondary_attacker_limited, support_mid_limited, holding_mid_halfway, defender_own_half]
    # Apply constraints
    # striker: can go deep but keep within field
    # secondary attacker: not more than halfway through opponent half
    # support mid: at most just above halfway through opponent half (~8m from halfway)
    # holding mid: should not cross halfway
    # defender: never cross halfway (stay clearly in our half)

    striker_x, sec_att_x, support_x, hold_x, def_x = x_targets

    if attack_dir == 1:
        sec_att_x = min(sec_att_x, opp_half_min + half_opp_half_depth)
        support_x = min(support_x, opp_half_min + half_opp_half_depth + 0.5)
        hold_x = min(hold_x, 0.0)
        def_x = min(def_x, -0.5)
    else:
        sec_att_x = max(sec_att_x, opp_half_min - half_opp_half_depth)
        support_x = max(support_x, opp_half_min - half_opp_half_depth - 0.5)
        hold_x = max(hold_x, 0.0)
        def_x = max(def_x, 0.5)

    return [striker_x, sec_att_x, support_x, hold_x, def_x]


def GenerateBasicFormation(strategy) -> list:
    """
    Generate context-aware formation points for 5 players, influenced by ball and opponents.

    Returns: list of 5 np.ndarray([x,y]) absolute targets corresponding to role slots.
    The stable marriage in Assignment will map players to these.
    Role slots (conceptual):
      0 - striker, 1 - secondary attacker (limited), 2 - support mid (limited), 3 - holding mid (never crosses halfway), 4 - defender (never crosses halfway)
    """
    ball = np.array(strategy.ball_2d[:2])
    attack_dir = _team_attacking_direction(strategy)
    xmin, xmax, ymin, ymax = _field_limits()

    # Opponent pressure heuristic: count opponents within radius around our half and ball lane
    opp_positions = [p for p in strategy.opponent_positions if p is not None]
    opp_np = np.array(opp_positions) if len(opp_positions) > 0 else np.zeros((0, 2))

    # Compute opponent density near ball corridor
    def opponent_pressure(point: np.ndarray, radius: float = 3.0) -> int:
        if opp_np.shape[0] == 0:
            return 0
        d = np.linalg.norm(opp_np - point, axis=1)
        return int(np.sum(d < radius))

    # Baseline lanes in y to spread players; center around ball.y but clamp
    lane_offsets = np.array([0.0, 2.5, -2.5, 5.0, -5.0])
    base_y = clamp(ball[1], ymin + 2.0, ymax - 2.0)
    lanes_y = np.clip(base_y + lane_offsets, ymin + 1.0, ymax - 1.0)

    # Place x positions relative to ball and attack direction
    # Target depths in front/behind ball, scaled by attack direction
    forward = 3.5
    support_forward = 2.0
    behind = -2.0
    deep = 6.0

    # Adaptive: pull back if many opponents ahead of ball in corridor
    corridor_probe = ball + np.array([attack_dir * 4.0, 0.0])
    pressure = opponent_pressure(corridor_probe, radius=4.0)
    pressure_pullback = clamp(pressure * 0.8, 0.0, 3.0)

    # Compute raw x targets per conceptual role
    striker_x = ball[0] + attack_dir * (deep - pressure_pullback)
    sec_att_x = ball[0] + attack_dir * (forward - 0.5 - pressure_pullback * 0.5)
    support_x = ball[0] + attack_dir * (support_forward - pressure_pullback * 0.5)
    hold_x = ball[0] + attack_dir * behind
    def_x = hold_x - attack_dir * 2.5

    # Enforce role band restrictions
    striker_x, sec_att_x, support_x, hold_x, def_x = _respect_role_bands(
        [striker_x, sec_att_x, support_x, hold_x, def_x], attack_dir
    )

    # Clamp to field bounds with a margin
    def clamp_xy(x, y):
        return np.array([
            clamp(x, xmin + 0.5, xmax - 0.5),
            clamp(y, ymin + 0.5, ymax - 0.5),
        ])

    # Distribute lanes to roles to create width and avoid clustering
    points = [
        clamp_xy(striker_x, lanes_y[0]),   # striker central-ish
        clamp_xy(sec_att_x, lanes_y[1]),   # secondary attacker slightly high
        clamp_xy(support_x, lanes_y[2]),   # support slightly low
        clamp_xy(hold_x, lanes_y[3]),      # holding mid wider high
        clamp_xy(def_x, lanes_y[4]),       # defender wider low
    ]

    # If ball is in our half, pull defender and holding mid closer to goal side for safety
    if (attack_dir == 1 and ball[0] < 0.0) or (attack_dir == -1 and ball[0] > 0.0):
        safety_shift = attack_dir * -1.5
        points[3][0] += safety_shift
        points[4][0] += safety_shift * 1.5

    # Possession-aware restraint: if opponents closer to ball than our closest teammate,
    # keep advanced roles from over-committing
    try:
        has_possession = strategy.min_teammate_ball_dist + 0.2 < strategy.min_opponent_ball_dist
    except Exception:
        has_possession = False

    if not has_possession:
        # Limit striker and secondary into or near the halfway at most
        if attack_dir == 1:
            # left team attacks +x: cap striker around +4..+6, secondary/support at <= 0..+2
            points[0][0] = clamp(points[0][0], -2.0, 6.0)
            points[1][0] = clamp(points[1][0], -4.0, 2.0)
            points[2][0] = clamp(points[2][0], -4.0, 1.0)
        else:
            # right team attacks -x
            points[0][0] = clamp(points[0][0], -6.0, 2.0)
            points[1][0] = clamp(points[1][0], -2.0, 4.0)
            points[2][0] = clamp(points[2][0], -1.0, 4.0)

    return points

