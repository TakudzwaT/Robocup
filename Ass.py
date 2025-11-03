# Agent.py
from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np
from typing import Optional

from strategy.Assignment import role_assignment
from strategy.Strategy import Strategy
from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    """
    Improved Agent with a lightweight FSM, predictive pass/shoot heuristics,
    and formation/role-awareness. Designed to be fast per-tick.
    """

    # FSM states (small ints to keep checks cheap)
    S_POSITION = 0
    S_APPROACH = 1
    S_CHALLENGE = 2
    S_KICK = 3
    S_RECOVER = 4

    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:

        # robot type table (kept from your original)
        robot_type = (0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4)[unum - 1]

        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name,
                         enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = Agent.S_POSITION
        self.kick_direction = 0.0
        self.kick_distance = 0.0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)
                # ---------- possession / hysteresis state ----------
        self.possession = False             # sticky possession flag
        self.possession_since = 0           # world.time_local_ms when possession was last set
        # hysteresis thresholds (tune if needed)
        self.possession_gain_in = 0.9       # m -> claim possession when closer than this
        self.possession_gain_out = 1.3      # m -> lose possession only when farther than this


        # initial formation slot (kept simple)
        init_list = [
            np.array([-14.0, 0.0]), np.array([-9.0, -4.0]), np.array([-6.0, 0.0]),
            np.array([-9.0, 4.0]), np.array([-3.0, 0.0])
        ]
        # If team has more players, use cyclic fallback
        self.init_pos = init_list[(unum - 1) % len(init_list)].copy()

        # timers / simple counters stored in ms (use world.time_local_ms when needed)
        self._last_action_time = 0

        # safety: limit how many opponents / teammates we examine each tick
        self._max_opponents_check = 3
        self._max_teammates_check = 3

    # ---------- BEAM / MOVE / FAT-PROXY ----------
    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = np.array(self.init_pos, dtype=float)
        self.state = Agent.S_POSITION

        if avoid_center_circle and np.linalg.norm(pos) < 2.5:
            pos[0] = -2.3

        # commit beam when far from initial
        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos.tolist(), M.vector_angle((-pos[0], -pos[1])))
        else:
            if self.fat_proxy_cmd is None:
                # relaxed idle posture
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:
                # minimal fat-proxy heartbeat
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)

    def move(self, target_2d=(0.0, 0.0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=None, is_aggressive=False, timeout=3000):
        """
        Move wrapper that handles fat-proxy and path manager integration.
        Keeps the arguments lightweight and always passes numpy arrays where needed.
        """
        r = self.world.robot
        target = np.array(target_2d, dtype=float)

        if self.fat_proxy_cmd is not None:
            # fat proxy always uses a simple move subroutine
            self.fat_proxy_move(target, orientation, is_orientation_absolute)
            return

        if avoid_obstacles:
            try:
                target, _, distance_to_final_target = self.path_manager.get_path_to_target(
                    target, priority_unums=(priority_unums or []), is_aggressive=is_aggressive, timeout=timeout)
                target = np.array(target, dtype=float)
            except Exception:
                # fallback in case path_manager errors
                distance_to_final_target = np.linalg.norm(target - r.loc_head_position[:2])
        else:
            distance_to_final_target = np.linalg.norm(target - r.loc_head_position[:2])

        # call walk with conservative parameters (fast)
        self.behavior.execute("Walk", target.tolist(), True, orientation, is_orientation_absolute, float(distance_to_final_target))

    def kick(self, kick_direction: Optional[float] = None, kick_distance: Optional[float] = None, abort=False, enable_pass_command=False):
        """
        Prefer Basic_Kick when we want a definite kick, otherwise 'Dribble' (safe fallback).
        We compute kick power from kick_distance (clamped).
        """
        if kick_direction is not None:
            self.kick_direction = float(kick_direction)
        if kick_distance is not None:
            self.kick_distance = float(kick_distance)

        # If opponents are very close, allow pass command upstream
        try:
            if enable_pass_command and hasattr(self, 'world') and hasattr(self.world, 'teammates'):
                # small heuristic: if any opponent within 1.2 m of ball -> pass command
                nearest_opp_dist = min(
                    (np.linalg.norm(np.array(o.state_abs_pos[:2]) - self.world.ball_abs_pos[:2]) for o in self.world.opponents if getattr(o, "state_abs_pos", None) is not None),
                    default=10.0
                )
                if nearest_opp_dist < 1.2:
                    self.scom.commit_pass_command()
        except Exception:
            pass

        # choose power mapping (0..10)
        power = np.clip((self.kick_distance / 6.0) * 10.0, 3.0, 10.0)
        if self.fat_proxy_cmd is None:
            # Basic_Kick takes direction in degrees; power often tuned inside server policy
            return self.behavior.execute("Basic_Kick", float(self.kick_direction), abort)
        else:
            return self.fat_proxy_kick()

    def kickTarget(self, strategyData, mypos_2d=(0.0, 0.0), target_2d=(0.0, 0.0), abort=False, enable_pass_command=False):
        """
        Higher-level kick-to-target: sets direction and distance and delegates to kick().
        """
        mypos = np.array(mypos_2d, dtype=float)
        target = np.array(target_2d, dtype=float)
        vector = target - mypos
        dist = float(np.linalg.norm(vector))
        if dist < 1e-6:
            direction_deg = 0.0
        else:
            direction_deg = float(math.degrees(math.atan2(vector[1], vector[0])))

        # enable pass command if opponents near ball (cheap distance check)
        try:
            if enable_pass_command and strategyData.min_opponent_ball_dist < 1.3:
                self.scom.commit_pass_command()
        except Exception:
            pass

        self.kick_direction = direction_deg
        self.kick_distance = dist
        return self.kick(direction_deg, dist, abort=abort, enable_pass_command=enable_pass_command)

    # ---------- MAIN LOOP ----------
    def think_and_send(self):
        try:
            behavior = self.behavior
            strategyData = Strategy(self.world)

            # Early game mode handling
            if strategyData.play_mode == self.world.M_GAME_OVER:
                # do nothing special
                pass
            elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
                self.beam()
            elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
                self.beam(True)
            elif self.state == Agent.S_RECOVER or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
                # try to get up
                self.state = Agent.S_POSITION if behavior.execute("Get_Up") else Agent.S_RECOVER
            else:
                if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                    self.select_skill(strategyData)

            # broadcast + send
            self.radio.broadcast()
            if self.fat_proxy_cmd is None:
                try:
                    self.scom.commit_and_send(strategyData.robot_model.get_command())
                except Exception:
                    # fallback to raw robot command if needed
                    self.scom.commit_and_send(self.world.robot.get_command())
            else:
                self.scom.commit_and_send(self.fat_proxy_cmd.encode())
                self.fat_proxy_cmd = ""
        except Exception as e:
            # very defensive: print and try to send something to avoid disconnection
            print(f"[ERROR] think_and_send failed for player {getattr(self.world.robot,'unum', '?')}: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.scom.commit_and_send(self.world.robot.get_command())
            except Exception:
                pass

    def generate_safe_formation(self, strategyData):
        """
        Return a list of 5 formation target positions (numpy arrays).
        This function mirrors positions depending on team_side_is_left so that
        'forward' is always towards the opponent goal. Uses ball position to
        select offensive / midfield / defensive templates.
        """
        try:
            ball_x = float(strategyData.ball_2d[0])
            ball_y = float(strategyData.ball_2d[1])
        except Exception:
            ball_x, ball_y = 0.0, 0.0

        # Mirror factor: +1 if we play on the left (opponent at +x), -1 if we play on the right (opponent at -x)
        mirror = 1.0 if getattr(self.world, "team_side_is_left", True) else -1.0

        # Project ball relative to our attacking direction
        # If ball_x * mirror > 5 => ball in opponent half (we should be offensive)
        # If ball_x * mirror < -5 => ball in our own half (defensive)
        rel_ball = ball_x * mirror

        if rel_ball > 5.0:
            # Offensive shape: staggered lines in opponent half, limited tie to ball_y
            forward = 11.0 * mirror
            wing = 12.5 * mirror
            support = 6.0 * mirror
            by = float(np.clip(ball_y, -7.0, 7.0))
            formation = [
                np.array([forward, by * 0.20]),    # striker stays high, modest y tracking
                np.array([wing, by - 4.0]),        # left wing
                np.array([wing, by + 4.0]),        # right wing
                np.array([support, -3.5]),         # support left band
                np.array([support, 3.5])           # support right band
            ]
        elif rel_ball < -5.0:
            # Defensive shape: stay in our half, compact width
            back = -12.0 * mirror
            line = -8.0 * mirror
            formation = [
                np.array([back, 0.0]),
                np.array([line, -3.0]),
                np.array([line, 3.0]),
                np.array([line + 4.0 * mirror, -4.0]),
                np.array([line + 4.0 * mirror, 4.0])
            ]
        else:
            # Midfield: maintain spacing bands relative to field, not tightly to ball
            line_back = -5.0 * mirror
            line_mid = 0.0 * mirror
            line_front = 6.0 * mirror
            by = float(np.clip(ball_y, -6.0, 6.0))
            formation = [
                np.array([line_back, by * 0.15]),
                np.array([line_mid - 1.5 * mirror, -3.0]),
                np.array([line_mid - 1.5 * mirror, 3.0]),
                np.array([line_front, -4.0]),
                np.array([line_front, 4.0])
            ]

        # Ensure we always return exactly 5 numpy arrays (role_assignment expects size match)
        # If your team uses a different player count, adjust accordingly.
        # Ensure non-active slots are not too close to ball to avoid crowding
        safe = []
        min_ball_radius = 2.6
        b = np.array([ball_x, ball_y])
        for p in formation:
            vec = p - b
            d = np.linalg.norm(vec)
            if d < 1e-6:
                vec = np.array([1.0, 0.0])
                d = 1.0
            if d < min_ball_radius:
                p = b + vec / d * min_ball_radius
            safe.append(np.array([p[0], np.clip(p[1], -9.0, 9.0)]))
        return safe


    # ---------- DECISION / SKILL SELECTION ----------
    def find_best_action(self, strategyData):
        """
        Decide between shooting, passing, or dribbling based on geometry and opponent pressure.
        Returns (target_position, action_type)
        """
        try:
            mypos = np.array(strategyData.mypos, dtype=float)
            ball = np.array(strategyData.ball_2d, dtype=float)
            opponent_goal = np.array([15.0, 0.0])
            dist_ball_to_goal = np.linalg.norm(opponent_goal - ball)

            # Check if path to goal is clear (reuse the fast test)
            clear_shot = self._fast_clear_test(ball, opponent_goal, strategyData.opponent_positions)

            # Check opponent pressure
            min_opp_dist = strategyData.min_opponent_ball_dist
            under_pressure = min_opp_dist < 1.5

            # 1️⃣ Try SHOOT — if close and clear, or under pressure with decent shot
            if (dist_ball_to_goal < 5.0 and clear_shot) or (under_pressure and dist_ball_to_goal < 8.0 and clear_shot):
                return opponent_goal, "SHOOT"

            # 2️⃣ Try PASS — if a good teammate is open
            receiver_pos, receiver_score = self._fast_find_pass_target(strategyData)
            # Adjust pass threshold based on pressure
            pass_threshold = 0.35 if not under_pressure else 0.25
            if receiver_pos is not None and receiver_score > pass_threshold:
                return receiver_pos, "PASS"

            # 3️⃣ Otherwise DRIBBLE — more aggressive when under pressure
            dribble_dist = 1.5 if under_pressure else 2.5
            goal_dir = opponent_goal - ball
            if np.linalg.norm(goal_dir) < 1e-6:
                goal_dir = np.array([1.0, 0.0])
            goal_dir /= np.linalg.norm(goal_dir)
            dribble_target = ball + goal_dir * dribble_dist
            return dribble_target, "DRIBBLE"

        except Exception as e:
            print(f"[ERROR] find_best_action failed: {e}")
            import traceback
            traceback.print_exc()
            # Default fallback: dribble straight ahead
            return np.array([ball[0] + 1.0, ball[1]]), "DRIBBLE"


    def handle_ball(self, strategyData, drawer):
        """Active player behavior — intelligently approach, control, and act on the ball."""
        try:
            mypos = np.array(strategyData.mypos, dtype=float)
            ball = np.array(strategyData.ball_2d, dtype=float)
            dist_to_ball = np.linalg.norm(mypos - ball)

            # --- Stage 1: approach the ball until close enough ---
            if dist_to_ball > 0.6:
                print(f"[DEBUG] Approaching ball ({dist_to_ball:.2f})")
                # Move directly to ball position
                return self.move(ball, orientation=0)

            # --- Stage 2: close enough — take action ---
            # Find best next action (shoot, pass, dribble)
            target, action_type = self.find_best_action(strategyData)

            # Visualization for debugging
            drawer.line(strategyData.mypos, target, 2, drawer.Color.red, "action_line")
            drawer.annotation(
                (0, 9.5),
                f"Action: {action_type}",
                drawer.Color.green if action_type == "SHOOT" else drawer.Color.blue,
                "action_type"
            )

            # --- Stage 3: execute the chosen action ---
            if action_type == "SHOOT":
                print(f"[Player {self.world.robot.unum}] SHOOTING at {target}")
            elif action_type == "PASS":
                print(f"[Player {self.world.robot.unum}] PASSING to {target}")
            else:
                print(f"[Player {self.world.robot.unum}] DRIBBLING towards {target}")

            # Kick/Dribble execution (only when close)
            return self.kickTarget(strategyData, mypos, target)

        except Exception as e:
            print(f"[ERROR] handle_ball failed for player {self.world.robot.unum}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic dribble behavior to keep motion alive
            return self.behavior.execute("Dribble", None, None)



    def select_skill(self, strategyData):
        try:
            drawer = self.world.draw

            # IMPORTANT: Limit computation time to prevent falling
            # Keep logic simple and fast

            # Keeper handling: keep in our half and guard goal
            if int(self.world.robot.unum) == 1:
                return self._keeper_behavior(strategyData, drawer)

            # Generate dynamic formation based on ball position
            formation_positions = self.generate_safe_formation(strategyData)
            point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)

            # --- safely extract desired slot (works whether role_assignment returns dict or list) ---
            desired = None
            if isinstance(point_preferences, dict):
                desired = point_preferences.get(strategyData.player_unum, None)
            else:
                # assume list/tuple indexed by (unum-1)
                try:
                    desired = point_preferences[strategyData.player_unum - 1]
                except Exception:
                    desired = None

            if desired is None:
                desired = np.array(self.init_pos, dtype=float)

            strategyData.my_desired_position = np.array(desired, dtype=float)

            # Calculate orientation towards ball/target
            strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
                strategyData.my_desired_position)

            # Am I the active player (closest to ball)?
            print(f"[DEBUG] active={strategyData.active_player_unum}, me={strategyData.robot_model.unum}, dist_to_ball={getattr(strategyData, 'ball_dist', '?')}")

            i_am_active = strategyData.active_player_unum == strategyData.robot_model.unum

            # For support players, keep a safe radius from the ball to avoid crowding
            if not i_am_active:
                b = np.array(strategyData.ball_2d, dtype=float)
                dpos = np.array(strategyData.my_desired_position, dtype=float)
                vec = dpos - b
                dist = np.linalg.norm(vec)
                min_radius = 2.8
                if dist < 1e-6:
                    vec = np.array([1.0, 0.0])
                    dist = 1.0
                if dist < min_radius:
                    dpos = b + vec / dist * min_radius
                    strategyData.my_desired_position = dpos

            # Check if I'm in position (simplified check)
            in_position = self.am_i_in_position(strategyData.my_desired_position, threshold=0.6)

            # Decision making based on role and game state
            if i_am_active:
                drawer.annotation((0, 10.5), "ACTIVE - Ball Handler", drawer.Color.yellow, "status")
                return self.handle_ball(strategyData, drawer)
            else:
                drawer.annotation((0, 10.5), "SUPPORT - Positioning", drawer.Color.cyan, "status")
                return self.support_play(strategyData, drawer, in_position)
        except Exception as e:
            print(f"[ERROR] select_skill failed for player {self.world.robot.unum}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: just move to a safe position
            return self.move((0, 0), orientation=0)


    # ---------- SUB-BEHAVIORS ----------
    def _keeper_behavior(self, strategyData, drawer):
        """
        Simple keeper: stay near goal, clear long when ball is inside penalty area.
        Keeper is conservative (avoids risky dribble).
        """
        mypos = strategyData.mypos
        ball = strategyData.ball_2d
        side_left = bool(self.world.team_side_is_left)
        goal = np.array([-15.0, 0.0]) if side_left else np.array([15.0, 0.0])
        # keeper's default safe spot (just in front of our goal)
        safe_spot = goal + np.array([1.0 if goal[0] < 0 else -1.0, 0.0])

        # Enforce keeper stays in own half
        if side_left:
            # own half is x <= 0
            safe_spot[0] = min(safe_spot[0], -0.5)
        else:
            # own half is x >= 0
            safe_spot[0] = max(safe_spot[0], 0.5)

        dist_ball_goal = float(np.linalg.norm(ball - goal))
        print(f"[KEEPER] unum={self.world.robot.unum} side_left={side_left} mypos={mypos} ball={ball} dist_ball_goal={dist_ball_goal:.2f}")

        # if ball is in our penalty zone (close to goal), go and clear
        if dist_ball_goal < 5.5:
            drawer.annotation((0, 9.5), "KEEPER: CLEAR", drawer.Color.red, "keeper")
            print(f"[KEEPER] action=CLEAR target=[0.0, 6.0]")
            return self.kickTarget(strategyData, mypos, np.array([0.0, 6.0]), enable_pass_command=True)
        else:
            # hold position and face ball
            drawer.annotation((0, 9.5), "KEEPER: HOLD", drawer.Color.white, "keeper")
            print(f"[KEEPER] action=HOLD target={safe_spot}")
            return self.move(safe_spot, orientation=strategyData.ball_dir)

    def _active_pipeline(self, strategyData, drawer):
        """Active player decision: fast approach -> quick decision -> action"""
        mypos = strategyData.mypos
        ball = strategyData.ball_2d
        dist_to_ball = float(strategyData.ball_dist)
        print(f"[DEBUG] P{self.world.robot.unum} dist_to_ball={dist_to_ball:.2f} possession={self.possession}")


        # if we don't see the ball or it's far, move to desired position or approach ball
        if (not self.world.ball_is_visible) and dist_to_ball > 2.5:
            return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)

        # If close to ball -> decide action
        if dist_to_ball < 1.0:
            # compute cheap metrics
            goal = np.array([15.0, 0.0])
            dist_ball_to_goal = float(np.linalg.norm(goal - ball))
            angle_to_goal = abs(M.normalize_deg(M.target_abs_angle(ball, goal) - strategyData.my_ori))

            # clear shot heuristic (cheap: check up to 3 opponents)
            clear_shot = (dist_ball_to_goal < 4.2 and angle_to_goal < 30 and self._fast_clear_test(ball, goal, strategyData.opponent_positions))

            if clear_shot:
                drawer.annotation((0, 9.0), "SHOOT", drawer.Color.green, "action")
                return self.kickTarget(strategyData, mypos, goal, enable_pass_command=True)

            # try safe pass
            receiver_pos, receiver_score = self._fast_find_pass_target(strategyData)
            if receiver_pos is not None and receiver_score > 0.35:
                drawer.annotation((0, 9.0), "PASS", drawer.Color.blue, "action")
                return self.kickTarget(strategyData, mypos, receiver_pos, enable_pass_command=True)

            # else dribble toward goal direction (controlled)
                        # else dribble: use follow-then-push logic (stable)
            goal_dir_vec = goal - ball
            if np.linalg.norm(goal_dir_vec) < 1e-6:
                goal_dir_vec = np.array([1.0, 0.0])
            goal_dir_unit = goal_dir_vec / np.linalg.norm(goal_dir_vec)

            follow_distance = 0.7
            push_distance = 2.0
            hysteresis = 0.15

            follow_point = ball - goal_dir_unit * follow_distance
            push_point = ball + goal_dir_unit * push_distance
            dist_to_follow = np.linalg.norm(mypos - follow_point)

            if dist_to_follow > (follow_distance * 0.6 + hysteresis):
                drawer.annotation((0, 9.0), "GET BEHIND BALL", drawer.Color.orange, "action")
                return self.move(follow_point, orientation=strategyData.ball_dir, avoid_obstacles=False)
            if dist_to_follow < (follow_distance * 0.6 - hysteresis):
                drawer.annotation((0, 9.0), "HOLD BEHIND BALL", drawer.Color.orange, "action")
                return self.move(mypos, orientation=strategyData.ball_dir, avoid_obstacles=False)
            drawer.annotation((0, 9.0), "DRIBBLE FORWARD", drawer.Color.orange, "action")
            return self.move(push_point, orientation=strategyData.ball_dir, avoid_obstacles=False)


        else:
            # not close enough: approach ball from an angle (behind ball relative to goal)
            approach_dir = (ball - np.array([15.0, 0.0])) if self.world.team_side_is_left else (ball - np.array([-15.0, 0.0]))
            if np.linalg.norm(approach_dir) < 1e-6:
                approach_point = ball + np.array([-0.5, 0.0])
            else:
                approach_point = ball + (approach_dir / np.linalg.norm(approach_dir)) * 0.8
            drawer.line(mypos, approach_point, 2, drawer.Color.green, "approach")
            return self.move(approach_point, orientation=strategyData.ball_dir)

    # ---------- FAST / LIMITED computations ----------
    def _fast_clear_test(self, start, end, opponent_positions):
        """
        Fast path-clear test: checks up to self._max_opponents_check opponents.
        Returns True if path is likely clear.
        """
        if opponent_positions is None:
            return True
        path = np.array(end) - np.array(start)
        L = np.linalg.norm(path)
        if L < 0.01:
            return True
        path_dir = path / L
        checked = 0
        for opp in opponent_positions:
            if opp is None:
                continue
            checked += 1
            if checked > self._max_opponents_check:
                break
            opp_pos = np.array(opp)
            to_opp = opp_pos - start
            proj = np.dot(to_opp, path_dir)
            if 0 < proj < L:
                closest = start + proj * path_dir
                d = np.linalg.norm(opp_pos - closest)
                if d < 0.9:
                    return False
        return True

    def _fast_find_pass_target(self, strategyData):
        """
        Quick pass candidate selection — checks a few teammates and evaluates simple score:
         - prefer forward/progressive teammates
         - penalize nearby opponents along pass line
         - limit checks to keep CPU low
        """
        best_pos = None
        best_score = -1.0
        ball = strategyData.ball_2d
        mypos = strategyData.mypos
        goal = np.array([15.0, 0.0])

        # iterate teammates but limit how many we consider (fast)
        considered = 0
        for i, tpos in enumerate(strategyData.teammate_positions):
            if tpos is None:
                continue
            if i == strategyData.player_unum - 1:
                continue
            considered += 1
            if considered > self._max_teammates_check:
                break

            # simple geometry
            dist_from_me = np.linalg.norm(tpos - mypos)
            if dist_from_me < 1.0 or dist_from_me > 12.0:
                continue

            # forwardness: how much closer to opponent goal
            forward_gain = max(0.0, (np.linalg.norm(goal - mypos) - np.linalg.norm(goal - tpos)) / 30.0)

            # line-block penalty: check up to 2 nearest opponents for this candidate
            block_penalty = 0.0
            opp_checked = 0
            for opp in sorted([o for o in (strategyData.opponent_positions or []) if o is not None],
                              key=lambda x: np.linalg.norm(np.array(x) - ball)):
                if opp_checked >= 2:
                    break
                opp_checked += 1
                # distance from opponent to segment
                seg = tpos - ball
                seg_len_sq = np.sum(seg * seg)
                if seg_len_sq == 0:
                    d = np.linalg.norm(np.array(opp) - ball)
                else:
                    t = np.clip(np.dot((np.array(opp) - ball), seg) / seg_len_sq, 0.0, 1.0)
                    proj = ball + t * seg
                    d = np.linalg.norm(np.array(opp) - proj)
                if d < 1.0:
                    block_penalty += 0.5  # heavy penalty

            # candidate score
            score = forward_gain * 0.7 + np.clip((dist_from_me - 1.5) / 10.0, 0.0, 0.3) - block_penalty
            if score > best_score:
                best_score = score
                best_pos = tpos

        # final quick path clearance on best_pos
        if best_pos is not None:
            if not self._fast_clear_test(ball, best_pos, strategyData.opponent_positions):
                best_score *= 0.4

        return best_pos, best_score

    # ---------- SUPPORT ----------
    def support_play(self, strategyData, drawer, in_position):
        """Support players keep formation or pressure if a nearby opponent has the ball"""
        try:
            mypos = strategyData.mypos
            ball = strategyData.ball_2d
            # If opponent has ball and is near, pressure
            for opp in (strategyData.opponent_positions or [])[: self._max_opponents_check]:
                if opp is None:
                    continue
                if np.linalg.norm(np.array(opp) - ball) < 1.2:
                    # pressure that opponent
                    return self.move(np.array(opp), orientation=strategyData.ball_dir, avoid_obstacles=True)
            # otherwise hold formation
            if not in_position:
                drawer.line(mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "movement_line")
                return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
            # if already in pos, face ball and be ready
            return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        except Exception as e:
            print(f"[ERROR] support_play failed: {e}")
            import traceback
            traceback.print_exc()
            return self.move((0.0, 0.0), orientation=0.0)

    # ---------- UTIL ----------
    def am_i_in_position(self, target_position, threshold=0.6):
        r = self.world.robot
        my_pos = np.array(r.loc_head_position[:2], dtype=float)
        target = np.array(target_position, dtype=float)
        return float(np.linalg.norm(target - my_pos)) < float(threshold)

    # ---------- FAT-PROXY helpers (copied & safe) ----------
    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg(self.kick_direction - r.imu_torso_orientation):.2f} 20)"
            self.fat_proxy_walk = np.zeros(3)
            return True
        else:
            # approach
            self.fat_proxy_move(ball_2d + np.array([-0.1, 0.0]), None, True)
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot
        target = np.array(target_2d, dtype=float)
        target_dist = np.linalg.norm(target - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target.tolist())

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += f"(proxy dash {100} {0} {0})"
            return

        if target_dist < 0.1:
            if is_orientation_absolute and orientation is not None:
                orientation = M.normalize_deg(orientation - r.imu_torso_orientation)
            tdir = float(np.clip(orientation if orientation is not None else 0.0, -60.0, 60.0))
            self.fat_proxy_cmd += f"(proxy dash {0} {0} {tdir:.1f})"
        else:
            self.fat_proxy_cmd += f"(proxy dash {20} {0} {target_dir:.1f})"
