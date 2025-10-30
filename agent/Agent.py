from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation

        # lightweight caches to avoid heavy per-tick compute
        self._formation_cache_time_ms = -1000
        self._formation_points_cache = None
        self._assignment_cache_time_ms = -1000
        self._point_preferences_cache = None


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        # Clamp target to field bounds and enforce role-side constraints (keeper/defender in our half)
        tx, ty = float(target_2d[0]), float(target_2d[1])
        # Field bounds with small margin
        tx = max(-14.8, min(14.8, tx))
        ty = max(-9.8,  min(9.8,  ty))
        # Keeper (1) and main defender (5) stay in our half (x <= 0 always)
        if int(r.unum) in (1,5):
            tx = min(tx, 0.0)
        target_2d = (tx, ty)

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(np.array(target_2d) - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""



        



    def select_skill(self,strategyData):
        #--------------------------------------- 2. Decide action
        drawer = self.world.draw
        path_draw_options = self.path_manager.draw_options


        #------------------------------------------------------
        #Role Assignment
        if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
            drawer.annotation((0,10.5), "Role Assignment Phase" , drawer.Color.yellow, "status")
        else:
            drawer.clear("status")

        # compute formation and role assignment with throttling to reduce CPU
        now_ms = self.world.time_local_ms
        if self._formation_points_cache is None or (now_ms - self._formation_cache_time_ms) > 120:
            self._formation_points_cache = GenerateBasicFormation(strategyData)
            self._formation_cache_time_ms = now_ms
        if self._point_preferences_cache is None or (now_ms - self._assignment_cache_time_ms) > 200:
            try:
                self._point_preferences_cache = role_assignment(strategyData.teammate_positions, self._formation_points_cache)
            except Exception:
                self._point_preferences_cache = {}
            self._assignment_cache_time_ms = now_ms
        formation_positions = self._formation_points_cache
        point_preferences = self._point_preferences_cache
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.my_desired_position)

        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2,drawer.Color.blue,"target line")

        # Determine if I am the active player now (closest to ball)
        i_am_active = strategyData.active_player_unum == strategyData.robot_model.unum

        # If not active and formation is not ready, prioritize getting into position
        if (not i_am_active) and (not strategyData.IsFormationReady(point_preferences)):
            return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation, avoid_obstacles=False, timeout=800)
        #else:
        #     return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)


    
        #------------------------------------------------------
        # Example Behaviour
        target = (15,0) # Opponents Goal

        if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
            drawer.annotation((0,10.5), "Pass Selector Phase" , drawer.Color.yellow, "status")
        else:
            drawer.clear_player()

        if i_am_active: # I am the active player 
            # Go to ball first if far, otherwise pass/shoot
            if strategyData.ball_dist > 0.6:
                try:
                    intercept_pt, intercept_dist = self.world.get_intersection_point_with_ball(player_speed=1.2)
                    target_pt = intercept_pt if intercept_dist < 8.0 else strategyData.ball_2d
                except Exception:
                    target_pt = strategyData.ball_2d
                return self.move(target_pt, orientation=strategyData.ball_dir, avoid_obstacles=False, timeout=600)

            # Simple pass/shoot decision: prefer forward teammate in space; otherwise shoot
            my_pos = np.array(strategyData.mypos)
            # Opponent goal is always at +x (15,0) per server handling
            goal = np.array((15.0,0.0))

            best_target = goal
            best_score = -1e9

            # Evaluate teammates as pass targets
            for idx, tpos in enumerate(strategyData.teammate_positions):
                if tpos is None:
                    continue
                if idx + 1 == strategyData.player_unum:
                    continue
                t = np.array(tpos)
                # Prefer targets ahead towards +x goal
                forward_gain = (t[0] - my_pos[0])
                # Disallow backward passes relative to ball (small tolerance)
                ball_x = float(strategyData.ball_2d[0])
                backward_rel_ball = (t[0] < ball_x - 0.3)
                if forward_gain < -0.2 or backward_rel_ball:
                    continue
                dist = np.linalg.norm(t - my_pos)
                if dist > 12.0:
                    continue
                # Opponent proximity penalty near target
                opp_penalty = 0.0
                for opp in strategyData.opponent_positions:
                    if opp is None:
                        continue
                    d = np.linalg.norm(np.array(opp) - t)
                    if d < 2.5:
                        opp_penalty += (2.5 - d)
                score = forward_gain * 2.2 - dist * 0.45 - opp_penalty * 1.5
                if score > best_score:
                    best_score = score
                    best_target = t

            drawer.line(strategyData.mypos, tuple(best_target), 2,drawer.Color.red,"pass line")
            return self.kickTarget(strategyData, strategyData.mypos, tuple(best_target))
        else:
            drawer.clear("pass line")
            # Enforce conservative bounds for wide/support players (e.g., 4 and 5)
            dpos = np.array(strategyData.my_desired_position)
            attack_dir = 1 if strategyData.side == 0 else -1
            unum = int(self.world.robot.unum)
            # Player 5: never cross halfway
            if unum == 5:
                if attack_dir == 1:
                    dpos[0] = min(dpos[0], 0.0)
                else:
                    dpos[0] = max(dpos[0], 0.0)
            # Player 4: not more than halfway through opponent half (~+/-7.5 from mid)
            if unum == 4:
                limit = 7.5
                if attack_dir == 1:
                    dpos[0] = min(dpos[0], limit)
                else:
                    dpos[0] = max(dpos[0], -limit)
            # If ball is in our half, pull both back to our half line
            ball_x = float(strategyData.ball_2d[0])
            if (attack_dir == 1 and ball_x < 0.0) or (attack_dir == -1 and ball_x > 0.0):
                if attack_dir == 1:
                    dpos[0] = min(dpos[0], -0.2 if unum == 5 else 0.5)
                else:
                    dpos[0] = max(dpos[0], 0.2 if unum == 5 else -0.5)
            return self.move(dpos, orientation=strategyData.ball_dir, avoid_obstacles=False, timeout=800)
        
































    

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")