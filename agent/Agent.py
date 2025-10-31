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
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)

        # Our goal is at -15, opponent goal at +15 (LEFT side)
        self.init_pos = ([-14,0],[-5,4],[-9,0],[-5,-4],[-2.3,0],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1]


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:]
        self.state = 0

        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1])))
        else:
            if self.fat_proxy_cmd is None:
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)

    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        r = self.world.robot

        if self.fat_proxy_cmd is not None:
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute)
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)

    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        return self.behavior.execute("Dribble",None,None)

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
            return self.fat_proxy_kick()

    def kickTarget(self, strategyData, mypos_2d=(0,0), target_2d=(0,0), abort=False, enable_pass_command=False):
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        kick_distance = np.linalg.norm(vector_to_target)
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        kick_direction = np.degrees(direction_radians)

        self.kick_direction = kick_direction
        self.kick_distance = kick_distance

        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
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
            self.beam(True)
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass

        self.radio.broadcast()

        if self.fat_proxy_cmd is None:
            self.scom.commit_and_send(strategyData.robot_model.get_command())
        else:
            self.scom.commit_and_send(self.fat_proxy_cmd.encode()) 
            self.fat_proxy_cmd = ""
        
    def select_skill(self, strategyData):
        drawer = self.world.draw
        r = self.world.robot
        unum = int(r.unum)
        
        # Field constants
        our_goal = np.array([-15.0, 0.0])
        opponent_goal = np.array([15.0, 0.0])
        my_pos = np.array(strategyData.mypos)
        ball_2d = np.array(strategyData.ball_2d)
        
        i_am_active = strategyData.active_player_unum == strategyData.robot_model.unum
        dist_to_ball = strategyData.ball_dist
        dist_to_opp_goal = np.linalg.norm(my_pos - opponent_goal)
        
        # Check if ball is in our defensive half (x < -2)
        ball_in_our_half = ball_2d[0] < -2.0
        
        #------------------------------------------------------
        # KEEPER LOGIC (unum == 1)
        if unum == 1:
            
            # Check for threats near our goal
            opponents_near_goal = []
            for opp_pos in strategyData.opponent_positions:
                if opp_pos is None:
                    continue
                opp = np.array(opp_pos)
                if opp[0] < -5.0:
                    dist_to_goal = np.linalg.norm(opp - our_goal)
                    if dist_to_goal < 8.0:
                        opponents_near_goal.append(opp)
            
            # Defensive response
            if len(opponents_near_goal) > 0:
                closest_opp = min(opponents_near_goal, key=lambda o: np.linalg.norm(o - our_goal))
                
                if ball_2d[0] < -8.0 and dist_to_ball < 4.0:
                    if dist_to_ball > 0.5:
                        return self.move(ball_2d, orientation=strategyData.ball_dir, avoid_obstacles=False, timeout=600)
                    else:
                        return self.kickTarget(strategyData, my_pos, opponent_goal, enable_pass_command=True)
                else:
                    opp_to_goal_vec = our_goal - closest_opp
                    intercept_pos = closest_opp + 0.6 * opp_to_goal_vec
                    intercept_pos[0] = max(intercept_pos[0], -13.5)
                    intercept_pos[1] = np.clip(intercept_pos[1], -1.5, 1.5)
                    return self.move(tuple(intercept_pos), orientation=strategyData.ball_dir, avoid_obstacles=False, timeout=600)
            
            # Check if closest to ball
            is_closest = True
            for idx, tpos in enumerate(strategyData.teammate_positions):
                if tpos is None or (idx + 1) == unum:
                    continue
                t_dist = np.linalg.norm(np.array(tpos) - ball_2d)
                if t_dist < (dist_to_ball - 0.5):
                    is_closest = False
                    break
            
            if is_closest and dist_to_ball < 3.0:
                if dist_to_ball > 0.5:
                    return self.move(ball_2d, orientation=strategyData.ball_dir, avoid_obstacles=False, timeout=600)
                else:
                    return self.kickTarget(strategyData, my_pos, opponent_goal, enable_pass_command=True)
            else:
                keeper_pos = np.array([-14.5, np.clip(ball_2d[1], -1.1, 1.1)])
                return self.move(keeper_pos, orientation=0, avoid_obstacles=False, timeout=800)
        
        #------------------------------------------------------
        # ATTACKERS LOGIC (players 2, 4, 5)
        elif unum in [2, 4, 5]:
            attacker_y_positions = {2: -3.0, 4: 0.0, 5: 3.0}
            target_y = attacker_y_positions[unum]
            
            # DEFENSIVE MODE: Help defend when ball is in our half
            if ball_in_our_half and not i_am_active and unum in [2, 4]:
                # Position defenders around goal in a spaced manner
                # Player 2: left side, Player 4: right side
                defensive_x = max(ball_2d[0] - 3.0, -13.0)  # Stay behind ball
                defensive_y = -2.5 if unum == 2 else 2.5
                
                # Adjust based on ball y position for better coverage
                if abs(ball_2d[1]) > 3.0:
                    defensive_y = np.clip(ball_2d[1] - (1.5 if unum == 2 else -1.5), -4.0, 4.0)
                
                defensive_pos = np.array([defensive_x, defensive_y])
                defensive_pos[0] = np.clip(defensive_pos[0], -14.5, 0.0)
                defensive_pos[1] = np.clip(defensive_pos[1], -9.8, 9.8)
                
                return self.move(tuple(defensive_pos), orientation=strategyData.ball_dir, avoid_obstacles=True, timeout=800)
            
            # ACTIVE ATTACKING MODE
            if i_am_active:
                
                # Move to ball if far
                if dist_to_ball > 0.5:
                    try:
                        intercept_pt, intercept_dist = self.world.get_intersection_point_with_ball(player_speed=1.2)
                        target_pt = intercept_pt if intercept_dist < 8.0 else ball_2d
                    except Exception:
                        target_pt = ball_2d
                    return self.move(target_pt, orientation=strategyData.ball_dir, avoid_obstacles=False, timeout=600)
                
                # CLOSE TO BALL - SHOOTING LOGIC
                # Always shoot if within 5 units of opponent goal (NO PASS EVALUATION)
                if dist_to_opp_goal <= 5.0:
                    return self.kickTarget(strategyData, my_pos, opponent_goal, enable_pass_command=True)
                
                # Shoot immediately if opponent is very close
                if strategyData.min_opponent_ball_dist < 1.2:
                    return self.kickTarget(strategyData, my_pos, opponent_goal, enable_pass_command=True)
                
                # Quick pass evaluation (only when not in shooting range)
                best_target = opponent_goal
                best_score = 4.0
                
                for idx, tpos in enumerate(strategyData.teammate_positions):
                    if tpos is None or (idx + 1) == unum:
                        continue
                    t = np.array(tpos)
                    
                    # Must be significantly forward
                    forward_gain = (t[0] - my_pos[0])
                    if forward_gain < 1.0:
                        continue
                    
                    dist = np.linalg.norm(t - my_pos)
                    if dist > 7.0 or dist < 3.0:
                        continue
                    
                    # Quick opponent check
                    min_opp_dist = min((np.linalg.norm(np.array(opp) - t) 
                                       for opp in strategyData.opponent_positions if opp is not None), default=10.0)
                    if min_opp_dist < 2.0:
                        continue
                    
                    score = forward_gain * 2.5 - dist * 0.3
                    if score > best_score:
                        best_score = score
                        best_target = t
                
                # Execute pass or shoot
                if best_score > 4.0 and np.array_equal(best_target, opponent_goal) == False:
                    # LEAD THE PASS: Aim slightly ahead of teammate toward opponent goal
                    teammate_to_goal = opponent_goal - best_target
                    teammate_to_goal_normalized = teammate_to_goal / np.linalg.norm(teammate_to_goal)
                    
                    # Lead distance based on pass distance (longer pass = more lead)
                    pass_distance = np.linalg.norm(best_target - my_pos)
                    lead_distance = min(pass_distance * 0.3, 2.0)  # Max 2m lead
                    
                    # Calculate lead point ahead of teammate
                    lead_target = best_target + (teammate_to_goal_normalized * lead_distance)
                    
                    # Clamp to field bounds
                    lead_target[0] = np.clip(lead_target[0], -14.8, 14.8)
                    lead_target[1] = np.clip(lead_target[1], -9.8, 9.8)
                    
                    return self.kickTarget(strategyData, my_pos, tuple(lead_target), enable_pass_command=True)
                else:
                    return self.kickTarget(strategyData, my_pos, opponent_goal, enable_pass_command=True)
            
            # NOT ACTIVE - POSITIONING MODE
            else:
                # Move forward maintaining spacing
                target_x = min(ball_2d[0] + 3.0, 12.0)
                
                # Collision avoidance with other attackers
                other_attackers_pos = []
                for idx, tpos in enumerate(strategyData.teammate_positions):
                    if tpos is None or (idx + 1) == unum or (idx + 1) not in [2, 4, 5]:
                        continue
                    other_attackers_pos.append((np.array(tpos), idx + 1))
                
                adjusted_target_y = target_y
                MIN_SEPARATION = 2.5
                
                for other_pos, other_unum in other_attackers_pos:
                    if abs(other_pos[0] - target_x) < 4.0:
                        dist_y = abs(other_pos[1] - adjusted_target_y)
                        if dist_y < MIN_SEPARATION:
                            if other_pos[1] > adjusted_target_y:
                                adjusted_target_y = other_pos[1] - MIN_SEPARATION
                            else:
                                adjusted_target_y = other_pos[1] + MIN_SEPARATION
                
                target_pos = np.array([target_x, adjusted_target_y])
                target_pos[0] = np.clip(target_pos[0], -14.8, 14.8)
                target_pos[1] = np.clip(target_pos[1], -9.8, 9.8)
                
                return self.move(tuple(target_pos), orientation=strategyData.ball_dir, avoid_obstacles=True, timeout=800)
        
        #------------------------------------------------------
        # DEFENDER LOGIC (player 3)
        else:  # unum == 3
            
            # Find opponents past defender
            opponents_past = []
            for idx, opp_pos in enumerate(strategyData.opponent_positions):
                if opp_pos is None:
                    continue
                opp = np.array(opp_pos)
                if opp[0] < my_pos[0]:
                    opponents_past.append((opp, idx + 1))
            
            # Defend if multiple opponents past
            if len(opponents_past) > 1:
                opponents_past.sort(key=lambda x: x[0][0])
                target_opponent = opponents_past[0][0]
                
                target_x = max(target_opponent[0] - 1.5, -14.0)
                target_y = target_opponent[1]
                target_x = min(target_x, 0.0)
                
                target_pos = np.array([target_x, target_y])
                target_pos[0] = np.clip(target_pos[0], -14.8, 0.0)
                target_pos[1] = np.clip(target_pos[1], -9.8, 9.8)
                
                return self.move(tuple(target_pos), orientation=strategyData.ball_dir, avoid_obstacles=False, timeout=800)
            else:
                # Default defensive positioning
                default_defensive_x = min(ball_2d[0] - 2.0, -3.0)
                default_defensive_x = min(default_defensive_x, 0.0)
                
                default_y = my_pos[1] * 0.8 if abs(my_pos[1]) < 3.0 else 0.0
                
                defensive_pos = np.array([default_defensive_x, default_y])
                defensive_pos[0] = np.clip(defensive_pos[0], -14.8, 0.0)
                defensive_pos[1] = np.clip(defensive_pos[1], -9.8, 9.8)
                
                return self.move(tuple(defensive_pos), orientation=strategyData.ball_dir, avoid_obstacles=False, timeout=800)
        
    
    #--------------------------------------- Fat proxy auxiliary methods

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
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True)
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
                orientation = M.normalize_deg(orientation - r.imu_torso_orientation)
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")