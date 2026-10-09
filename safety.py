# safety.py
import numpy as np

 # DEFAULT ROBOT VELOCITY LIMITS
ROBOT_VELOCITY_LIMITS = {
    "Kawasaki": 1.0,
    "Staubli": 1.0,
    "Nachi": 1.0
}

def _get_robot(arm):
    """"
    Accept either:
    - wrappar class (e.g. KawasakiPickPlace) with a .robot attribute, or
    - the actual DHRobot3D object (e.g. RS007N)
    Retun the actual DHRobot3D object.
    
    """
    return arm.robot if hasattr(arm, 'robot') else arm
def _joint_positions(robot,q = None):
    """
    Return the XYZ positions of every robot frame.

    These points approximate each physical link as a straight line between neighbouring joint frames.
    """
    robot = _get_robot(robot)

    if q is None:
        q = robot.q

    transforms = robot.fkine_all(q)
    points = []
    for T in transforms:
        points.append(np.array(T.t, dtype=float))
    return np.asarray(points)

def _point_segment_distance(point, seg_a, seg_b):
    """

    Shortest distance between a point and a line segment AB

    """
    point = np.asarray(point, dtype=float)
    seg_a = np.asarray(seg_a, dtype=float)
    seg_b = np.asarray(seg_b, dtype=float)

    ab = seg_b - seg_a
    denominator = np.dot(ab, ab)

    small = 1e-12
    if denominator < small:
        return np.linalg.norm(point - seg_a)
    t = np.dot(point - seg_a, ab) / denominator

    #Keep the closet location inside the segment
    t = np.clip(t, 0.0, 1.0)
    closest_point = seg_a + t * ab
    return np.linalg.norm(point - closest_point)

def _segment_segment_distance(p1,p2,q1,q2):
    """
    Shortest distance between two 3D line segments: P1 -> P2 and Q1 -> Q2

    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    denominator = a * c - b * b

    small = 1e-12

    if denominator < small:
        # Segments are parallel
        distance = [_point_segment_distance(p1, q1, q2), _point_segment_distance(p2, q1, q2),             
                    _point_segment_distance(q1, p1, p2), _point_segment_distance(q2, p1, p2)]
        return min(distance)
    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    point_on_p = p1 + s * u
    point_on_q = q1 + t * v

    return np.linalg.norm(point_on_p - point_on_q)

# COLLISION SAFETY FUNCTIONS

class CollisionSafety:
    """
    Collision monitoring for the robotic workcell

    Robot links are approximated using line segments between 
    consecutive joint frames. 
    
    The distance between each pair of segments is computed and compared to a safety threshold.

    This is intentionally conservative:
    if 2 links come closer than the specific clearance, the pose will treated as unsafe, even if the links do not actually collide.
    
    """
    def __init__(self,robot_clearance = 0.10,
                 self_clearance = 0.06,
                 obstacle_clearance = 0.05):
        self.robot_clearance = robot_clearance
        self.self_clearance = self_clearance
        self.obstacle_clearance = obstacle_clearance

    #ROBOT VERSUS ROBOT
    def robot_robot(self, arm1,arm2, q1 = None, q2 = None, clearance = None):
        robot1 = _get_robot(arm1)
        robot2 = _get_robot(arm2)

        if clearance is None:
            clearance = self.robot_clearance
        points1 = _joint_positions(robot1, q1)
        points2 = _joint_positions(robot2, q2)

        minimum_distance = np.inf
        for i in range(len(points1)-1):
            a1 = points1[i]
            a2 = points1[i+1]

            for j in range(len(points2)-1):
                b1 = points2[j]
                b2 = points2[j+1]

                distance = _segment_segment_distance(a1,a2,b1,b2)
                minimum_distance = min(minimum_distance, distance)

                if distance < clearance:
                    return {"safe": False, "distance": distance, "link1": (i), "link2": (j), "reason":"Robot-to-robot collision risk" }

        return {"safe": True, "distance": minimum_distance, "reason": "No collision risk" }

    #SELF COLLISION
    def self_collision(self, arm, q = None, clearance = None):
        robot = _get_robot(arm)

        if clearance is None:
            clearance = self.self_clearance
        points = _joint_positions(robot, q)

        minimum_distance = np.inf
        link_count = len(points)-1

        for i in range(link_count):
            for j in range(i+1,link_count): 

                # Adjacent links share a joint, so ignore them
                if abs(i-j) == 1:
                    continue

                #Neighbouring links share a joint, therefore they should not be treated as a collision
                distance = _segment_segment_distance(points[i], points[i+1], points[j], points[j+1] )
                minimum_distance = min(minimum_distance, distance)

                if distance < clearance:
                    return {"safe": False, "distance": distance, "link1": (i), "link2": (j), "reason":"Self-collision risk" }

        return {"safe": True, "distance": minimum_distance, "reason": "No self-collision detected" }

    #ROBOT VERSUS SPHERICAAL OBSTACLE
    def sphere_obstacle(self, arm, sphere_center, sphere_radius, q = None, clearance = None):
        robot = _get_robot(arm)

        if clearance is None:
            clearance = self.obstacle_clearance

        sphere_center = np.asarray(sphere_center, dtype=float)
        points = _joint_positions(robot, q)
        minimum_distance = np.inf

        link_count = len(points)-1

        for i in range(link_count):
            distance = _point_segment_distance(sphere_center, points[i], points[i+1] )

            #distance from robot link to obstacle surface
            surface_distance = distance - sphere_radius
            minimum_distance = min(minimum_distance, surface_distance)

            if surface_distance < clearance:
                return {"safe": False, "distance": surface_distance, "link": (i), "reason":"Collision risk with spherical obstacle" }

        return {"safe": True, "distance": minimum_distance, "reason": "No collision risk with spherical obstacle" }

class SingularitySafety:
    """
    Detects whether a robot is approaching a kinematic
    singularity using the Jacobian matrix.
    """

    def __init__(self,sigma_warning=0.05,sigma_stop=0.01,condition_warning=100.0):
        self.sigma_warning = sigma_warning
        self.sigma_stop = sigma_stop
        self.condition_warning = condition_warning

    def metrics(self, arm, q=None):
        robot = _get_robot(arm)

        if q is None:
            q = robot.q

        J = np.asarray(robot.jacob0(q),dtype=float)

        singular_values = np.linalg.svd(J,compute_uv=False)

        sigma_max = np.max(singular_values)
        sigma_min = np.min(singular_values)

        if sigma_min < 1e-12:
            condition_number = np.inf
        else:
            condition_number = sigma_max / sigma_min

        manipulability = float(
            np.prod(singular_values)
        )

        rank = int(
            np.linalg.matrix_rank(J)
        )

        return {"jacobian": J,"singular_values": singular_values,"sigma_max": float(sigma_max),"sigma_min": float(sigma_min),"condition_number": float(condition_number),"manipulability": manipulability,"rank": rank}

    def check(self, arm, q=None):
        data = self.metrics(arm,q)

        sigma = data["sigma_min"]
        condition = data["condition_number"]
        rank = data["rank"]

        # Severe singularity
        if rank < 6 or sigma <= self.sigma_stop:
            data.update({"safe": False,"level": "STOP","reason": "Severe singularity detected"})
            return data

        # Approaching singularity
        if (
            sigma <= self.sigma_warning
            or condition >= self.condition_warning
        ):
            data.update({"safe": True,"level": "WARNING","reason": "Approaching singularity"})
            return data

        # Safe configuration
        data.update({"safe": True,"level": "SAFE","reason": "Robot is in a safe configuration"})

        return data

#ROBOT VELOCITY & E-STOP SAFETY
class RobotSafetyController:
    """
    Controls the maximum allowable robot velocity based on:
    
    1. Operator/manual velocity request
    2. Collision status (robot-to-robot, self-collision, robot-to-obstacle)
    3. Singularity status (approaching kinematic singularity)
    4. Emergency stop status (E-STOP button pressed)

    Safety alwasy overrides the operator setting.
    
    """
    def __init__(self, max_velocity=1.0, warning_scale=0.5):

        self.max_velocity = float(max_velocity)

        #User-selected percentage of maximum velocity (0.0 to 1.0)
        self.manual_scale = 1.0

        #Velocity reduction when approaching singularity
        self.warning_scale = warning_scale

        #E-stop state
        self.estop_active = False

    #MANUAL VELOCITY CONTROL
    def set_manual_scale(self, scale):
        """
        Set operator requested velocity scale (0.0 to 1.0)
        
        example:
            1.0 = 100%
            0.5 = 50%
            0.2 = 20%

        """
        self.manual_scale = float(np.clip(scale, 0.0, 1.0))

    #E-STOP CONTROL

    def trigger_estop(self):
        """
        Trigger the E-STOP button.
        This will immediately stop all robot motion.
        """
        self.estop_active = True
    def reset_estop(self):
        """
        Reset the E-STOP button.
        This will allow robot motion to resume.
        """
        self.estop_active = False
        
    def is_estopped(self):
        """
        Return True if the E-STOP button is currently active.
        """
        return self.estop_active

    #SAFETY VELOCITY CALCULATION
    def get_velocity_scale(self, collision_result,singularity_result):
        """
        Determine final velocity scale
        Priority: 
        E-stop -> 0%
        Collision risk -> 0%
        Severe singularity -> 0%
        Singularity warning -> reduced velocity
        Otherwise -> operator requested velocity

        """
        #Highest priority: E-STOP button pressed
        if self.estop_active:
            return 0.0

        #Collision = stop immediately
        if not collision_result["safe"]:
            return 0.0

        singularty_level = singularity_result["level"]

        #Severe singularity = stop immediately
        if singularty_level == "STOP":
            return 0.0

        # Reduce velocity if approaching singularity
        if singularty_level == "WARNING":
            safety_scale = self.warning_scale
        else:
            safety_scale = 1.0
        #Safety limit must override user setting
        return min(self.manual_scale, safety_scale)
    def get_allowed_velocity(self,collision_result, singularity_result):

        """
        Return actual allowable robot velocity
        
        """
        scale = self.get_velocity_scale(collision_result, singularity_result)
        return self.max_velocity * scale