from Bezier import Bezier
import json
import math
import numpy as np

THIGH_LENGTH = 199.0
CALF_LENGTH = 195.0
ANKLE_LENGTH = 59.7
LEG_LENGTH = 453.7

class Matrix:
    def __init__(self) -> None:
        self.m = [
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1
        ]

    def SetTransform(self, point, angle):
        Cx = math.cos(angle[0] * math.pi / 180.0)
        Cy = math.cos(angle[1] * math.pi / 180.0)
        Cz = math.cos(angle[2] * math.pi / 180.0)
        Sx = math.sin(angle[0] * math.pi / 180.0)
        Sy = math.sin(angle[1] * math.pi / 180.0)
        Sz = math.sin(angle[2] * math.pi / 180.0)

        self.m[0] = Cz * Cy
        self.m[1] = Cz * Sy * Sx - Sz * Cx
        self.m[2] = Cz * Sy * Cx + Sz * Sx
        self.m[3] = point[0]
        self.m[4] = Sz * Cy
        self.m[5] = Sz * Sy * Sx + Cz * Cx
        self.m[6] = Sz * Sy * Sx + Cz * Cx
        self.m[7] = point[1]
        self.m[8] = -Sy
        self.m[9] = Cy * Sx
        self.m[10] = Cy * Cx
        self.m[11] = point[2]
        
    def Inverse(self):
        src = Matrix()
        dst = Matrix()
        tmp = Matrix()

        for i in range(4):
            src.m[i] = self.m[i*4]
            src.m[i + 4] = self.m[i*4 + 1]
            src.m[i + 8] = self.m[i*4 + 2]
            src.m[i + 12] = self.m[i*4 + 3]

        tmp.m[0] = src.m[10] * src.m[15]
        tmp.m[1] = src.m[11] * src.m[14]
        tmp.m[2] = src.m[9] * src.m[15]
        tmp.m[3] = src.m[11] * src.m[13]
        tmp.m[4] = src.m[9] * src.m[14]
        tmp.m[5] = src.m[10] * src.m[13]
        tmp.m[6] = src.m[8] * src.m[15]
        tmp.m[7] = src.m[11] * src.m[12]
        tmp.m[8] = src.m[8] * src.m[14]
        tmp.m[9] = src.m[10] * src.m[12]
        tmp.m[10] = src.m[8] * src.m[13]
        tmp.m[11] = src.m[9] * src.m[12]

        dst.m[0] = (tmp.m[0]*src.m[5] + tmp.m[3]*src.m[6] + tmp.m[4]*src.m[7]) - (tmp.m[1]*src.m[5] + tmp.m[2]*src.m[6] + tmp.m[5]*src.m[7])
        dst.m[1] = (tmp.m[1]*src.m[4] + tmp.m[6]*src.m[6] + tmp.m[9]*src.m[7]) - (tmp.m[0]*src.m[4] + tmp.m[7]*src.m[6] + tmp.m[8]*src.m[7])
        dst.m[2] = (tmp.m[2]*src.m[4] + tmp.m[7]*src.m[5] + tmp.m[10]*src.m[7]) - (tmp.m[3]*src.m[4] + tmp.m[6]*src.m[5] + tmp.m[11]*src.m[7])
        dst.m[3] = (tmp.m[5]*src.m[4] + tmp.m[8]*src.m[5] + tmp.m[11]*src.m[6]) - (tmp.m[4]*src.m[4] + tmp.m[9]*src.m[5] + tmp.m[10]*src.m[6])
        dst.m[4] = (tmp.m[1]*src.m[1] + tmp.m[2]*src.m[2] + tmp.m[5]*src.m[3]) - (tmp.m[0]*src.m[1] + tmp.m[3]*src.m[2] + tmp.m[4]*src.m[3])
        dst.m[5] = (tmp.m[0]*src.m[0] + tmp.m[7]*src.m[2] + tmp.m[8]*src.m[3]) - (tmp.m[1]*src.m[0] + tmp.m[6]*src.m[2] + tmp.m[9]*src.m[3])
        dst.m[6] = (tmp.m[3]*src.m[0] + tmp.m[6]*src.m[1] + tmp.m[11]*src.m[3]) - (tmp.m[2]*src.m[0] + tmp.m[7]*src.m[1] + tmp.m[10]*src.m[3])
        dst.m[7] = (tmp.m[4]*src.m[0] + tmp.m[9]*src.m[1] + tmp.m[10]*src.m[2]) - (tmp.m[5]*src.m[0] + tmp.m[8]*src.m[1] + tmp.m[11]*src.m[2])

        tmp.m[0] = src.m[2]*src.m[7]
        tmp.m[1] = src.m[3]*src.m[6]
        tmp.m[2] = src.m[1]*src.m[7]
        tmp.m[3] = src.m[3]*src.m[5]
        tmp.m[4] = src.m[1]*src.m[6]
        tmp.m[5] = src.m[2]*src.m[5]

        tmp.m[6] = src.m[0]*src.m[7]
        tmp.m[7] = src.m[3]*src.m[4]
        tmp.m[8] = src.m[0]*src.m[6]
        tmp.m[9] = src.m[2]*src.m[4]
        tmp.m[10] = src.m[0]*src.m[5]
        tmp.m[11] = src.m[1]*src.m[4]

        dst.m[8] = (tmp.m[0]*src.m[13] + tmp.m[3]*src.m[14] + tmp.m[4]*src.m[15]) - (tmp.m[1]*src.m[13] + tmp.m[2]*src.m[14] + tmp.m[5]*src.m[15])
        dst.m[9] = (tmp.m[1]*src.m[12] + tmp.m[6]*src.m[14] + tmp.m[9]*src.m[15]) - (tmp.m[0]*src.m[12] + tmp.m[7]*src.m[14] + tmp.m[8]*src.m[15])
        dst.m[10] = (tmp.m[2]*src.m[12] + tmp.m[7]*src.m[13] + tmp.m[10]*src.m[15]) - (tmp.m[3]*src.m[12] + tmp.m[6]*src.m[13] + tmp.m[11]*src.m[15])
        dst.m[11] = (tmp.m[5]*src.m[12] + tmp.m[8]*src.m[13] + tmp.m[11]*src.m[14]) - (tmp.m[4]*src.m[12] + tmp.m[9]*src.m[13] + tmp.m[10]*src.m[14])
        dst.m[12] = (tmp.m[2]*src.m[10] + tmp.m[5]*src.m[11] + tmp.m[1]*src.m[9]) - (tmp.m[4]*src.m[11] + tmp.m[0]*src.m[9] + tmp.m[3]*src.m[10])
        dst.m[13] = (tmp.m[8]*src.m[11] + tmp.m[0]*src.m[8] + tmp.m[7]*src.m[10]) - (tmp.m[6]*src.m[10] + tmp.m[9]*src.m[11] + tmp.m[1]*src.m[8])
        dst.m[14] = (tmp.m[6]*src.m[9] + tmp.m[11]*src.m[11] + tmp.m[3]*src.m[8]) - (tmp.m[10]*src.m[11] + tmp.m[2]*src.m[8] + tmp.m[7]*src.m[9])
        dst.m[15] = (tmp.m[10]*src.m[10] + tmp.m[4]*src.m[8] + tmp.m[9]*src.m[9]) - (tmp.m[8]*src.m[9] + tmp.m[11]*src.m[10] + tmp.m[5]*src.m[8])
        
        det = src.m[0]*dst.m[0] + src.m[1]*dst.m[1] + src.m[2]*dst.m[2] + src.m[3]*dst.m[3]

        if det == 0:
            det = 0
            return False
        else:
            det = 1/det

        for i in range(len(dst.m)):
            self.m[i] = dst.m[i] * det

        return True

    def copy(self, mat):
        for i in range(len(self.m)):
            self.m[i] = mat.m[i]

    def __mul__(self, mat):
        result = Matrix()
        result.m[0] = 0
        result.m[5] = 0
        result.m[10] = 0
        result.m[15] = 0

        for i in range(4):
            for j in range(4):
                for c in range(4):
                    result.m[j*4+i] += self.m[j*4+c] * mat.m[c*4+i]

        return result

def compute_ik(x, y, z, a, b, c):
    result = [0,0,0,0,0,0]
    Tad = Matrix() 
    Tda = Matrix() 
    Tcd = Matrix() 
    Tdc = Matrix() 
    Tac = Matrix()

    Tad.SetTransform(
        [x, y, z - LEG_LENGTH],
        [a, b, c])

    X = x + Tad.m[2] * ANKLE_LENGTH
    Y = y + Tad.m[6] * ANKLE_LENGTH
    Z = (z - LEG_LENGTH) + Tad.m[10] * ANKLE_LENGTH

    # Get Knee
    _Rac = math.sqrt(X*X + Y*Y + Z*Z)
    calc = (_Rac * _Rac - THIGH_LENGTH * THIGH_LENGTH - CALF_LENGTH * CALF_LENGTH) / (2 * THIGH_LENGTH * CALF_LENGTH)
    if calc > 1.0: return False
    _Acos = math.acos(
        (_Rac * _Rac - THIGH_LENGTH * THIGH_LENGTH - CALF_LENGTH * CALF_LENGTH) /
        (2 * THIGH_LENGTH * CALF_LENGTH))
    result[3] = _Acos

    # Get Ankle Roll
    Tda.copy(Tad)
    if (Tda.Inverse() == False) :
        return False

    _k = math.sqrt(Tda.m[7] * Tda.m[7] + Tda.m[11] * Tda.m[11])
    _l = math.sqrt(Tda.m[7] * Tda.m[7] + (Tda.m[11] - ANKLE_LENGTH) * (Tda.m[11] - ANKLE_LENGTH))
    _m = (_k * _k - _l * _l - ANKLE_LENGTH * ANKLE_LENGTH) / (2 * _l * ANKLE_LENGTH)
    if (_m > 1.0):
        _m = 1.0
    elif (_m < -1.0):
        _m = -1.0

    _Acos = math.acos(_m)
    if (math.isnan(_Acos) == True):
        return False

    if (Tda.m[7] < 0.0):
        result[5] = -_Acos
    else:
        result[5] = _Acos

    # Get Hip Yaw
    Tcd.SetTransform(
        [0, 0, -ANKLE_LENGTH],
        [result[5] * 180.0 / math.pi, 0, 0])
    Tdc.copy(Tcd)
    if (Tdc.Inverse() == False):
        return False

    Tac = Tad * Tdc
    _Atan = math.atan2(-Tac.m[1], Tac.m[5])
    if (math.isinf(_Atan) == 1):
        return False

    result[0] = _Atan

    # Get Hip Roll
    _Atan = math.atan2(Tac.m[9], -Tac.m[1] * math.sin(result[0]) + Tac.m[5] * math.cos(result[0]))
    if (math.isinf(_Atan) == True):
        return False
    result[1] = _Atan

    # Get Hip Pitch and Ankle Pitch
    _Atan = math.atan2(
        Tac.m[2] * math.cos(result[0]) + Tac.m[6] * math.sin(result[0]), Tac.m[0] * math.cos(
        result[0]) + Tac.m[4] * math.sin(result[0]))
    if (math.isinf(_Atan) == True):
        return False

    _theta = _Atan
    _k = math.sin(result[3]) * CALF_LENGTH
    _l = -THIGH_LENGTH - math.cos(result[3]) * CALF_LENGTH
    _m = math.cos(result[0]) * X + math.sin(result[0]) * Y
    _n = math.cos(result[1]) * Z + math.sin(result[0]) * math.sin(result[1]) * X - math.cos(result[0]) * math.sin(
        result[1]) * Y
    _s = (_k * _n + _l * _m) / (_k * _k + _l * _l)
    _c = (_n - _k * _s) / _l
    _Atan = math.atan2(_s, _c)
    if (math.isinf(_Atan) == True):
        return False

    result[2] = _Atan
    result[4] = _theta - result[3] - result[2]

    return result

def save_to_json(json_name: str, action_name: str, next: str, curve_set: np.array):
    data = {}
    data["name"] = action_name
    data["next"] = next
    data["poses"] = []

    for i in range(len(curve_set)):
        pose = {}
        pose["joints"] = {}
        pose["name"] = "pose" + str(i)
        pose["pause"] = 0
        pose["speed"] = 0.04
        
        ik_result = compute_ik(curve_set[i][0], curve_set[i][1], curve_set[i][2], 0, 0, 0)
        pose['joints']["left_ankle_pitch"] = math.degrees(ik_result[4])
        pose['joints']["left_ankle_roll"] = math.degrees(ik_result[5])
        pose['joints']["left_elbow"] = 0
        pose['joints']["left_hip_pitch"] = math.degrees(ik_result[2])
        pose['joints']["left_hip_roll"] = math.degrees(ik_result[1])
        pose['joints']["left_hip_yaw"] = math.degrees(ik_result[0])
        pose['joints']["left_knee"] = math.degrees(ik_result[3])
        pose['joints']["left_shoulder_pitch"] = 0
        pose['joints']["left_shoulder_roll"] = 0
        pose['joints']["neck_pitch"] = 0
        pose['joints']["neck_yaw"] = 0
        pose['joints']["right_ankle_pitch"] = math.degrees(ik_result[4])
        pose['joints']["right_ankle_roll"] = math.degrees(ik_result[5])
        pose['joints']["right_elbow"] = 0
        pose['joints']["right_hip_pitch"] = math.degrees(ik_result[2])
        pose['joints']["right_hip_roll"] = math.degrees(ik_result[1])
        pose['joints']["right_hip_yaw"] = math.degrees(ik_result[0])
        pose['joints']["right_knee"] = math.degrees( ik_result[3])
        pose['joints']["right_shoulder_pitch"] = 0
        pose['joints']["right_shoulder_roll"] = 0 

        data["poses"].append(pose)
  
    data["start_delay"] = 0
    data["stop_delay"] = 0

    with open("data/" + json_name + ".json", "w") as f:
        json.dump(data, f, indent=2)
    # print(json.dumps(data, indent=2))


def main():
    ## Raising foot kick ##
    points_set_1 = np.array([[0, 4.5, 5.5], [0, 4.5, 7], [-8, 5.5, 7], [-12, 6.5, 9], [-12, 7.5, 13.5]])
    t_points = np.arange(0, 1, 0.04)
    curve_set_1 = Bezier.Curve(t_points, points_set_1)

    ## Kick path foot ##
    #points_set_1 = np.array([[-12, 7.5, 13.5], [-8, 7.5, 6.5], [-1.34, 7.5, 5.45], [20, 7.5, 6.5]])
    #t_points = np.arange(0, 1, 0.04)
    #curve_set_1 = Bezier.Curve(t_points, points_set_1)

    # print(compute_ik(10, 10, 20, 0, 0, 0))
    save_to_json("kick", "kick", "", curve_set_1)

if __name__ == '__main__':
    main()