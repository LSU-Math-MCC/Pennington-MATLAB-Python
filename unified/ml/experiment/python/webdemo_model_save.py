from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler

from utilities.common_functions import Map, timestamp
from ml_core import ext_train_save
from DataSets import StykuDataSet

left_arm  = Map(name="Left Arm", short_name="LARM",
                dexa =Map(volume="LARM_volume", fat_mass="LARM_FAT", lean_mass="LARM_LEAN" , bmc= "LARM_BMC",pmass="LARM_PFAT"),
                styku=Map(volume="Styku_lArmVolume"),
                ss20 =Map(volume="SS20_lArmVolume"))
right_arm = Map(name="Right Arm", short_name="RARM",
                dexa =Map(volume="RARM_volume", fat_mass="RARM_FAT", lean_mass="RARM_LEAN", bmc= "RARM_BMC",pmass="RARM_PFAT"),
                styku=Map(volume="Styku_rArmVolume"),
                ss20 =Map(volume="SS20_rArmVolume"))
left_leg  = Map(name="Left Leg", short_name="LLEG",
                dexa =Map(volume="LLEG_volume", fat_mass="L_LEG_FAT", lean_mass="L_LEG_LEAN" ,bmc= "LLEG_BMC",pmass="L_LEG_PFAT"),
                styku=Map(volume="Styku_lLegVolume"),
                ss20 =Map(volume="SS20_lLegVolume"))
right_leg = Map(name="Right Leg", short_name="RLEG",
                dexa =Map(volume="RLEG_volume", fat_mass="R_LEG_FAT", lean_mass="R_LEG_LEAN" ,bmc= "RLEG_BMC",pmass="R_LEG_PFAT"),
                styku=Map(volume="Styku_rLegVolume"),
                ss20 =Map(volume="SS20_rLegVolume"))
head = Map(name="Head", short_name="HEAD",
                dexa =Map(volume="HEAD_volume", fat_mass="HEAD_FAT", lean_mass="HEAD_LEAN",bmc= "HEAD_BMC",pmass="HEAD_PFAT"),
                styku=Map(volume="Styku_headVolume"),
                ss20 =Map(volume="SS20_headVolume"))
trunk = Map(name="Trunk", short_name="TRUNK",
                dexa =Map(volume="TRUNK_volume", fat_mass="TRUNK_FAT", lean_mass="TRUNK_LEAN", bmc = 'TRUNK_BMC',pmass="TRUNK_PFAT"),
                styku=Map(volume="Styku_trunkVolume"),
                ss20 =Map(volume="SS20_trunkVolume"))
total = Map(name="Trunk", short_name="TRUNK",
                dexa =Map(volume="TRUNK_volume", fat_mass="TOTAL_FAT", lean_mass="TOTAL_LEAN", bmc = 'WBTOT_BMC',pmass="TOTAL_PFAT"),
                styku=Map(volume="Styku_TotalVolume"),
                ss20 =Map(volume="SS20_TotalVolume"))
body_parts = [left_arm, right_arm, left_leg, right_leg, head, trunk, total]

targets = ["FAT", "LEAN", "PFAT", "BMC"]
model = MLPRegressor(solver="lbfgs", activation="identity", max_iter=800, hidden_layer_sizes=(1,))
dataset = StykuDataSet()

bmi = ['BMI1']
age = ['age']
sex = ['SEX']
volumes = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
m_common = ["waist circ", "hip circ",
            "rThighGirth", "rbicepGirth"]
m_all = ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                    "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                    "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                    "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"]
a_b_four = ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"]
a_b_all = ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
            "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B", "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B",
            "lBicepGirth A_B", "rAnkle A_B", "rWristGirth A_B", "Lankle A_B"]
scaler_config = {
    "SEX": LabelBinarizer,
    "age": MinMaxScaler,
    "default": StandardScaler
}
features = m_all + a_b_all + age + volumes + bmi + sex

timestamp = timestamp()
for target in targets:
    for body_part in body_parts:
        if target == "FAT":
            target_cname = body_part.dexa.fat_mass
        elif target == "LEAN":
            target_cname = body_part.dexa.lean_mass
        elif target == "PFAT":
            target_cname = body_part.dexa.pmass
        elif target == "BMC":
            target_cname = body_part.dexa.bmc

        print(f'[STATUS] Training and saving {type(model).__name__} on {target_cname}')
        ext_train_save(model, dataset, features, target_cname, scaler_config=scaler_config, eval_type='regressor', timestamp=timestamp)