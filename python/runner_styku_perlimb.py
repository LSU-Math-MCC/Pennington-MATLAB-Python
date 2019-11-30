import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge, LassoLars
from sklearn.preprocessing import StandardScaler, LabelBinarizer,MinMaxScaler
from common import Map
from datasets import DataSet,StykuDataSet_2
from utilities.paramutils import combine_options
from runner import execute
from utilities.data_transformers import standardize_subject_ids, column_filter, mean_body_part_transformer, average_transformer
from utilities.data_transformers import discrete_class, standardize_subject_ids
from utilities.folder_searcher import Searcher
from utilities.DirectoryGrab import DirGrab
from utilities.data_merger import ListMaker, ListStandardizer, CBDrowMaker, MergeMan
import time




# Maybe move the standardization functions to utilities.data_transformers
def standardize_units_3(df, body_parts):
    for bp in body_parts:
        df[bp.styku.volume] = df[bp.styku.volume] * (2.54 ** 3)  # in3 to cm3
        df[bp.styku.volume] = df[bp.styku.volume] / 1000  # cm3 to L
        df[bp.dexa.volume] = df[bp.dexa.volume] / 1000  # cm3 to L
        #df[bp.ss20.volume] = df[bp.ss20.volume] / 1000000  # mm3 to L
        #df[dexa_total_volume] = df[dexa_total_volume] / 1000
    return df
#Maybe move it into MDF
def standardize_units_2(df):
    for col_name in df.columns:
        if 'Volume' in col_name:
            df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
        elif 'A/B' in col_name:
            df[col_name] = df[col_name]
        else:
            df[col_name] = df[col_name].map(lambda x: x * 2.54 if type(x) is float else x)
    return df

#keeping sources here for now in order to manage the input files while we amend the dataset entry

class ModifiedDataSet(DataSet):


    def load_data(self):
        asdf = StykuDataSet_2()
        Combo_df = asdf.common_dataframes_2()

        Styku_df = StykuDataSet_2.search_update()  # loads DataFrame
        Styku_df = standardize_units_2(Styku_df)
        Styku_df['SubjectID'] = ListMaker(Styku_df.Name)  # creates Subject ID column from Name column
        Styku_df = Styku_df.drop(Styku_df.columns[[0, 1, 2, 3, 4]], axis=1)
        Styku_df.rename(columns={col: f"Styku_{col.replace(' ', '_')}" for col in Styku_df.columns},
                        inplace=True)  # Rename columns with dataset
        Styku_df.rename(columns={Styku_df.columns[-1]: "SubjectID"}, inplace=True)
        modData = CBDrowMaker(Styku_df)  # creates modified DataFrame
        modData['SubjectID'] = ListStandardizer(modData.SubjectID)  # standardizes subject ID column

        modData_2 = CBDrowMaker(Combo_df)  # creates a modified Dataframe
        modData_2['SubjectID'] = ListStandardizer(modData_2.SubjectID)  #

        combined_df = MergeMan(modData, modData_2)

        DXA_Q2 = pd.read_excel("data/DXA_Q2_plusRegional_brief.xlsx")
        DXA_Q2 = DXA_Q2.rename(columns={"Subject ID": "SubjectID"})  # Remove space in column

        modData_3 = CBDrowMaker(DXA_Q2)
        modData_3['SubjectID'] = ListStandardizer(modData_3.SubjectID)
        combined_df = MergeMan(combined_df, modData_3)
        return standardize_units_3(combined_df, body_parts)


def data_callback(data_list):
    pass


def perlimbstring(old_list):
    string = 'Styku_'
    my_new_list = [string + x for x in old_list]
    #string2 = 'SS20_'
    #my_new_list2 = [string2 + x for x in old_list]
    return my_new_list
    #return (my_new_list + my_new_list2)


#I'm using this to measure the run time of this program
start = time.time()

dftest = pd.DataFrame()

#This is implementing the dot notation for the different parts. For example. left_arm.dexa.lean_mass refers to the LARM_LEAN column form the dataset (the dexa dataset appended into to the combined Shapeup())
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

#defining the different body parts
body_parts = [left_arm, right_arm, left_leg, right_leg, head, trunk]



#assigning the features [body part name, measurements for the body part, a/b values , Volume ]. These are the names of the column in the Dataset
body_part_features = (
    (left_arm,   ["LarmLength", "lForearmGirth", "lBicepGirth", "lWristGirth"], ["lForearmGirth A_B", "lBicepGirth A_B", "lWristGirth A_B"],["lArmVolume"]),
    (right_arm,  ["RArmLength", "rForearm", "rbicepGirth", "rWristGirth"], ["rForearm A_B", "rbicepGirth A_B", "rWristGirth A_B"],["rArmVolume"]),
    (head,   ["CollarScalp"], [],["headVolume"]), #currently missing a_b_values and other measurements
    (left_leg,   ["lLegLength", "lCalfCirc", "Lankle", "lThighGirth"], ["lCalfCirc A_B", "Lankle A_B", "lThighGirth A_B"],["lLegVolume"]),
    (right_leg,  ["rLegLength", "rCalfCirc", "rAnkle", "rThighGirth"], ["rCalfCirc A_B", "rAnkle A_B", "rThighGirth A_B"],["rLegVolume"]),
    (trunk , ["Chest_circ", "waist_circ", "hip_circ","crotchHeight","TrunkLength"],["Chest circ A_B", "hip circ A_B", "hip_a_over_b"],["trunkVolume"])
)

MDF = ModifiedDataSet()
essential_transformers = [mean_body_part_transformer(False)] + [average_transformer(cname, cname + "\\d+") for cname in ["ArmR", "ThighR", "Waist", "Hip"]]

loader_params = dict(
    data_transformers=essential_transformers,
    scaler_config={"SEX": LabelBinarizer,
                    "age": MinMaxScaler,
                   "default": StandardScaler},
)

loader_param_grid = dict(
    data_transformers={
        "__cols": "SEX",
        #"M": essential_transformers + [column_filter("SEX", "M")],
        #"F": essential_transformers + [column_filter("SEX", "F")],
        "M/F": essential_transformers,
    },
)
#these are just different regressors for our prediction system. Erin and Claire have an extensive research for the optimal hyper-parameters

#regressor = MLPRegressor(max_iter=900,
#                       solver="lbfgs",
#                       batch_size=8,
#                      activation="identity",
#                     hidden_layer_sizes=tuple())

#regressor = Ridge(alpha=1.9, max_iter=None, tol=0.001, solver="auto")
regressor_parameter_grid = dict()

save_name = "mfking_report"

targets = ["PFAT","FAT", "LEAN"] #"FAT", "LEAN","PFAT" names of the targets to work on. The conditional in the next for loop assigns which label we are using for each body part. The dot notation allows for easy assignments.

#for example, when the body part is left arm, the label cname is LARM_{whichever target we use}
for target in targets:

    dftest3 = pd.DataFrame()
    dftest3["newid_ABMV"]=""
    print(f"[STATUS] Running regressor on {target}")
    for body_part, measurement_cnames, ab_cnames, vol_cnames in body_part_features:
        dftest2 = pd.DataFrame()
        i = 0
        # loader_param_grid["label_cname"] = [label_cname]
        #print(f"[STATUS] Running regressor on {target}")
        #might need to find a better conditional on this for all the different target types

        if target == "FAT":
            label_cname = body_part.dexa.fat_mass
        elif target == "LEAN":
            label_cname = body_part.dexa.lean_mass
        elif target == "PFAT":
            label_cname = body_part.dexa.pmass
        elif target == "BMC":
            label_cname = body_part.dexa.bmc
        loader_params['label_cname'] = label_cname

        #comment out the loader params for different specifications, As of now, its all on.
        loader_param_grid["feature_cnames"] = combine_options({
            "bmi": {
                "N": [],
                "Y": ["BMI1"]
            },
            "a_b": {
                "N": [],
                "Y": ab_cnames,
            },
            "volume": {
                "N": [],
                "Y": perlimbstring(vol_cnames),
            },
            "measurements": {
                "Y": perlimbstring(measurement_cnames),
                "N": [],
            }
        }, additional_options=["SEX"])

        results = execute(MDF,
                      loader_params,
                      loader_param_grid,
                          [KernelRidge(alpha=3.00)],
                      regressor_parameter_grid,
                      save_name=f"{save_name}-{body_part.short_name}-{label_cname}",
                      save_df_report=False,
                      extract_data_callback=data_callback,
                      include_regressor_and_mlpdata=False
                      )
        results.insert(1, 'Body_part',  body_part.short_name)
        results.insert(1, 'Regressor', target)
        dftest = pd.concat([dftest,results])

        '''
        if subjectwisereview==True:
            use = list(loader_param_grid["feature_cnames"].columns)
            use.remove("__data__")
            # Subject-wise comparison of test vs predicted values
            for idx, row in results.iterrows():
                best_regressor = row["best_regressor"]
                mlp_data = row["mlp_data"]
                # print("$types:")
                # print(type(best_regressor))
                # print(type(mlp_data))
                # print(mlp_data.y)
                predicted_df = mlp_data.predict(best_regressor).rename(columns={label_cname: "prediction_"+label_cname})
                actual_df = mlp_data.y.rename(columns={label_cname: "actual_"+label_cname})
                df = actual_df.merge(predicted_df, left_index=True, right_index=True)
                #df["body_part"] = body_part.short_name
                #df["Target"] = label_cname
                df["SubID"] = df.index
                for feature_cname in loader_param_grid["feature_cnames"].columns:
                    if feature_cname != "__data__":
                        df[feature_cname] = row[feature_cname]
                df['newid_ABMV'] = df.apply(
                    lambda x: str(x['SubID']) + x['a_b'] + x['bmi'] + x['measurements'] + x['volume'], axis=1)
                df = df.drop(columns=["SubID", "a_b", 'bmi', 'measurements', 'volume'])
                #print(df)
                dftest2 = pd.concat([dftest2, df], sort=False)
            print (dftest2)
            if i==0:
                dftest3 = dftest2.merge(dftest3, on='newid_ABMV', how='outer' )
                i=i+1
            elif i!=0:
                dftest3 = dftest2.merge(dftest3, on='newid_ABMV', how='inner')
        '''
#print(dftest3)
#dftest3.to_excel("data/Styku_perlimb_regional_PFAT.xlsx")




#this one is for predicted /vs actual data and lists it for each subject. FOR THIS TO RUN, MUST ENABLE  include_regressor_and_mlpdata in Execute to be true.
# save_df(results, f"{save_name}-{body_part.short_name}")
print(dftest)

end = time.time()
times = (end - start)/60
print("Run time=%d mins" %times)

#df = MDF.load_data()
#df.to_excel("data/Alex'sExcelSheet.xlsx")
