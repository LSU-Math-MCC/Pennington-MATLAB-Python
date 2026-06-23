import pandas as pd, numpy as np
from DataSets import StykuDataSet, to_DataSet
from utilities.common_functions import append_dict
from utilities.PathMaker import PathMan
from utilities.data_transformers import standardize_subject_ids, cut_subject_ids

Path = PathMan()
GITPATH = Path.getter()

bmi = ['BMI1']
age = ['age']
sex = ['SEX']
volumes = ["TotalVolume", #"headVolume",
           "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
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
features = m_all + a_b_all + age + volumes + bmi + sex


# cname_dict = {
#     'SubjectID': 'SubjectID',
#     'SEX': 'DEM_SEX',
#     'age': 'DEM_AGE',
#     'Race': 'DEM_RACE',
#
#     'GLU': 'Blood_GLU',
#     '_HBA1C': 'Blood_HBA1C',
#     'LDL': 'Blood_LDL',
#     'HDL': 'Blood_HDL',
#     'TRIG': 'Blood_TRI',
#     'INS': 'Blood_INS',
#     'CHOL': 'Blood_CHL',
#     'HB': 'Blood_BP',
#
#     'WBTOT_FAT': 'BC_DXA_FAT_TOT',
#     'TRUNK_FAT': 'BC_DXA_FAT_Trunk',
#     'RARM_FAT': 'BC_DXA_FAT_Arm_R',
#     'LARM_FAT': 'BC_DXA_FAT_Arm_L',
#     'R_LEG_FAT': 'BC_DXA_FAT_Leg_R',
#     'L_LEG_FAT': 'BC_DXA_FAT_Leg_L',
#
#     'TOTAL_PFAT': 'BC_DXA_PFAT_TOT', # WBOTOT?
#     'TRUNK_PFAT': 'BC_DXA_PFAT_Trunk',
#     'RARM_PFAT': 'BC_DXA_PFAT_Arm_R',
#     'LARM_PFAT': 'BC_DXA_PFAT_Arm_L',
#     'R_LEG_PFAT': 'BC_DXA_PFAT_Leg_R',
#     'L_LEG_PFAT': 'BC_DXA_PFAT_Leg_L',
#
#     'WBTOT_LEAN': 'BC_DXA_LST_TOT',
#     'TRUNK_LEAN': 'BC_DXA_LST_Trunk',
#     'RARM_LEAN': 'BC_DXA_LST_Arm_R',
#     'LARM_LEAN': 'BC_DXA_LST_Arm_L',
#     'R_LEG_LEAN': 'BC_DXA_LST_Leg_R',
#     'L_LEG_LEAN': 'BC_DXA_LST_Leg_L',
#
#     'WBTOT_BMC': 'BC_DXA_BMC_TOT',
#     'TRUNK_BMC': 'BC_DXA_BMC_Trunk',
#     'RARM_BMC': 'BC_DXA_BMC_Arm_R',
#     'LARM_BMC': 'BC_DXA_BMC_Arm_L',
#     'RLEG_BMC': 'BC_DXA_BMC_Leg_R',
#     'LLEG_BMC': 'BC_DXA_BMC_Leg_L',
#
#     # Need to valiBCte
#     'WBTOT_BMD': 'BC_DXA_BMD_TOT',
#     # 'TRUNK_BMD': 'BC_DXA_BMD_Trunk',
#     'RARM_BMD': 'BC_DXA_BMD_Arm_R',
#     'LARM_BMD': 'BC_DXA_BMD_Arm_L',
#     'RLEG_BMD': 'BC_DXA_BMD_Leg_R',
#     'LLEG_BMD': 'BC_DXA_BMD_Leg_L',
#
#     'TOTAL_VOLUME': 'VOL_DXA_TOT',
#     'TRUNK_VOLUME': 'VOL_DXA_Trunk',
#     'LEG_VOLUME': 'VOL_DXA_Leg',
#     # 'RARM_volume': 'VOL_DXA_Arm_R',
#     # 'LARM_volume': 'VOL_DXA_Arm_L',
#     # 'RLEG_volume': 'VOL_DXA_Leg_R',
#     # 'LLEG_volume': 'VOL_DXA_Leg_L',
#
#     # '': 'VOL_BODPOD_TOT',
#
#     # Entering our own column names
#     'rbicepGirth': 'DA_3DO3_CIRC_B_R',
#     'lBicepGirth': 'DA_3DO3_CIRC_B_L',
#     'rForearm': 'DA_3DO3_CIRC_F_R',
#     'lForearmGirth': 'DA_3DO3_CIRC_F_L',
#     'rWristGirth': 'DA_3DO3_CIRC_Wr_R',
#     'lWristGirth': 'DA_3DO3_CIRC_Wr_L',
#     'rThighGirth': 'DA_3DO3_CIRC_Th_R',
#     'lThighGirth': 'DA_3DO3_CIRC_Th_L',
#     'rCalfCirc': 'DA_3DO3_CIRC_C_R',
#     'lCalfCirc': 'DA_3DO3_CIRC_C_L',
#     'rAnkle': 'DA_3DO3_CIRC_A_R',
#     'Lankle': 'DA_3DO3_CIRC_A_L',
#     'Chest circ': 'DA_3DO3_CIRC_Ch',
#     'waist circ': 'DA_3DO3_CIRC_W',
#     'hip circ': 'DA_3DO3_CIRC_H',
#
#
#     'rbicepGirth A_B': 'DA_3DO3_ER_B_R',
#     'lBicepGirth A_B': 'DA_3DO3_ER_B_L',
#     'rForearm A_B': 'DA_3DO3_ER_F_R',
#     'lForearmGirth A_B': 'DA_3DO3_ER_F_L',
#     'rWristGirth A_B': 'DA_3DO3_ER_Wr_R',
#     'lWristGirth A_B': 'DA_3DO3_ER_Wr_L',
#     'rThighGirth A_B': 'DA_3DO3_ER_Th_R',
#     'lThighGirth A_B': 'DA_3DO3_ER_Th_L',
#     'rCalfCirc A_B': 'DA_3DO3_ER_C_R',
#     'lCalfCirc A_B': 'DA_3DO3_ER_C_L',
#     'rAnkle A_B': 'DA_3DO3_ER_A_R',
#     'Lankle A_B': 'DA_3DO3_ER_A_L',
#     'Chest circ A_B': 'DA_3DO3_ER_Ch',
#     'waist circ A_B': 'DA_3DO3_ER_W',
#     'hip circ A_B': 'DA_3DO3_ER_H',
#
#     'r_bicep_a_over_b': 'DA_3DO3_ER_B_R',
#     'l_bicep_a_over_b': 'DA_3DO3_ER_B_L',
#     'r_forearm_a_over_b': 'DA_3DO3_ER_F_R',
#     'l_forearm_a_over_b': 'DA_3DO3_ER_F_L',
#     'r_wrist_a_over_b': 'DA_3DO3_ER_Wr_R',
#     'l_wrist_a_over_b': 'DA_3DO3_ER_Wr_L',
#     'rThigh_a_over_b': 'DA_3DO3_ER_Th_R',
#     'lThigh_a_over_b': 'DA_3DO3_ER_Th_L',
#     'rCalf_a_over_b': 'DA_3DO3_ER_C_R',
#     'lCalf_a_over_b': 'DA_3DO3_ER_C_L',
#     'r_ankle_a_over_b': 'DA_3DO3_ER_A_R',
#     'l_ankle_a_over_b': 'DA_3DO3_ER_A_L',
#     'chest_a_over_b': 'DA_3DO3_ER_Ch',
#     'waist_a_over_b': 'DA_3DO3_ER_W',
#     'hip_a_over_b': 'DA_3DO3_ER_H',
#
#
#     'CollarScalp': 'DA_3DO3_LEN_Collar',
#     'crotchHeight': 'DA_3DO3_LEN_Crotch',
#     'TrunkLength': 'DA_3DO3_LEN_Trunk',
#
#
#     'RArmLength': 'DA_3DO3_LEN_Arm_R',
#     'LarmLength': 'DA_3DO3_LEN_Arm_L',
#     'rLegLength': 'DA_3DO3_LEN_Leg_R',
#     'lLegLength': 'DA_3DO3_LEN_Leg_L',
#
#
#     # 'SA_total': 'DA_3DO3_SA_TOT',
#     # 'SA_trunk': 'DA_3DO3_SA_Trunk',
#     # 'SA_rArm': 'DA_3DO3_SA_Arm_R',
#     # 'SA_lArm': 'DA_3DO3_SA_Arm_L',
#     # 'SA_rLeg': 'DA_3DO3_SA_Leg_R',
#     # 'SA_lLeg': 'DA_3DO3_SA_Leg_L',
#
#
#     'TotalVolume': 'VOL_3DO3_TOT',
#     'trunkVolume': 'VOL_3DO3_Trunk',
#     'rArmVolume': 'VOL_3DO3_Arm_R',
#     'lArmVolume': 'VOL_3DO3_Arm_L',
#     'rLegVolume': 'VOL_3DO3_Leg_R',
#     'lLegVolume': 'VOL_3DO3_Leg_L',
#
#
#     'BMI1': 'CA_BMI',
#     'HeightCMAvg': 'CA_Height',
#     'Weight1KG': 'CA_Weight',
#
#
#     'ArmR1': 'CA_CIRC_B-1Samp',
#     'ArmR2': 'CA_CIRC_B-2Samp',
#     'ArmR3': 'CA_CIRC_B-3Samp',
#     'ThighR1': 'CA_CIRC_Th-1Samp',
#     'ThighR2': 'CA_CIRC_Th-2Samp',
#     'ThighR3': 'CA_CIRC_Th-3Samp',
#     'Waist1': 'CA_CIRC_W-1Samp',
#     'Waist2': 'CA_CIRC_W-2Samp',
#     'Waist3': 'CA_CIRC_W-3Samp',
#     'Hip1': 'CA_CIRC_H-1Samp',
#     'Hip2': 'CA_CIRC_H-2Samp',
#     'Hip3': 'CA_CIRC_H-3Samp',
#     # # '': 'CA_CIRC_B_L',
#     # '': 'CA_CIRC_F_R',
#     # '': 'CA_CIRC_F_L',
#     # '': 'CA_CIRC_Wr_R',
#     # '': 'CA_CIRC_Wr_L',
#     # '': 'CA_CIRC_Th_R',
#     # '': 'CA_CIRC_Th_L',
#     # '': 'CA_CIRC_C_R',
#     # '': 'CA_CIRC_C_L',
#     # '': 'CA_CIRC_A_R',
#     # '': 'CA_CIRC_A_L',
#     # '': 'CA_CIRC_Ch',
#     #
#     # '': 'CA_LEN_Arm', # have bicep len?
#     # '': 'CA_LEN_Leg', # have thigh len?
# }

# print('Volumes:')
# print(f'{[cname_dict[vol] for vol in volumes]}')
# print('m_common:')
# print(f'{[cname_dict[vol] for vol in m_common]}')
# print('m_all:')
# print(f'{[cname_dict[vol] for vol in m_all]}')
# print('a_b_four:')
# print(f'{[cname_dict[vol] for vol in a_b_four]}')
# print('a_b_all:')
# print(f'{[cname_dict[vol] for vol in a_b_all]}')

# df = pd.read_excel(GITPATH + 'python/data/ShapeUp/ObjOrganizerStyku_v14.xlsx')
# df = df.drop(df.columns[[0, 1, 3, 4]], axis=1)  # remove unnecessary label columns
# df.rename(columns={col: f"{col.replace('/', '_')}" for col in df.columns}, inplace=True)  # stardardize styku column names
# StykuDataset = to_DataSet(df, combine_common=True, subject_cname='Name')
# df = StykuDataset.df
# # print(StykuDataset.df)  # print the full DataSet as a pandas DataFrame
# df.rename(columns=cname_dict, inplace=True)
# # df.dropna(subset=['DA_3DO3_LEN_Crotch'])
# df = df[cname_dict.values()]
#
# left_cnames = [cname[:-2] for cname in cname_dict.values() if cname.endswith('_L')]
# for cname in left_cnames:
#     cname_L, cname_R = cname + '_L', cname + '_R'
#     df[cname] = df[[cname_L, cname_R]].mean(axis=1)

# sampled_cnames = [cname[:-6] for cname in cname_dict.values() if cname.endswith('-3Samp')]
# for cname in sampled_cnames:
#     cname_1, cname_2, cname_3 = cname + '-1Samp', cname + '-2Samp', cname + '-3Samp'
#     df[cname] = df[[cname_1, cname_2, cname_3]].mean(axis=1)
#     for bad_cname in [cname_1, cname_2, cname_3]:
#         del df[bad_cname]

# df.to_excel(GITPATH + 'python/data/ShapeUp/AllStyku_v14.xlsx')
# print(df)

'''
Styku v14 (from DataStorage)
'''
# file_in = 'DataStorage_ShapeUpAdults_v4.xlsx'
# file_out = 'pdDataStorage_v4out.xlsx'
#
# df_A = pd.read_excel(GITPATH + 'python/data/ShapeUp/' + file_in, sheet_name='DA_VOL_3DO_ScanA')
# df_B = pd.read_excel(GITPATH + 'python/data/ShapeUp/' + file_in, sheet_name='DA_VOL_3DO_ScanB')
# print('here')
# df = pd.concat([df_A, df_B], sort=False)
# df = df.groupby(['Subject ID'], as_index=False).aggregate('mean')
# df_Common = pd.read_excel(GITPATH + 'python/data/ShapeUp/' + file_in, sheet_name='DEM_BLOOD_BC_CA')
# df = df.merge(df_Common, on='Subject ID', how='outer')
# df = df.rename(columns={'Subject ID': 'SubjectID'})
# # Combine L-R
# left_cnames = [cname[:-2] for cname in cname_dict.values() if cname.endswith('_L')]
# for cname in left_cnames:
#     cname_L, cname_R = cname + '_L', cname + '_R'
#     df[cname] = df[[cname_L, cname_R]].mean(axis=1)
# df.to_excel(GITPATH + 'python/data/ShapeUp/' + file_out, index=False)
# print(df)

'''
Styku v6 (remake)
Uncomment surface areas above
'''
# df = pd.read_excel(GITPATH + 'python/data/ShapeUp/v6/ObjOrganizerStyku_v6.xlsx', na_values=["[]", 0])
# df = df.drop(df.columns[[0, 1, 3, 4]], axis=1)  # remove unnecessary label columns
# df['SubjectID'] = cut_subject_ids(df['Name'])
# df = df.groupby(['SubjectID'], as_index=False).aggregate('mean')
# df_ER = pd.read_csv(GITPATH + 'python/data/ShapeUp/v6/Styku_a_over_b.csv')
# df = df.merge(df_ER, on='SubjectID', how='outer')
# # stardardize styku column names
# df.rename(columns={col: f"{col.replace('/', '_')}" for col in df.columns}, inplace=True)
# StykuDataset = to_DataSet(df, combine_common=True, subject_cname='SubjectID')
# df = StykuDataset.df
# df.rename(columns=cname_dict, inplace=True) # Rename AFTER combine common
# # Combine L-R
# left_cnames = [cname[:-2] for cname in cname_dict.values() if cname.endswith('_L')]
# for cname in left_cnames:
#     cname_L, cname_R = cname + '_L', cname + '_R'
#     df[cname] = df[[cname_L, cname_R]].mean(axis=1)
# df.to_excel(GITPATH + 'python/data/ShapeUp/AllStyku_v6.xlsx', index=False)


'''
NHANES 21SEPT2015
'''
# file_in = 'NHANES 21SEPT2015.xlsx'
# file_out = 'pd_NHANES_21SEPT2015.xlsx'
#
# combined_df = pd.DataFrame()
# for sheet_name in ['DXA_Imp_1','DXA_Imp_2','DXA_Imp_3','DXA_Imp_4','DXA_Imp_5']:
#     combined_df = pd.concat([
#             combined_df,
#             pd.read_excel(GITPATH + 'python/data/NHANES/' + file_in, sheet_name=sheet_name)
#          ], sort=False)
# df = combined_df.groupby(['SEQN'], as_index=False).aggregate('mean')
# df.to_excel(GITPATH + 'python/data/NHANES/' + file_out, index=False)
#
# # df_A = pd.read_excel(GITPATH + 'python/data/NHANES/' + file_in, sheet_name='DXA_Imp_1')
# # df_B = pd.read_excel(GITPATH + 'python/data/NHANES/' + file_in, sheet_name='DXA_Imp_2')
# # df_C = pd.read_excel(GITPATH + 'python/data/NHANES/' + file_in, sheet_name='DXA_Imp_3')
# # df_D = pd.read_excel(GITPATH + 'python/data/NHANES/' + file_in, sheet_name='DXA_Imp_4')
# # df_E = pd.read_excel(GITPATH + 'python/data/NHANES/' + file_in, sheet_name='DXA_Imp_5')
# # df = pd.concat([df_A, df_B, df_C, df_D, df_E], sort=False)
# # df = df.groupby(['SEQN'], as_index=False).aggregate('mean')
# # df.to_excel(GITPATH + 'python/data/NHANES/' + file_out, index=False)



df = pd.read_excel(GITPATH + 'python/data/NHANES/pd_NHANES_21SEPT2015.xlsx')
df = df.loc[df['RIDEXPRG']==2]
cnames = ['BMXCALF', 'bmxht', 'bmxbmi']
df.dropna(axis=0, subset=cnames, inplace=True)
print(len(df.loc[(18 <= df['RIDAGEYR']) & (df['RIDAGEYR'] <= 90)]))
# print(np.quantile(df[cname].to_numpy(), .5))
age_groups = {**{"18 to 29": [18,29]},
              **{str(n) + " to " + str(n+9): [n,n+9] for n in range(30,61,10)},
              **{"70 to 90": [70,90]}}
race_names = ['White', 'Black', 'Hispanic', 'Other']  # list ordered by NHANES conventions (1: White, 2: Black...)
tb1 = pd.DataFrame()
for group_name, group_range in age_groups.items():
    # slice df for age
    df0 = df.loc[(group_range[0] <= df['RIDAGEYR']) & (df['RIDAGEYR'] <= group_range[1])]
    for sex in [1,2]:
        # slice df for sex
        df1 = df0.loc[df0['RIAGENDR'] == sex]
        # create dictionary for each row of table
        row_dict = dict()
        row_dict['Age'] = group_name
        row_dict['Sex'] = 'Male' if sex == 1 else 'Female'
        # for similar table using quantiles
        # for quant in [.25,.5,.75]:
        #     row_dict[quant] = np.quantile(df1[cname].to_numpy(), quant)
        for n in range(1, len(race_names)+1):
            row_dict[race_names[n-1]] = len(df1.loc[df0['RACETH'] == n])
        # add row to final table
        tb1 = append_dict(tb1, row_dict)
tb1.to_excel(GITPATH + 'python/data/NHANES/tb1.xlsx', index=False)
print(tb1)
print(tb1.sum()[2::].sum())