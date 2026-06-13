from python.utilities.PathMaker import PathMan
from collections import OrderedDict
import pandas as pd
import numpy as np
from python.utilities.data_transformers import standardize_subject_ids


path=PathMan()
filepath = path.getter()+ 'python\\data\\bad_scans.txt'
styku=  pd.read_excel(r'C:\Users\Pujan Shrestha\Downloads\ObjOrganizerStyku_v12(1).xlsx', na_values=["[]", 0])


class Scan_Removal:
    #pulls the subject ID as keys and bodyparts as data for a dictionary from the bad scans file.
    def obtain_files(self, filepath):
        bad_scan = OrderedDict()
        with open(filepath, 'r') as file:
            lines = file.readlines()

        for row in lines:
            ssplit = row.split()
            key = ssplit.pop(0)
            key=self.fix_subject_id(key)
            bad_scan[key] = ssplit
        return bad_scan

    #where actual work takes place
    def remove_bad_scans(self, filepath , Dataset):
        bad_scan = self.obtain_files(filepath)
        Dataset['Name'] = Dataset['Name'].map(lambda names: "02ADL0" + names[0:3] + "_A" if names[0] != "0" else names)
        #we create a new subject id column in the dataframe to compare it with the subject ids from the bad scans list
        Dataset['SubjectID'] = standardize_subject_ids(Dataset['Name'])

        #Setting the subject ID column as the idx for easier navigation of the data
        Dataset.set_index(Dataset.SubjectID, inplace=True, verify_integrity=True)

        #print(Dataset)
        for idx, body_parts in bad_scan.items():
            if body_parts == []:
                # place to drop subject ids if we want to drop the whole row of values
                Dataset = Dataset[Dataset.index != idx]

            else:
                #print(idx, ":", body_parts)
                # Needs a method to give body parts and return column names to drop
                cnames_to_drop = self.return_column_names(body_parts)
                #Dataset.loc[idx, cnames_to_drop] = ""
                for cnames in cnames_to_drop:
                    #print (idx, cnames)
                    Dataset.at[idx , cnames]= np.nan
        #after removing the bad scans, we reset the index and remove the subject ID column
        Dataset.index = range(len(Dataset.index))
        Dataset = Dataset.drop(columns="SubjectID")
        return Dataset

    #this fixes the subject ids that come from the bad scans file path. When the names/subject id's are standarized by Marceline, we might need to rework this
    def fix_subject_id(self,subject_id):
        length = len(subject_id)
        if subject_id[0] != "0":
            subject_id = "02ADL0" + subject_id
            if subject_id[9] =="A":
                subject_id=subject_id[0:9]
            elif subject_id[9] =="B":
                subject_id = subject_id[0:9]+"_2"
        elif length == 11:
            if subject_id[10] == "A":
                subject_id = subject_id[0:9]
            elif subject_id[10] == "B":
                subject_id = subject_id[0:9] + "_2"
            return subject_id
        elif length > 11:
            if subject_id[15] == "1":
                subject_id = subject_id[0:9]
            elif subject_id[15] == "2":
                subject_id = subject_id[0:9] + "_2"
        return subject_id

    #anytime there is a new body_part, add it into the return column names list. This method looks at all the body parts and returns a list of columns names those parts are related to
    def return_column_names(self,body_parts):
        cnames=[]

        for bodypart in body_parts:
            addum = []
            if bodypart=="Right_arm":
                addum=["rArmVolume","trunkVolume","rbicepGirth A/B","rForearm A/B", "rWristGirth A/B" ,"RArmLength", "rbicepGirth" , "rForearm","rWristGirth"]
            elif bodypart=="Left_arm":
                addum=["lArmVolume","trunkVolume","lBicepGirth A/B","lForearmGirth A/B", "lWristGirth A/B" ,"LarmLength", "lBicepGirth" , "lForearmGirth","lWristGirth"]
            elif bodypart == "Head":
                addum = ["headVolume", "trunkVolume", "CollarScalp"]
            elif bodypart=="Left_leg":
                addum = ["lLegLength", "lCalfCirc", "Lankle", "lThighGirth","lCalfCirc A/B", "Lankle A/B", "lThighGirth A/B","lLegVolume","trunkVolume"]
            elif bodypart == "Right_leg":
                addum = ["rLegLength", "rCalfCirc", "Rankle", "rThighGirth", "rCalfCirc A/B", "Rankle A/B",
                         "rThighGirth A/B", "rLegVolume", "trunkVolume"]
            elif bodypart == "Trunk":
                addum = ["Chest circ", "waist circ", "hip circ" , "crotchHeight" , "TrunkLength", "Chest circ A/B", "hip circ A/B", "trunkVolume"]
            elif bodypart == "Right_ankle":
                addum = ["Rankle A/B", "Rankle" ]
            elif bodypart == "Left_ankle":
                addum = ["Lankle","Lankle A/B"]

            for add in addum:
                if add not in cnames:
                    cnames.append(add)

        return cnames

def __main__():
    Remove = Scan_Removal()
    df = Remove.remove_bad_scans(filepath,styku)
    #df.to_excel("D:/ML/python/data/Experiment_2.xlsx")
if __name__ == "__main__":
    __main__()