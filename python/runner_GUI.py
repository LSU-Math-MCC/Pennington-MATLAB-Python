#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPRegressor

'''
Resources Used
    The Tkinter Grid Geometry Manager: https://effbot.org/tkinterbook/grid.htm
'''
from datetime import date
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge,LassoLars
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from datasets import StykuDataSet, SS20DataSet, CombinedDataSet, StykuDataSet_2, TrimmedStykuDataSet
from utilities.data_transformers import column_filter, mean_body_part_transformer, average_transformer, cut_subject_ids
from utilities.paramutils import combine_options
from utilities.PathMaker import PathMan
from runner import execute
import tkinter as tk
import tkinter.filedialog
from tkinter import messagebox
from tkinter.ttk import *
from PIL import Image, ImageTk
from datasets import DataSet
import pandas as pd
from python.datasets import MLPData, DataFrameScaler


Path = PathMan()
ico_path = Path.getter() + "python\data\\"

class MainApplication(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.master = master
        self.winfo_toplevel().title("Machine Learning Backend GUI")
        self.winfo_toplevel().iconbitmap(ico_path+"Face_scanning.ico")
        # Fill left-hand GUI
        left_master = tk.Frame(self.master)
        self.__draw_inout(left_master).grid(sticky='NW', row=0, column=0)

        # This textbox implementation needs revision
        left_master.grid_rowconfigure(1, weight=1)
        left_master.grid_columnconfigure(0, weight=1)
        self.__draw_textbox(left_master).grid(sticky=tk.N + tk.S + tk.W + tk.E, row=1, column=0)
        left_master.pack(side=tk.LEFT, anchor='nw', fill='both', expand=1, pady=10, padx=10)

        # Fill right-hand GUI
        self.right_master = tk.Frame(self.master)
        self.__draw_features(self.right_master).grid(row=0, column=0)
        self.__draw_targets(self.right_master).grid(row=1, column=0)
        self.__draw_regressors(self.right_master).grid(row=2, column=0)
        # Check the selection in 100 ms
        self.right_master.after(100, self.check_for_selection)

        # Fill Bottom frame
        bottom_frame = tk.Frame(self.master)
        bottom_frame.pack(side="bottom", fill="both")
        run = Image.open(ico_path + "run.png")
        run = run.resize((129, 86))
        Run = ImageTk.PhotoImage(run)

        exit_im = Image.open(ico_path + "exit.png")
        exit_im = exit_im.resize((129, 86))
        Exit_im = ImageTk.PhotoImage(exit_im)

        self.run_button= tk.Button(bottom_frame, image=Run,  command=lambda: self.run_ml(self.text))
        self.run_button.image=Run
        self.run_button.grid( row=0, column=0, sticky='w')
        self.exit_button = tk.Button(bottom_frame, image = Exit_im,command=lambda: self.client_exit())
        self.exit_button.image = Exit_im
        self.exit_button.grid( row=0, column=1,  sticky='e')

        self.right_master.pack(side=tk.RIGHT, anchor='ne', fill='none', pady=10, padx=10)
    #whole dataset  class need rework for this to work
    class gui_dataset():
        def __init__(self,styku_path, dxa_path, blood_path, questionaire_path, manual_path):
            self.styku_path =styku_path
            self.blood_path = blood_path
            self.questionaire_path =questionaire_path
            self.dxa_path =dxa_path
            self.manual_path = manual_path
            self.df = self.load_data()

        def standardize_units(self, df):
            for col_name in df.columns:
                if 'Volume' in col_name:
                    df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
                elif 'A/B' in col_name:
                    df[col_name] = df[col_name]
                else:
                    df[col_name] = df[col_name].map(lambda x: x * 2.54 if isinstance(x, float) else x)
            return df

        def extract_data(self, feature_cnames, label_cname, scaler_config={}, data_transformers=[], blacklist_sids=[]):
            df = self.df.copy()
            df = df.set_index('SubjectID', verify_integrity=True)
            df = df[~df.index.isin(blacklist_sids)]
            feature_cnames = feature_cnames.copy()
            df = self.__transform(df, data_transformers, feature_cnames)
            # print(df[feature_cnames])
            df = self.__prune(df, feature_cnames + [label_cname])
            # print(feature_cnames + [label_cname])
            # print(df[feature_cnames])
            x = df[feature_cnames]
            y = df[[label_cname]]
            return MLPData(x, y, DataFrameScaler(scaler_config))

        def __transform(self, df, data_transformers, feature_cnames):
            for transformer in data_transformers:
                ret = transformer(df, feature_cnames)
                if isinstance(ret, pd.DataFrame):
                    df = ret
            return df

        def __prune(self, df, cnames_to_keep):
            # abort if columns are missing
            missing_cnames = [cname for cname in cnames_to_keep if cname not in df.columns.values]
            if len(missing_cnames) > 0:
                print(f"Missing columns: {missing_cnames}")
                exit()

            # remove columns not in use
            df = df[cnames_to_keep]
            # df.drop([cname for cname in df.columns if cname not in cnames_to_keep],
            #        axis='columns', inplace=True)

            # remove rows with empty values
            df = df.replace([0, "", "nan"], None).dropna()
            # cnames_to_drop = []
            # for index, row in df.iterrows():
            #   empty = {}
            #   for cname, cvalue in row.iteritems():
            #       cvalue = row[cname]
            #       if cvalue == 0 or cvalue == "" or cvalue == "nan" or pd.isnull(cvalue):
            #           empty[cname] = cvalue
            #   if len(empty) > 0:
            #       #print(f"dropping SubjectID '{index}' due to empty values: {empty}")
            #       cnames_to_drop.append(index)
            # df.drop(cnames_to_drop, inplace=True)
            return df

        def load_data(self):
            # proposed method
            styku_df = pd.read_excel(self.styku_path, na_values=["[]", 0])
            # added this to account for the names with "205A.obj" in the styku datasheet. Please remove if that gets fixed or if we get ones with "xxxB.obj"
            styku_df['Name'] = styku_df['Name'].map(
                lambda names: "02ADL0" + names[0:3] + "_A" if names[0] != "0" else names)
            styku_df['SubjectID'] = cut_subject_ids(styku_df['Name'])
            styku_df = styku_df.drop(styku_df.columns[[0, 1, 2, 3, 4]], axis=1)
            styku_df = styku_df.groupby(styku_df['SubjectID'], as_index=False).aggregate('mean')
            styku_df = self.standardize_units(styku_df)
            styku_df.rename(columns={col: f"{col.replace('/', '_')}" for col in styku_df.columns},
                            inplace=True)

            questionnaire_df = pd.read_csv(self.questionaire_path)
            questionnaire_df['SubjectID'] = cut_subject_ids(questionnaire_df['SubjectID'])

            dexa_df = pd.read_excel(self.dxa_path, na_values=['#N/A'])
            dexa_df = dexa_df.dropna(axis=0, subset=['WBTOT_FAT', 'WBTOT_LEAN', 'WBTOT_PFAT'])
            dexa_df['TRUNK_BMC'] = dexa_df.apply(
                lambda x: x['LRIB_BMC'] + x['RRIB_BMC'] + x['T_S_BMC'] + x['L_S_BMC'] + x['PELV_BMC'], axis=1)
            dexa_df['TOTAL_PLEAN'] = dexa_df.apply(lambda x: x['TOTAL_LEAN'] / x['TOTAL_MASS'], axis=1)
            dexa_df['SubjectID'] = cut_subject_ids(dexa_df['SubjectID'])

            blood_df = pd.read_csv(self.blood_path)
            blood_df['SubjectID'] = cut_subject_ids(blood_df['SubjectID'])

            manual_df = pd.read_csv(self.manual_path)
            manual_df['SubjectID'] = cut_subject_ids(manual_df['SubjectID'])

            dfs = [questionnaire_df, dexa_df,manual_df, blood_df]
            combined_df = None
            for df in dfs:
                df.drop_duplicates(subset='SubjectID', keep='last', inplace=True)
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = combined_df.merge(df, on='SubjectID', how='outer')

            combined_df = combined_df.merge(styku_df, on='SubjectID', how='outer')
            combined_df['age'] = combined_df['BIRTHDATE'].astype(str).map(
                lambda row: date.today().year - int(row[2:4]) - 1900 if row != 'NaT' else "")

            combined_df['age'] = combined_df.apply(lambda row: date.today().year - row['qff_dob_year'] if row['age'] == "" else row['age'], axis=1)
            combined_df['BMI_act'] = combined_df.apply(
                lambda row: row['bmi_calculated'] if row['BMI1'] == 0 or row['BMI1'] == "" or row['BMI1'] == "nan" or pd.isnull(row['BMI1']) else row['BMI1'], axis=1)
            combined_df['BMI_act'] = pd.to_numeric(combined_df['BMI_act'], errors='ignore')

            combined_df['BMI'] = combined_df.apply(
                lambda row: row['BMI_y'] if row['BMI_act'] == 0 or row['BMI_act'] == "" or row[
                    'BMI_act'] == "nan" or pd.isnull(row['BMI_act']) else row['BMI_act'], axis=1)
            combined_df['BMI'] = pd.to_numeric(combined_df['BMI'], errors='ignore')

            combined_df = combined_df.drop(columns="BMI_act")
            combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')
            return combined_df

    def __draw_inout(self, master):
        self.styku=	tk.StringVar()
        self.dxa= tk.StringVar()
        self.blood= tk.StringVar()
        self.output = tk.StringVar()
        self.questionaire = tk.StringVar()
        self.manual = tk.StringVar()
        folder = Image.open(ico_path + "folder.png")
        folder = folder.resize((22, 22))
        Folder = ImageTk.PhotoImage(folder)
        upload = Image.open(ico_path + "upload.png")
        upload = upload.resize((22,22))
        Upload = ImageTk.PhotoImage(upload)

        inout_frame = tk.Frame(master)
        tk.Label(inout_frame, font=("Helvetica", 17), text="INPUT:", fg="blue").grid(sticky='W', row=0, columnspan=2)
        tk.Label(inout_frame, font=("Helvetica", 10), text="Styku:").grid(sticky='W', row=1, column=0)
        self.Styku = tk.Entry(inout_frame, textvariable=self.styku)
        self.Styku.grid(row=1, column=1)
        tk.Label(inout_frame, font=("Helvectica", 10), text="Dexa:").grid(sticky='W', row=2, column=0)
        self.Dxa = tk.Entry(inout_frame, textvariable=self.dxa)
        self.Dxa.grid(row=2, column=1)
        tk.Label(inout_frame, font=("Helvectica", 10), text="Blood").grid(sticky='W', row=3, column=0)
        self.Blood = tk.Entry(inout_frame, textvariable=self.blood)
        self.Blood.grid(row=3, column=1)
        tk.Label(inout_frame, font=("Helvectica", 10), text="Questionaire:").grid(sticky='W', row=4, column=0)
        self.Questionaire = tk.Entry(inout_frame, textvariable=self.questionaire)
        self.Questionaire.grid(row=4, column=1)

        tk.Label(inout_frame, font=("Helvectica", 10), text="Manual:").grid(sticky='W', row=5, column=0)
        self.Manual = tk.Entry(inout_frame, textvariable=self.manual)
        self.Manual.grid(row=5, column=1)

        tk.Label(inout_frame, font=("Helvetica", 17), text="OUTPUT:", fg="blue").grid(sticky='W', row=0, column=3, columnspan=2)
        tk.Label(inout_frame, font=("Helvetica", 10), text="Output folder:").grid(sticky='W', row=1, column=3)
        self.output_folder = tk.Entry(inout_frame, textvariable=self.output)
        self.output_folder.grid(row=1, column=4)
        tk.Label(inout_frame, font=("Helvetica", 10), text="Output name:").grid(sticky='W', row=2, column=3)
        self.output_name = tk.Entry(inout_frame)
        self.output_name.grid(sticky='W', row=2, column=4)


        tk.Label(inout_frame, font=("Helvetica", 10), text="Save File? :").grid(sticky='W', row=3, column=3)
        self.save_file = tk.BooleanVar()
        self.save_file.set(1)  # checked by default
        tk.Checkbutton(inout_frame, variable=self.save_file).grid(sticky='W', row=3, column=4)


        self.Styku_button = tk.Button(inout_frame, image=Upload , height=22, width =22 , relief='groove',
                  command=lambda: self.get_file('Styku'))
        self.Styku_button.image= Upload
        self.Styku_button.grid(sticky='WE', row=1, column=2)

        self.DXA_button = tk.Button(inout_frame, image= Upload, height=22, width =22 , relief='groove',
                  command=lambda: self.get_file('DXA'))
        self.DXA_button.image = Upload
        self.DXA_button.grid(sticky='WE', row=2, column=2)

        self.Blood_button = tk.Button(inout_frame, image= Upload,height=22, width =22 , relief='groove',
                  command=lambda: self.get_file('Blood'))
        self.Blood_button.image = Upload
        self.Blood_button.grid(sticky='WE', row=3, column=2)

        self.Questionaire_button = tk.Button(inout_frame, image=Upload, height=22, width=22, relief='groove',
                                      command=lambda: self.get_file('Questionaire'))
        self.Questionaire_button.image = Upload
        self.Questionaire_button.grid(sticky='WE', row=4, column=2)

        self.Manual_button = tk.Button(inout_frame, image=Upload, height=22, width=22, relief='groove',
                                             command=lambda: self.get_file('Manual'))
        self.Manual_button.image = Upload
        self.Manual_button.grid(sticky='WE', row=5, column=2)

        self.Output_button = tk.Button(inout_frame, image = Folder, height=22, width =22 , relief='groove',
                  command=lambda: self.get_file('Output'))
        self.Output_button.image = Folder
        self.Output_button.grid(sticky='WE', row=1, column=5)
        return inout_frame

    def __draw_features(self, master):
        feature_frame = tk.Frame(master, relief = 'ridge', borderwidth= 1)
        tk.Label(feature_frame, font=("Helvetica", 17), text="Features:", fg="blue").grid(sticky='W', row=0, column=0,columnspan=3)
        for i in range(2):
            feature_frame.grid_columnconfigure(i, minsize=50)

        # BMI, Age, Volumes
        tk.Label(feature_frame, font=("Helvetica", 12), text="BMI:").grid(sticky='W',row=1, column=0)
        self.bmi = tk.BooleanVar()
        tk.Checkbutton(feature_frame, variable=self.bmi).grid(sticky='W', row=2, column=0)

        tk.Label(feature_frame, font=("Helvetica", 12), text="Age:").grid(sticky='W', row=1, column=1)
        self.age = tk.BooleanVar()
        tk.Checkbutton(feature_frame, variable=self.age).grid(sticky='W', row=2, column=1)

        tk.Label(feature_frame, font=("Helvetica", 12), text="Volume:").grid(sticky= 'W', row=1, column=2, columnspan=2)
        self.Volume = tk.BooleanVar()
        self.Volume.set(1) # checked by default
        tk.Checkbutton(feature_frame, variable=self.Volume).grid(sticky='W',row=2, column=2, columnspan=2)

        # Measurements
        tk.Label(feature_frame, font=("Helvetica", 12), text="Measurements:").grid(sticky='W', row=3, column=0)
        self.m_common = tk.BooleanVar()
        tk.Checkbutton(feature_frame, text='Common', variable=self.m_common).grid(sticky='W', row=4, column=0)
        self.m_all = tk.BooleanVar()
        tk.Checkbutton(feature_frame, text='All', variable=self.m_all).grid(sticky='W', row=5, column=0)

        # a_b
        tk.Label(feature_frame, font=("Helvetica", 12), text="A over B:").grid(sticky='W', row=3, column=1)
        self.a_b_four = tk.BooleanVar()
        tk.Checkbutton(feature_frame, text='Four', variable=self.a_b_four).grid(sticky='W', row=4, column=1)
        self.a_b_all = tk.BooleanVar()
        tk.Checkbutton(feature_frame, text='All', variable=self.a_b_all).grid(sticky='W', row=5, column=1)

        # Sex
        tk.Label(feature_frame, font=("Helvetica", 12), text="Sex:").grid(sticky='W', row=3, column=2, columnspan=2)
        self.sex_M = tk.BooleanVar()
        tk.Checkbutton(feature_frame, text='M', variable=self.sex_M).grid(sticky='W', row=4, column=2)
        self.sex_F = tk.BooleanVar()
        tk.Checkbutton(feature_frame, text='F', variable=self.sex_F).grid(sticky='W', row=5, column=2)
        self.sex_MF = tk.BooleanVar()
        tk.Checkbutton(feature_frame, text='M/F', variable=self.sex_MF).grid(sticky='W', row=5, column=3)
        return feature_frame

    def __draw_targets(self, master):
        feature_target = tk.Frame(master=master, relief = 'ridge', borderwidth= 1)
        tk.Label(feature_target, font=("Helvetica", 17), text="Targets:", fg="blue").grid( row=4, column=0, columnspan=3)
        for i in range(2):
            feature_target.grid_columnconfigure(i, minsize=10)

        self.TFAT = tk.BooleanVar()
        tk.Checkbutton(feature_target, text="Total Fat", variable=self.TFAT).grid(sticky='W', row=5, column=1)
        self.TLEAN = tk.BooleanVar()
        tk.Checkbutton(feature_target, text='Total Lean', variable=self.TLEAN).grid(sticky='W', row=5, column=3)
        self.PFAT = tk.BooleanVar()
        tk.Checkbutton(feature_target, text='Percent Fat', variable=self.PFAT).grid(sticky='W', row=6, column=1)
        self.PLEAN = tk.BooleanVar()
        tk.Checkbutton(feature_target, text='Percent Lean', variable=self.PLEAN).grid(sticky='W', row=6, column=3)

        return feature_target

    def __draw_regressors(self,master):
        feature_frame = tk.Frame(master=master, relief = 'ridge', borderwidth= 1)
        tk.Label(feature_frame, font=("Helvetica", 17), text="Regressor:", fg="blue").grid(sticky='W', row=0, column=0,columnspan=3)
        # Set up the Combobox
        self.selections = Combobox(feature_frame, state="readonly")
        self.selections['values'] = ['MLP', 'Ridge', 'Linear Regression', 'Kernal Ridge', 'Custom']
        self.selections.current(0)
        self.selections.grid(sticky='WE', row=1, column=0,columnspan=2)

        # The Entry to be shown if "Custom" is selected
        self.custom_field = tk.Entry(feature_frame)
        #self.custom_field.grid(sticky='WE', row=4, column=0,columnspan=2)
        self.show_custom_field = False


        return feature_frame

    def check_for_selection(self):
        value = self.selections.get()

        # If the value is equal to "Custom" and show_field is set to False
        if value == 'Custom' and not self.show_custom_field:

            # Set show_field to True and place the custom entry field
            self.show_custom_field = True

            # Create a new window how we did when we made self.root
            self.custom_field.grid(sticky='WE', row=4, column=0, columnspan=2)



        # If the value DOESNT equal "Custom"
        elif value != 'Custom':

            # Destroy the text box that was created if it exists
            if self.show_custom_field:
                self.custom_field.grid_remove()
            # Set show_field to False
            self.show_custom_field = False

        # Call this method again to keep checking the selection box
        self.master.after(100, self.check_for_selection)

    def __draw_textbox(self, master):
        txtbox_frame = tk.Frame(master=master)
        # ensure a consistent GUI size
        txtbox_frame.grid_propagate(False)
        # implement stretchability
        txtbox_frame.grid_rowconfigure(0, weight=1)
        txtbox_frame.grid_columnconfigure(0, weight=1)

        self.text = tk.Text(master=txtbox_frame, wrap=tk.NONE)
        self.text.grid(row=0, column=0, sticky="nsew")

        # create a Scrollbar and associate it with text
        scrolly = tk.Scrollbar(txtbox_frame, command=self.text.yview)
        scrolly.grid(row=0, column=1, sticky='nsew')
        self.text.config(yscrollcommand=scrolly.set)
        scrollx = tk.Scrollbar(txtbox_frame, command=self.text.xview, orient=tk.HORIZONTAL)
        scrollx.grid(row=2, column=0, sticky='nsew')
        self.text.config(xscrollcommand=scrollx.set)

        return txtbox_frame

    # method to define quitting the windows
    def client_exit(self):
        exit()

    def get_file(self,entry):
        if entry=="Styku":
            filename = tkinter.filedialog.askopenfilename(parent=root,title='Choose the Styku file', initialdir=ico_path)
            self.styku.set(filename)
            if filename != "":
                self.Styku['state'] = 'disabled'
        elif entry=="DXA":
            filename = tkinter.filedialog.askopenfilename(parent=root, title='Choose the DXA file', initialdir=ico_path)
            self.dxa.set(filename)
            if filename != "":
                self.Dxa['state'] = 'disabled'
        elif entry=="Blood":
            filename = tkinter.filedialog.askopenfilename(parent=root, title='Choose the Blood file', initialdir=ico_path)
            self.blood.set(filename)
            if filename != "":
                self.Blood['state'] = 'disabled'
        elif entry=="Questionaire":
            filename = tkinter.filedialog.askopenfilename(parent=root, title='Choose the Questionaire file', initialdir=ico_path)
            self.questionaire.set(filename)
            if filename != "":
                self.Questionaire['state'] = 'disabled'

        elif entry=="Manual":
            filename = tkinter.filedialog.askopenfilename(parent=root, title='Choose the Manual file', initialdir=ico_path)
            self.manual.set(filename)
            if filename != "":
                self.Manual['state'] = 'disabled'
        elif entry=="Output":
            filename = tkinter.filedialog.askdirectory (parent=root, title='Choose the output folder', initialdir=Path.getter())
            self.output.set(filename)
            if filename != "":
                self.output_folder['state'] = 'disabled'

    def run_ml(self, text):


        text.delete(1.0,tk.END)
        text.insert(tk.END, "[STATUS] Loading..............\n")
        text.tag_add("Status1", "1.0", "1.8")
        text.tag_config("Status1", background="white", foreground="green",font=('times new roman', 10, 'bold'))
        text.tag_add("start2", "1.9", "1.29")
        text.tag_config("start2", background="white", foreground="green")

        feature_grid = {"bmi": {"N": []}, "age": {"N": []}, "volumes": {"N": []}, "measurements": {"none": []},
                        "a_b": {"none": []}}
        if self.Volume.get():
            feature_grid['volumes']['Y'] = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
        if self.age.get():
            feature_grid['age']['Y'] = ['age']
        if self.bmi.get():
            feature_grid['bmi']['Y'] = ['BMI']
        if self.m_common.get():
            feature_grid['measurements']['common'] = ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"]
        if self.m_all.get():
            feature_grid['measurements']['all'] = ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                                                   "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                                                   "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                                                   "LarmLength", "RArmLength", "CollarScalp", "TrunkLength",
                                                   "lLegLength", "rLegLength", "crotchHeight"]
        if self.a_b_four.get():
            feature_grid['a_b']['Four'] = ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"]
        if self.a_b_all.get():
            feature_grid['a_b']['All'] = ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B",
                                               "lThighGirth A_B", "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B",
                                               "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B",
                                               "lBicepGirth A_B", "rAnkle A_B", "rWristGirth A_B", "Lankle A_B"]

        essential_transformers = [mean_body_part_transformer(False)] + \
                        [average_transformer(cname, cname + "\\d+") for cname in ["ArmR", "ThighR", "Waist", "Hip"]]
        loader_params = dict(
            data_transformers=essential_transformers,
            scaler_config={"SEX": LabelBinarizer,
                           "age": MinMaxScaler,
                           "default": StandardScaler
                           },
        )
        loader_param_grid = dict(
            data_transformers={
                "__cols": "SEX",
                "M/F": essential_transformers,
            },
            feature_cnames=combine_options(additional_options=["SEX"], input=feature_grid)
        )

        regressor_parameter_grid = dict()

        targets = []
        if self.TFAT.get():
            targets += ['WBTOT_FAT']
        if self.PFAT.get():
            targets += ['WBTOT_PFAT']
        if self.TLEAN.get():
            targets += ['WBTOT_LEAN']
        if self.PLEAN.get():
            targets += ['TOTAL_PLEAN']

        styku_path = self.Styku.get()
        blood_path = self.Blood.get()
        questionaire_path = self.Questionaire.get()
        dxa_path = self.Dxa.get()
        manual_path = self.Manual.get()

        if styku_path == "" or dxa_path == "":
            messagebox.showinfo("Dataset Error", "No Dataset Selected")
            text.insert(tk.END, "[STATUS] -------ERROR-------\n")
            text.tag_add("here", "2.0", "2.8")
            text.tag_config("here", background="white", foreground="red",font=('times new roman', 10, 'bold'))
            text.tag_add("start", "2.9", "2.29")
            text.tag_config("start", background="white", foreground="red")

        else:
            '''
            i think a good idea here is to combine blood, questionaire and manual into a list and pass it. according to the list we can choose to ignore imports for the ones without empty files.
            This edit might require changes in  HERE as well as in the GUI DATASET class. 
            '''
            dataset = self.gui_dataset(styku_path, dxa_path, blood_path, questionaire_path,  manual_path)
            value = self.selections.get()
            regressors= [MLPRegressor(solver="lbfgs",activation="identity",max_iter=800, hidden_layer_sizes=(1,), tol=0.001, alpha=0.01),  Ridge(alpha=1.2), LinearRegression(),  KernelRidge(alpha=3)]

        # space for code for getting the hyperparameters [P2]
            if value=="MLP":
                regressor=regressors[0]
            elif value == 'Ridge':
                regressor = regressors[1]
            elif value == 'Linear Regression':
                regressor = regressors[2]
            elif value == 'Kernal Ridge':
                regressor = regressors[3]
            elif value== 'Custom':
                regressor=eval(self.custom_field.get())

            save_name = f"GUITrials_{type(dataset).__name__}"
            results = execute(dataset,
                          loader_params,
                          loader_param_grid,
                          regressor,
                          regressor_parameter_grid,
                          targets=targets,
                          cv=4,
                          report=save_name
                          )

            text.insert(tk.END, results)
            text.see(tk.END)
            if self.save_file is True:
                results.to_excel(self.output_folder.get() + "/" + self.output_name.get()+'.xlsx')

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    MainApplication(root).pack(side="top", fill="both", expand=True)

    root.mainloop()

