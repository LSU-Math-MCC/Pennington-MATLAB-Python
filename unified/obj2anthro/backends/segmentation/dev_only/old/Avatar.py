from networkx import is_empty
import numpy as np
from numpy import uint32

class Avatar:
    
    def __init__(self, input, varargin):
            if isinstance(input, dict):
                self.v = input.v
                self.f = uint32(input.f)
            elif isinstance(input, str): 
                if input[-3:] == 'obj':
                    try:
                        myObj = readObj(input);
                    except:
                        myObj = read_obj(input);
                    
                    self.v = myObj.v
                    self.f = uint32(myObj.f)
                elif input[-3:] == 'obj':
                    [v,f] = read_ply(input);
                    self.v = v
                    self.f = uint32(f)
                
            
            
            [CaseRot,alpha2,alpha3,self.v] = fixOrientation(self);
            
            self.circ_ellipse=0
            self.circ_cpd=0
            self.steps=[1, 2, 3]
            self.Vol_SA=0
            self.template_only=0
            markers=0
            markers_template=0
            armpitOld=0
            
             
            if (varargin != []):
                for i=1:2:length(varargin)-1
                    if varargin[i]=='circumference':
                        if varargin[i+1]=='ellipse':
                            self.circ_ellipse=1;
                        elif varargin[i+1]=='cpd':
                            self.circ_cpd=1;
                        elif varargin[i+1]=='all':
                            self.circ_cpd=1;
                            self.circ_ellipse=1;
                    elif varargin[i]=='steps':
                        self.steps=varargin[i+1];
                    elif varargin[i]=='Vol_SA':
                        if varargin[i+1]=='on' or varargin[i+1]==1:
                            self.Vol_SA=1;
                    elif varargin[i]=='SA':
                        if varargin[i+1]=='on'  or  varargin[i+1]==1:
                            self.SA=1;
                     
                    elif varargin[i]=='WB_SA_only':
                        if varargin[i+1]=='on'  or  varargin[i+1]==1:
                            self.WB_SA_only=1;
             
                    elif varargin[i]=='template_only':
                        if varargin[i+1]=='on'  or  varargin[i+1]==1:
                            self.template_only=1;
                        
                    elif varargin[i]=='markers':
                        if varargin[i+1]=='on'  or  varargin[i+1]==1:
                            markers=1;
                         
                    elif varargin[i]=='markers_template':
                        if varargin[i+1]=='on'  or  varargin[i+1]==1:
                            markers_template=1;
                         
                    elif varargin[i]=='armpits_old':
                        if varargin[i+1]=='on'  or  varargin[i+1]==1:
                            armpitOld = 1;
                        elif varargin[i+1]=='off'  or  varargin[i+1]==0:
                            armpitOld = 0;
                        
                    else:
                        print('Wrong input arguments');
                        break;
                    
                
             


            if ismember(1,self.steps) #CleaningMesh
           # [self.f,self.v] = deleteFaceIntersections(self.f,self.v);
            [self.v,self.f] = CleaningMesh(self.v,self.f);
            [self.f,self.v] = removeBoundaryProblems(self.f,self.v);
            end
            if ismember(2,self.steps) # Meshrepair
            [~,self.f,self.v] = meshRepair(self.f,self.v);
            #[self.f,self.v] = deleteFaceIntersections(self.f,self.v);
            end
             
            
            if (self.circ_cpd  || self.template_only) #|| self.Vol_SA
                # initialize the configuration for the CPD template matching algorithm
                [self.circ_template_s,self.circ_template_l,self.circ_template_xl] = init_circ();
                self.config_cpd = initialize_config_cpd;
                self.config_cpd.model = self.circ_template_s;
            end
            
            if (ismember(3,self.steps) || self.Vol_SA)
                # feet mins
                [self.l_foot, self.r_foot] = getLegsMin(self);

                # crotch
                [self.crotch] = getCrotch(self);

                # armpits
                if armpitOld == 1
                    [self.r_armpit, self.l_armpit] = getArmpits(self);    
                else # Use alternative armpit method
                    [self.r_armpit, self.l_armpit] = getArmpitsAlt(self); 
                end

                # find arms
                [~, self.rArmIdx, ~] = armSearch(self, 'r');
                [~, self.lArmIdx, ~] = armSearch(self, 'l');

                # shoulders
                [self.lShoulder, self.rShoulder] = getShoulders(self);

                # correct Crotch
                [self.crotch] = adjustCrotch(self); 

                # collar
                [self.collar] = getCollar(self);

                # wrists
                [self.r_wrist, self.l_wrist, self.r_wristgirth, self.l_wristgirth,...
                 self.rwrist_front,self.rwrist_back,self.rwrist_lateral,self.rwrist_medial,...
                 self.lwrist_front,self.lwrist_back,self.lwrist_lateral,self.lwrist_medial] = getWrist(self);

                # hips
                [self.hipCircumference, hipPoints] = getHip(self);
                self.r_hip = hipPoints(1,:);
                self.l_hip = hipPoints(2,:);
                self.lowerBack = hipPoints(3,:);

                # leg indices
                [self.legIdx] = getLegs(self);

                # head and trunk
                self.headIdx = getHead(self);
                self.trunkIdx = trunkSearch(self);

                # waist
                [self.waistCircumference, ~] = getWaist(self);

                # arm lengths
                [self.leftArmLength, self.rightArmLength, self.armMaxR, self.armMaxL] = getArmLength(self);

                # length from collar to scalp
                [self.collarScalpLength] = getCollarScalpLength(self);

                # length from crotch to collar
                [self.trunkLength] = getTrunkLength(self);

                # ankles
                [self.l_ankle,self.r_ankle, self.r_ankle_girth, self.l_ankle_girth, self.rankle_medialPoint, self.rankle_lateralPoint, self.lankle_medialPoint, self.lankle_lateralPoint] = getAnkleGirth(self);

                # length of legs
                [self.lLegLength, self.rLegLength] = getLegLength(self);

                # Calves
                [self.lCalfCircumference, self.rCalfCircumference, self.rcalf_backPoint, self.rcalf_medialPoint, self.rcalf_lateralPoint, self.lcalf_backPoint, self.lcalf_medialPoint, self.lcalf_lateralPoint] = getCalf(self);

                # left and right thigh girths
                [self.rThighGirth, self.lThighGirth,...
                 self.rthigh_front,self.rthigh_back,self.rthigh_lateral,self.rthigh_medial,...
                 self.lthigh_front,self.lthigh_back,self.lthigh_lateral,self.lthigh_medial] = getThighGirth(self);

                # crotch height
                [self.crotchHeight] = getCrotchHeight(self);

                # arm girths
                [self.r_forearmgirth,self.l_forearmgirth,...
                 self.r_bicepgirth, self.l_bicepgirth,...
                 self.rforearm_front, self.rforearm_back, self.rforearm_lateral, self.rforearm_medial,...
                 self.lforearm_front, self.lforearm_back, self.lforearm_lateral, self.lforearm_medial,...
                 self.rbicep_front, self.rbicep_back, self.rbicep_lateral, self.rbicep_medial,...
                 self.lbicep_front, self.lbicep_back, self.lbicep_lateral, self.lbicep_medial] = getArmGirth(self);

                # chest circumference
                self.chestCircumference = getChestCircumference(self);

                self.nose_tip = getNoseTip(self);
                [self.rtoe_tip, self.rheel_tip, self.ltoe_tip, self.lheel_tip] = getFeetTips(self);

                # front, back, left, and right curve
                [self.rcurve, self.lcurve, ~] = getCurve(self,1,12);
                [self.fcurve, self.bcurve, ~] = getCurve(self,2,12);

                # get body type
                self.bodyType = getBodyType(self);
                
                self.list_of_properties(1:44) = [
                    self.r_armpit(1); self.r_armpit(3); self.l_armpit(1); self.l_armpit(3); self.r_foot(:); self.l_foot(:); self.crotch(:); self.lShoulder(:); self.rShoulder(:); self.collar(:); ...
                    self.chestCircumference.value; self.waistCircumference.value; self.hipCircumference.value; ...
                    self.rThighGirth.value; self.lThighGirth.value; self.rCalfCircumference.value; self.lCalfCircumference.value; ...
                    self.r_wristgirth.value; self.l_wristgirth.value; self.r_forearmgirth.value; self.l_forearmgirth.value;... 
                    self.r_bicepgirth.value; self.l_bicepgirth.value; self.r_ankle_girth.value; self.l_ankle_girth.value;...
                    self.leftArmLength; self.rightArmLength; self.collarScalpLength; self.trunkLength; self.lLegLength; self.rLegLength; self.crotchHeight
                    ];  
            end
            
            if  self.template_only #|| self.Vol_SA 

                # fit template to head and neck
                [self.head_neck_template_v,self.head_neck_template_f, self.head_neck_bottomSliceReducted_v, self.head_neck_bottomSliceReducted_f] = templateFitting_headNeck(self);

                # list of faces and vertices for the fitted template
                self.v_t = [self.r_leg_template_v;self.l_leg_template_v;self.trunk_template_v;self.r_arm_template_v;self.l_arm_template_v;self.head_neck_template_v];
                self.f_t = [self.r_leg_template_f;self.l_leg_template_f;self.trunk_template_f;self.r_arm_template_f;self.l_arm_template_f;self.head_neck_template_f];
            end 
                
            if self.SA
                self.f = fixFaceOrientation2(self.f,self.v);
                [self.volume, self.surfaceArea] = getSurfaceArea(self);
                
                 self.list_of_properties(45:51) = [... 
                 self.surfaceArea.total; self.surfaceArea.lArm; self.surfaceArea.rArm; self.surfaceArea.head...
                 ;self.surfaceArea.lleg; self.surfaceArea.rleg;self.surfaceArea.trunk]; ...; self.surfaceArea.legs
            end 
        
            if self.WB_SA_only
                self.f = fixFaceOrientation2(self.f,self.v);
                [self.surfaceArea] = getWBSurfaceArea(self);
            end 
        
            if self.Vol_SA
                # volume and surface area
                self.f = fixFaceOrientation2(self.f,self.v);
                [self.volume, self.surfaceArea] = getSurfaceAreaAndVolume(self);
               
                  self.list_of_properties(45:end-2) = [self.volume.trunk; ...self.volume.legs;... # self.volume.lleg; self.volume.rleg;
                 self.volume.total; self.volume.lArm; self.volume.rArm; self.volume.head;... 
                 self.surfaceArea.total; self.surfaceArea.lArm; self.surfaceArea.rArm; self.surfaceArea.head;...
                # ;self.surfaceArea.legs;
                 self.surfaceArea.lleg; self.surfaceArea.rleg;self.surfaceArea.trunk]; ...     ; self.volume.legs;
            end 
            
            if markers
                if max(max(v))>1000
                    v = v*10^(-3);
                end 
                self.v = returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,[]);
                self = OriginalRoatationLandmarks (CaseRot,alpha2,alpha3,self);
                self.createMarkers_Txt_Jpg(input);
            end 
            if markers_template
                self.v = returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,[]);
                self = OriginalRoatationLandmarks (CaseRot,alpha2,alpha3,self);
                self.createMarkers_template(input);
            end 
         end
    v = None
    f = None
    v_t = None  # Vertices for matched template
    f_t = None  # Faces for matched template     
    
    list_of_properties = np.zeros((58, 1))
    circ_template_s = None
    circ_template_l = None
    circ_template_xl = None
    
    r_wrist = np.nan * np.ones((1,3)) # right wrist
    rwrist_ulnar = np.nan * np.ones((1,3))
    rwrist_radial = np.nan * np.ones((1,3))
    r_armpit = np.nan * np.ones((1,3)) # right armpit
    r_hip = np.nan * np.ones((1,3)) # right hip
    l_hip = np.nan * np.ones((1,3)) # left hip
    l_armpit = np.nan * np.ones((1,3)) # left armpit
    l_wrist = np.nan * np.ones((1,3)) # left wrist
    lwrist_ulnar = np.nan * np.ones((1,3))
    lwrist_radial = np.nan * np.ones((1,3))
    r_foot = np.nan * np.ones((1,3)) # right foot min
    l_foot = np.nan * np.ones((1,3)) # left foot min
    crotch = np.nan * np.ones((1,3)) # crotch
    l_ankle = np.nan * np.ones((1,3))
    r_ankle = np.nan * np.ones((1,3))
    lShoulder = np.nan * np.ones((1,3))
    rShoulder = np.nan * np.ones((1,3))
    armMaxR = np.nan * np.ones((1,3)) # far-end of the right arm
    armMaxL = np.nan * np.ones((1,3)) # far-end of the left arm
    collar = None
    
    nose_tip = np.nan * np.ones((1,3)) # max y value in the 1/3 between the top of the head and the highest shoulder
    rwrist_front = np.nan * np.ones((1,3))
    rwrist_back = np.nan * np.ones((1,3))
    rwrist_lateral = np.nan * np.ones((1,3))
    rwrist_medial = np.nan * np.ones((1,3))
    lwrist_front = np.nan * np.ones((1,3))
    lwrist_back = np.nan * np.ones((1,3))
    lwrist_lateral = np.nan * np.ones((1,3))
    lwrist_medial = np.nan * np.ones((1,3))
    rforearm_front = np.nan * np.ones((1,3))
    rforearm_back = np.nan * np.ones((1,3))
    rforearm_lateral = np.nan * np.ones((1,3))
    rforearm_medial = np.nan * np.ones((1,3))
    lforearm_front = np.nan * np.ones((1,3))
    lforearm_back = np.nan * np.ones((1,3))
    lforearm_lateral = np.nan * np.ones((1,3))
    lforearm_medial = np.nan * np.ones((1,3))
    rbicep_front = np.nan * np.ones((1,3))
    rbicep_back = np.nan * np.ones((1,3))
    rbicep_lateral = np.nan * np.ones((1,3))
    rbicep_medial = np.nan * np.ones((1,3))
    lbicep_front = np.nan * np.ones((1,3))
    lbicep_back = np.nan * np.ones((1,3))
    lbicep_lateral = np.nan * np.ones((1,3))
    lbicep_medial = np.nan * np.ones((1,3))
    lowerBack = np.nan * np.ones((1,3))
    rtoe_tip = np.nan * np.ones((1,3))
    rheel_tip = np.nan * np.ones((1,3))
    ltoe_tip = np.nan * np.ones((1,3))
    lheel_tip = np.nan * np.ones((1,3))
    rankle_medialPoint = np.nan * np.ones((1,3))
    rankle_lateralPoint = np.nan * np.ones((1,3))
    lankle_medialPoint = np.nan * np.ones((1,3))
    lankle_lateralPoint = np.nan * np.ones((1,3))
    rcalf_backPoint = np.nan * np.ones((1,3))
    rcalf_medialPoint = np.nan * np.ones((1,3))
    rcalf_lateralPoint = np.nan * np.ones((1,3))
    lcalf_backPoint = np.nan * np.ones((1,3))
    lcalf_medialPoint = np.nan * np.ones((1,3))
    lcalf_lateralPoint = np.nan * np.ones((1,3))
    rthigh_front = np.nan * np.ones((1,3))
    rthigh_back = np.nan * np.ones((1,3))
    rthigh_lateral = np.nan * np.ones((1,3))
    rthigh_medial = np.nan * np.ones((1,3))
    lthigh_front = np.nan * np.ones((1,3))
    lthigh_back = np.nan * np.ones((1,3))
    lthigh_lateral = np.nan * np.ones((1,3))
    lthigh_medial = np.nan * np.ones((1,3))
    
    r_leg_template_v = None
    l_leg_template_v = None
    r_arm_template_v = None
    l_arm_template_v = None
    trunk_template_v = None
    head_neck_template_v = None
    r_leg_template_f = None
    l_leg_template_f = None
    r_arm_template_f = None
    l_arm_template_f = None
    trunk_template_f = None
    head_neck_template_f = None
    head_neck_bottomSliceReducted_v = None
    head_neck_bottomSliceReducted_f = None
    
    chestCircumference = None
    waistCircumference = None
    hipCircumference = None # Circumference of the hip
    rThighGirth = None
    lThighGirth = None
    rCalfCircumference = None #Circumference of right calf
    lCalfCircumference = None #Circumference of left calf
    r_wristgirth = None
    l_wristgirth = None
    r_forearmgirth = None
    l_forearmgirth = None
    r_bicepgirth = None
    l_bicepgirth = None
    r_ankle_girth = None
    l_ankle_girth = None
    
    
    volume = None
    surfaceArea = None
    
    leftArmLength = None
    rightArmLength = None
    collarScalpLength = None
    trunkLength = None
    lLegLength = None
    rLegLength = None
    crotchHeight = None
    
    bodyType = None
    
    lcurve = None
    rcurve = None
    fcurve = None
    bcurve = None
    
    lArmIdx = None # Indices of left arm
    rArmIdx = None # Indices of right arm
    legIdx = None # Indices of legs
    headIdx = None # Indices of head
    trunkIdx = None # Indices of trunk 
     
    circ_ellipse = None
    circ_cpd = None
    steps = None
    SA = None
    WB_SA_only = None
    Vol_SA = None
    template_only = None
    
    config_cpd = None
