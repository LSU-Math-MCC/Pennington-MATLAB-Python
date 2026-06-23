classdef Avatar
    
    %% This avatar includes the legs division for the volume calculation
    
    
 % General description: Avatar finds landmarks and computes lengths, circumferences, volumes, SA of an avatar
    
 % How to use the code:
    % call Avatar(my_example)
     
    %  my_example can be a string either  'my_example.obj' or 'my_example.ply' or a struct with my_example.f being the
    %  list of faces and my_example.v being the list of vertices.
    %  my_example.v has 3 columns containing (x,y,z) coordinates of the
    %  point cloud and the my_example.f has 3 columns containing vertex indices, i.e., row numbers of my_example.v
     
     % Additional inputs include 'steps','circumference','Vol_SA', 'template_only'
     
        % Values for 'steps' can be:
            % 1 for Cleaning mesh; 
            % 2 for Mesh repair;
            % 3 for Landmark detection and lengths and circumferences.
            % default is [1,2,3]
            % example : Avatar(my_example,'steps',[2,3])
        % Values for 'circumference' can be:
            % 'CPD' calculates circumferences after fitting a circle template using CPD
            % 'ellipse' calculates circumferences after fitting an ellipse 
            % 'all' does both
            % default does not do any of them
            % example : Avatar(my_example,'circumference','ellipse')            
        % Use 'Vol_SA' > 'on' to calculate volumes and surface areas, default is 'off'
        % Use 'SA' > 'on' to calculate SA only.
        % Use 'WB_SA_only' > 'on' to calculate only the total body SA.
        % Use 'template_only' > 'on' to just do the template fitting, default is 'off'
            % example : Avatar(my_example,'template_only','on') 
        % Use 'markers' > 'on' to save the txt file and the 2D figure from
        % Use 'markers_template' > 'on' to save the txt file and the 2D figure from
        % the front and the back of the Avatar with the 30 markers
        % Use 'armpits_alt' > 'on' to use alternative armpit method, default is 'off'
        


        % Note: 
            % More than one property can be set at the same time
            % Ordering which properties are mentionned does not matter
            % example: Avatar(my_example,'Vol_SA','on','circumference','ellipse','steps',[2,3])   
    
    properties
        v % Vertices
        f % Faces
        v_t % Vertices for matched template
        f_t % Faces for matched template     
        
        list_of_properties = zeros(58,1)

        circ_template_s;
        circ_template_l;
        circ_template_xl;
        
        r_wrist = NaN(1,3) % right wrist
        rwrist_ulnar = NaN(1,3)
        rwrist_radial = NaN(1,3)
        r_armpit = NaN(1,3) % right armpit
        r_hip = NaN(1,3) % right hip
        l_hip = NaN(1,3) % left hip
        l_armpit = NaN(1,3) % left armpit
        l_wrist = NaN(1,3) % left wrist
        lwrist_ulnar = NaN(1,3)
        lwrist_radial = NaN(1,3)
        r_foot = NaN(1,3) % right foot min
        l_foot = NaN(1,3) % left foot min
        crotch = NaN(1,3) % crotch
%         r_waist = NaN(1,3) %right waist
%         l_waist = NaN(1,3) %left waist   
        l_ankle = NaN(1,3)
        r_ankle = NaN(1,3)
        lShoulder = NaN(1,3)
        rShoulder = NaN(1,3)
        armMaxR = NaN(1,3) % far-end of the right arm
        armMaxL = NaN(1,3) % far-end of the left arm
        collar
        
        nose_tip = NaN(1,3) % max y value in the 1/3 between the top of the head and the highest shoulder
        rwrist_front = NaN(1,3)
        rwrist_back = NaN(1,3)
        rwrist_lateral = NaN(1,3)
        rwrist_medial = NaN(1,3)
        lwrist_front = NaN(1,3)
        lwrist_back = NaN(1,3)
        lwrist_lateral = NaN(1,3)
        lwrist_medial = NaN(1,3)
        rforearm_front = NaN(1,3)
        rforearm_back = NaN(1,3)
        rforearm_lateral = NaN(1,3)
        rforearm_medial = NaN(1,3)
        lforearm_front = NaN(1,3)
        lforearm_back = NaN(1,3)
        lforearm_lateral = NaN(1,3)
        lforearm_medial = NaN(1,3)
        rbicep_front = NaN(1,3)
        rbicep_back = NaN(1,3)
        rbicep_lateral = NaN(1,3)
        rbicep_medial = NaN(1,3)
        lbicep_front = NaN(1,3)
        lbicep_back = NaN(1,3)
        lbicep_lateral = NaN(1,3)
        lbicep_medial = NaN(1,3)
        lowerBack = NaN(1,3)
        rtoe_tip = NaN(1,3)
        rheel_tip = NaN(1,3)
        ltoe_tip = NaN(1,3)
        lheel_tip = NaN(1,3)
        rankle_medialPoint = NaN(1,3)
        rankle_lateralPoint = NaN(1,3)
        lankle_medialPoint = NaN(1,3)
        lankle_lateralPoint = NaN(1,3)
        rcalf_backPoint = NaN(1,3)
        rcalf_medialPoint = NaN(1,3)
        rcalf_lateralPoint = NaN(1,3)
        lcalf_backPoint = NaN(1,3)
        lcalf_medialPoint = NaN(1,3)
        lcalf_lateralPoint = NaN(1,3)
        rthigh_front = NaN(1,3)
        rthigh_back = NaN(1,3)
        rthigh_lateral = NaN(1,3)
        rthigh_medial = NaN(1,3)
        lthigh_front = NaN(1,3)
        lthigh_back = NaN(1,3)
        lthigh_lateral = NaN(1,3)
        lthigh_medial = NaN(1,3)
        
        r_leg_template_v
        l_leg_template_v
        r_arm_template_v
        l_arm_template_v
        trunk_template_v
        head_neck_template_v
        r_leg_template_f
        l_leg_template_f
        r_arm_template_f
        l_arm_template_f
        trunk_template_f
        head_neck_template_f
        head_neck_bottomSliceReducted_v
        head_neck_bottomSliceReducted_f
        
        chestCircumference
        waistCircumference
        hipCircumference % Circumference of the hip
        rThighGirth
        lThighGirth
        rCalfCircumference %Circumference of right calf
        lCalfCircumference %Circumference of left calf
        r_wristgirth
        l_wristgirth
        r_forearmgirth
        l_forearmgirth
        r_bicepgirth
        l_bicepgirth
        r_ankle_girth
        l_ankle_girth
        
        
        volume
        surfaceArea
        
        leftArmLength
        rightArmLength
        collarScalpLength
        trunkLength
        lLegLength
        rLegLength
        crotchHeight
        
        bodyType
        
        lcurve
        rcurve
        fcurve
        bcurve
        
        lArmIdx % Indices of left arm
        rArmIdx % Indices of right arm
        legIdx % Indices of legs
        headIdx % Indices of head
        trunkIdx % Indices of trunk 
         
        circ_ellipse
        circ_cpd
        steps
        SA
        WB_SA_only
        Vol_SA
        template_only
        
        config_cpd
    end
    
    methods
        %% Constructor
        function self = Avatar(input,varargin)
            if isstruct(input)
                self.v=input.v;
                self.f=uint32(input.f);
            elseif ischar(input) 
                if strcmp(input(end-2:end),'obj')
                    myObj = readObj(input);
                    self.v = myObj.v;
                    self.f = uint32(myObj.f);
                elseif strcmp(input(end-2:end),'ply')
                    [v,f] = read_ply(input);
                    self.v = v;
                    self.f = uint32(f);
                end
            end
            




            
            [CaseRot,alpha2,alpha3,self.v] = fixOrientation(self);
            
            self.circ_ellipse=0;
            self.circ_cpd=0;
            self.steps=[1 2 3];
            self.Vol_SA=0;
            self.template_only=0;
            markers=0;
            markers_template=0;
            armpitOld=0;
            
             
            if (~isempty(varargin))
                for i=1:2:length(varargin)-1
                    if strcmp(varargin{i},'circumference')
                        if strcmp(varargin{i+1},'ellipse')
                            self.circ_ellipse=1;
                        elseif strcmp(varargin{i+1},'cpd')
                            self.circ_cpd=1;
                        elseif strcmp(varargin{i+1},'all')
                            self.circ_cpd=1;
                            self.circ_ellipse=1;
                        end
                    elseif strcmp(varargin{i},'steps')
                        self.steps=varargin{i+1};
                       
%                         if sum(ismember(varargin{i+1},1),2) %CleaningMesh
%                             self.steps=1;
%                         elseif sum(ismember(varargin{i+1},2),2) % MeshRepair
%                             self.steps=2;
%                         elseif sum(ismember(varargin{i+1},3),2) %LandmarkDetection_DAs
%                            self.steps=3;
%                         end
                     elseif strcmp(varargin{i},'Vol_SA')
                         if (strcmp(varargin{i+1},'on') || (varargin{i+1}==1))
                            self.Vol_SA=1;
                         end
                     elseif strcmp(varargin{i},'SA')
                     if (strcmp(varargin{i+1},'on') || (varargin{i+1}==1))
                        self.SA=1;
                     end
                     elseif strcmp(varargin{i},'WB_SA_only')
                     if (strcmp(varargin{i+1},'on') || (varargin{i+1}==1))
                        self.WB_SA_only=1;
                     end
                     elseif strcmp(varargin{i},'template_only')
                        if (strcmp(varargin{i+1},'on') || (varargin{i+1}==1))
                            self.template_only=1;
                        end
                     elseif strcmp(varargin{i},'markers')
                         if (strcmp(varargin{i+1},'on') || (varargin{i+1}==1))
                            markers=1;
                         end
                     elseif strcmp(varargin{i},'markers_template')
                         if (strcmp(varargin{i+1},'on') || (varargin{i+1}==1))
                            markers_template=1;
                         end
                    elseif strcmp(varargin{i},'armpits_old')
                        if (strcmp(varargin{i+1},'on') || (varargin{i+1}==1))
                            armpitOld = 1;
                        elseif (strcmp(varargin{i+1},'off') || (varargin{i+1}==0))
                            armpitOld = 0;
                        end
                    else
                        disp('Wrong input arguments');
                        break;
                    end
                end
            end 
                
%                 if max(max(self.v))>1000
%                     self.v = self.v*10^(-3)/2;
%                 end 
             


            if ismember(1,self.steps) %CleaningMesh
           % [self.f,self.v] = deleteFaceIntersections(self.f,self.v);
            [self.v,self.f] = CleaningMesh(self.v,self.f);
            [self.f,self.v] = removeBoundaryProblems(self.f,self.v);
            end
            if ismember(2,self.steps) % Meshrepair
            [~,self.f,self.v] = meshRepair(self.f,self.v);
            %[self.f,self.v] = deleteFaceIntersections(self.f,self.v);
            end
            
             %self.v = fixOrientation(self);

            
             
            %%
            if (self.circ_cpd  || self.template_only) %|| self.Vol_SA
                % initialize the configuration for the CPD template matching algorithm
                [self.circ_template_s,self.circ_template_l,self.circ_template_xl] = init_circ();
                self.config_cpd = initialize_config_cpd;
                self.config_cpd.model = self.circ_template_s;
            end
            
            if (ismember(3,self.steps) || self.Vol_SA)
                % feet mins
                [self.l_foot, self.r_foot] = getLegsMin(self);

                % crotch
                [self.crotch] = getCrotch(self);

                % armpits
                if armpitOld == 1
                    [self.r_armpit, self.l_armpit] = getArmpits(self);    
                else % Use alternative armpit method
                    [self.r_armpit, self.l_armpit] = getArmpitsAlt(self); 
                end

                % find arms
                [~, self.rArmIdx, ~] = armSearch(self, 'r');
                [~, self.lArmIdx, ~] = armSearch(self, 'l');

                % shoulders
                [self.lShoulder, self.rShoulder] = getShoulders(self);

                % correct Crotch
                [self.crotch] = adjustCrotch(self); 

                % collar
                [self.collar] = getCollar(self);

                % wrists
%                 [self.r_wrist, self.l_wrist, self.r_wristgirth, self.l_wristgirth,self.rwrist_ulnar,self.rwrist_radial,self.lwrist_ulnar,self.lwrist_radial] = getWrist(self);
                [self.r_wrist, self.l_wrist,...
                 self.r_wristgirth, self.l_wristgirth,...
                 self.rwrist_front,self.rwrist_back,self.rwrist_lateral,self.rwrist_medial,...
                 self.lwrist_front,self.lwrist_back,self.lwrist_lateral,self.lwrist_medial] = getWrist(self);

                % hips
                [self.hipCircumference, hipPoints] = getHip(self);
                self.r_hip = hipPoints(1,:);
                self.l_hip = hipPoints(2,:);
                self.lowerBack = hipPoints(3,:);

                % leg indices
                [self.legIdx] = getLegs(self);

                % head and trunk
                self.headIdx = getHead(self);
                self.trunkIdx = trunkSearch(self);

                % waist
                [self.waistCircumference, ~] = getWaist(self);
    %             self.k10 = waistPoints(1,:);
    %             self.k11 = waistPoints(2,:);

                % arm lengths
                [self.leftArmLength, self.rightArmLength, self.armMaxR, self.armMaxL] = getArmLength(self);

                % length from collar to scalp
                [self.collarScalpLength] = getCollarScalpLength(self);

                % length from crotch to collar
                [self.trunkLength] = getTrunkLength(self);

                % ankles
                [self.l_ankle,self.r_ankle, self.r_ankle_girth, self.l_ankle_girth, self.rankle_medialPoint, self.rankle_lateralPoint, self.lankle_medialPoint, self.lankle_lateralPoint] = getAnkleGirth(self);

                % length of legs
                [self.lLegLength, self.rLegLength] = getLegLength(self);

                % Calves
                [self.lCalfCircumference, self.rCalfCircumference, self.rcalf_backPoint, self.rcalf_medialPoint, self.rcalf_lateralPoint, self.lcalf_backPoint, self.lcalf_medialPoint, self.lcalf_lateralPoint] = getCalf(self);

                % left and right thigh girths
                [self.rThighGirth, self.lThighGirth,...
                 self.rthigh_front,self.rthigh_back,self.rthigh_lateral,self.rthigh_medial,...
                 self.lthigh_front,self.lthigh_back,self.lthigh_lateral,self.lthigh_medial] = getThighGirth(self);

                % crotch height
                [self.crotchHeight] = getCrotchHeight(self);


    %             self.trunkIdx = getTrunk(self);

                % arm girths
                [self.r_forearmgirth,self.l_forearmgirth,...
                 self.r_bicepgirth, self.l_bicepgirth,...
                 self.rforearm_front, self.rforearm_back, self.rforearm_lateral, self.rforearm_medial,...
                 self.lforearm_front, self.lforearm_back, self.lforearm_lateral, self.lforearm_medial,...
                 self.rbicep_front, self.rbicep_back, self.rbicep_lateral, self.rbicep_medial,...
                 self.lbicep_front, self.lbicep_back, self.lbicep_lateral, self.lbicep_medial] = getArmGirth(self);

                % chest circumference
                self.chestCircumference = getChestCircumference(self);

                self.nose_tip = getNoseTip(self);
                [self.rtoe_tip, self.rheel_tip, self.ltoe_tip, self.lheel_tip] = getFeetTips(self);

                
                % front, back, left, and right curve
                [self.rcurve, self.lcurve, ~] = getCurve(self,1,12);
                [self.fcurve, self.bcurve, ~] = getCurve(self,2,12);

                % get body type
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
            
            if  self.template_only %|| self.Vol_SA 
                
%                 % fit template to legs
%                 [self.r_leg_template_v,self.r_leg_template_f, self.l_leg_template_v,self.l_leg_template_f] = templateFitting_leg(self);            
% 
%                 % fit template to trunk
%                 [self.trunk_template_v,self.trunk_template_f] = templateFitting_trunk(self);
% 
%                 % fit template to arms
%                 [self.r_arm_template_v,self.r_arm_template_f, self.l_arm_template_v,self.l_arm_template_f] = templateFitting_arm(self);

                % fit template to head and neck
                [self.head_neck_template_v,self.head_neck_template_f, self.head_neck_bottomSliceReducted_v, self.head_neck_bottomSliceReducted_f] = templateFitting_headNeck(self);

                % list of faces and vertices for the fitted template
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
                % volume and surface area
                self.f = fixFaceOrientation2(self.f,self.v);
                [self.volume, self.surfaceArea] = getSurfaceAreaAndVolume(self);
               
                  self.list_of_properties(45:end) = [self.volume.trunk; self.volume.legs;... % self.volume.lleg; self.volume.rleg;
                 self.volume.total; self.volume.lArm; self.volume.rArm; self.volume.head;... 
                 self.surfaceArea.total; self.surfaceArea.lArm; self.surfaceArea.rArm; self.surfaceArea.head...
                 ;self.surfaceArea.legs;self.surfaceArea.lleg; self.surfaceArea.rleg;self.surfaceArea.trunk]; ...     ; self.volume.legs;
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
            % Knees
           % [self.left_Knee, self.right_Knee] = getKnee(self);
         end
  
        %% Methods
        
        function [Case,alpha2,alpha3,newV] = fixOrientation(self)
            % Rotates the Avatar into the Styku position
            newV = self.v;
            
            Case = [];
            alpha2 = [];
            alpha3 = [];
            dist = [max(newV(:,1)) - min(newV(:,1)),...
                    max(newV(:,2)) - min(newV(:,2)),...
                    max(newV(:,3)) - min(newV(:,3))]; % maximum distance in x,y,z direction
            [~,order] = sort(dist);
            
            if isequal(order,[1 2 3])
                [newV(:,1), newV(:,2)] = rotate_person(newV(:,1), newV(:,2), -pi/2);
                Case = 1;
            elseif isequal(order,[1 3 2])
                [newV(:,1), newV(:,3)] = rotate_person(newV(:,1), newV(:,3), -pi/2);
                [newV(:,2), newV(:,3)] = rotate_person(newV(:,2), newV(:,3), -pi/2);
                Case = 2;
            elseif isequal(order,[2 3 1])
                [newV(:,1), newV(:,3)] = rotate_person(newV(:,1), newV(:,3), pi/2);
                Case = 3;
            elseif isequal(order, [3 1 2])
                [newV(:,2), newV(:,3)] = rotate_person(newV(:,2), newV(:,3), -pi/2);
                Case = 4;
            elseif isequal(order, [3 2 1])
                [newV(:,2), newV(:,3)] = rotate_person(newV(:,2), newV(:,3), -pi/2);
                [newV(:,1), newV(:,3)] = rotate_person(newV(:,1), newV(:,3), -pi/2);
                Case = 5;
            end
            
            % Compare distance of top and bottom in the x direction
            % Rotate if feet are on top
            M = max(newV(:,3));
            m = min(newV(:,3));
            top90 = newV(:,3) > (M - m)*0.9+m;
            bottom10 = newV(:,3) < (M - m)*0.1+m;
            topX = newV(top90,1);
            bottomX = newV(bottom10,1);
            distTop = max(topX) - min(topX);
            distBottom = max(bottomX) - min(bottomX);
            if distTop > distBottom
                [newV(:,2), newV(:,3)] = rotate_person(newV(:,2), newV(:,3), pi);
                M = max(newV(:,3));
                m = min(newV(:,3));
                bottom10 = newV(:,3) < (M - m)*0.1+m;
                alpha2 = 1;
            end
            
            % Feet need to point in the negative y direction
            % Fist, find the middle of the ankle in the y direction
            vOnLine = getVOnLine(self, newV, (M - m)*0.1+m, (1:length(newV)));
            centerY = mean([max(vOnLine(:,2)), min(vOnLine(:,2))]);
            
            if abs(centerY-max(newV(bottom10,2))) > abs(centerY-min(newV(bottom10,2)))
                [newV(:,1), newV(:,2)] = rotate_person(newV(:,1), newV(:,2), pi);
                alpha3 = 1;
            end
        end   
        
        function newV = returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,BdyPart)
            
            newV = self.v;
            
            if  ~isempty(BdyPart)
                newV = BdyPart;
            end
            
            if CaseRot == 1
                [newV(:,1), newV(:,2)] = rotate_person(newV(:,1), newV(:,2), pi/2);
            elseif CaseRot == 2
                [newV(:,1), newV(:,3)] = rotate_person(newV(:,1), newV(:,3), pi/2);
                [newV(:,2), newV(:,3)] = rotate_person(newV(:,2), newV(:,3), pi/2);
            elseif CaseRot == 3
                [newV(:,1), newV(:,3)] = rotate_person(newV(:,1), newV(:,3), -pi/2);
            elseif CaseRot == 4
                [newV(:,2), newV(:,3)] = rotate_person(newV(:,2), newV(:,3), pi/2);
            elseif CaseRot == 5
                [newV(:,2), newV(:,3)] = rotate_person(newV(:,2), newV(:,3), pi/2);
                [newV(:,1), newV(:,3)] = rotate_person(newV(:,1), newV(:,3), pi/2);
            end

            if alpha2 == 1
                [newV(:,2), newV(:,3)] = rotate_person(newV(:,2), newV(:,3), -pi);
            end

            if alpha3 == 1
                [newV(:,1), newV(:,2)] = rotate_person(newV(:,1), newV(:,2), -pi);
            end
        end
        
        function [var_v, var_f] = peelSkin(self,pCases)
            V = self.v;
            F = self.f;
            
            %figure; plot3(self.v(pCases,1),self.v(pCases,2),self.v(pCases,3),'-*'); hold on;
            for p = pCases'
                index = F(:,1) == p | F(:,2) == p | F(:,3) == p;
                F(index,:) = [];

                for j = 1:3
                   idx = F(:,j) > p; 
                   F(idx,j) = F(idx, j) - 1;
                end

                V(p,:) = [];
            end
            var_v = V;
            var_f = F;
        end
        
        function [v,f] = deleteProblems(self,bdyEdges)
            %bdyEdges = getBoundaryEdges(self);
            subs = [bdyEdges(:,1);bdyEdges(:,2)];
            A = accumarray(subs,1);
            vertices = find(A ~= 2 & A~=0);
            %vertices = unique([bdyEdges(:,1);bdyEdges(:,2)]);
            [v,f] = peelSkin(self,vertices);
        end
        
        function [leftknee, rightknee] = getKnee(self)
             v1 = self.v(:,1);
            rLegIdx = self.legIdx(v1(self.legIdx) < self.crotch(1,1));
            lLegIdx = self.legIdx(v1(self.legIdx) >= self.crotch(1,1));
%             %option 1
%             dist = self.k3(3) - self.r_ankle(3);
%             zValue = 0.5*dist + self.r_ankle(3);
          %option 2
            v2 = self.rThighGirth.points;
            v3 = self.lThighGirth.points;
            [~,rightIdx] = min(v2(:,2));
            [~,leftIdx] = min(v3(:,2));
            z_rThigh = v2(rightIdx,3);
            z_lThigh = v3(leftIdx,3);
            
            
            v4 = self.rCalfCircumference.points;
            v5 = self.lCalfCircumference.points;
            [~,rightIdx] = min(v4(:,2));
            [~,leftIdx] = min(v5(:,2));
            z_rCalf = v4(rightIdx,3);
            z_lCalf = v5(leftIdx,3);
            
            zValueR = mean([z_rThigh,z_rCalf]);
            zValueL = mean([z_lThigh,z_lCalf]);
            
            
            vOnLine = getVOnLine(self, self.v, zValueR, rLegIdx);
            [rightknee.value,b] = getCircumference(vOnLine(:,1), vOnLine(:,2));
            rightknee.points = vOnLine(b,:);
            [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(vOnLine(:,1), vOnLine(:,2));
            rightknee.a_over_b = semiminor_axis/semimajor_axis;
            rightknee.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            
            if(self.circ_elipse)
                [rightknee.templatePoints,rightknee.templateValue] = template_circumference(vOnLine);
            end
%             dist = self.k6(3) - self.l_ankle(3);
%             zValue = 0.5*dist + self.l_ankle(3);
            vOnLine = getVOnLine(self, self.v, zValueL, lLegIdx);
            [leftknee.value,b] = getCircumference(vOnLine(:,1), vOnLine(:,2));
            leftknee.points = vOnLine(b,:); 
            [semimajor_axis, semiminor_axis, x0, y0, ~] = ellipse_fit(vOnLine(:,1), vOnLine(:,2));
            leftknee.a_over_b = semiminor_axis/semimajor_axis;
            leftknee.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            %plot_ellipse(x0,y0,semimajor_axis,semiminor_axis,vOnLine(b,:))
            
            if(self.circ_elipse)
                [leftknee.templatePoints,leftknee.templateValue] = template_circumference( vOnLine);
            end
        end
        
        function [lLeg, rLeg] = getLegsMin(self)
        % Gets legs vertices    
            v1 = self.v(:,1);
            v2 = self.v(:,2);
            v3 = self.v(:,3);
            [m_V, m_I] = min(v3);          %Gives lowest point on avatar, which
            leg1 = [v1(m_I) v2(m_I) m_V];  %would be on one of the feet

            if leg1(1,1) > 0   %If Leg1 is the left leg
                OppositeSide =[v1(v1<0) v2(v1<0) v3(v1<0)]; %Gives right side
            else               %If Leg1 is the right leg
                OppositeSide =[v1(v1>0) v2(v1>0) v3(v1>0)]; %Gives left side
            end

            [~, I] = min(OppositeSide(:,3));        %Gives lowest point on other leg
            leg2 = [OppositeSide(I, 1) OppositeSide(I, 2) OppositeSide(I, 3)];

            if leg1(1,1) < leg2(1,1)
                lLeg = leg2;
                rLeg = leg1;
            else
                lLeg = leg1;
                rLeg = leg2;
            end
        end
        
        function [k9] = getCrotch(self)
        % Gets the crotch (only x and z coords)
            [k9_1, k9_3] = self.findMaxMin(self.r_foot, self.l_foot, 50);
            k9_2 = getMissingY(self,k9_1,k9_3);
            k9 = [k9_1, k9_2, k9_3];
        end
        
        function [y] = getMissingY(self,x,z)	
            roundv1 = round(self.v(:,1));	
            roundv3 = round(self.v(:,3));	
            y = self.v(roundv1 == round(x)...	
                              & roundv3 == round(z),:);	
            y = median(y(:,2)); % find average if more than one was found	
        end
        
        function [k9_adj] = adjustCrotch(self)
            k9_adj = self.crotch;
             N = 20;
%             zPoints = linspace(self.k9(3), (min(self.k2(3),self.k4(3))+self.k9(3))/2, N);
            zPoints = linspace(self.crotch(3), min(self.r_armpit(3),self.l_armpit(3)), N);
            mx_v_bot = zeros(2,N); % max points in y direction
            delta_v2 = zeros(1,N);
%             delta_v1 = zeros(1,N);
            cnd_vector = ones(1,N);
            armIdx = [self.lArmIdx; self.rArmIdx];
            noArmIdx = (1:length(self.v));
            noArmIdx(armIdx) = [];   
%             figure;
            for i = 1:N
                vOnLine = getVOnLine(self, self.v, zPoints(i), noArmIdx);
                v1 = vOnLine(:,1);
                v2 = vOnLine(:,2);
                
                k = convhull(v1,v2);
                v1_cnvh = v1(k);
                v2_cnvh = v2(k);
                mean_v2 = mean(v2);
                
                % Get v1 and v2 of bottom
                idx = v2 < mean_v2;
                v2_bot = v2(idx);
                v1_bot = v1(idx);
                
                % Get v1 and v2 of bottom convexhull
                idx = v2_cnvh < mean_v2;
                v2h_bot = v2_cnvh(idx);
                v1h_bot = v1_cnvh(idx);    
                
                % Get v1 and v2 of bottom and middle
                qtrL = mean(v1_bot) + ((max(v1_bot)-mean(v1_bot))./2);
                qtrS = mean(v1_bot) + ((min(v1_bot)-mean(v1_bot))./2);
                idx = logical((v1_bot > qtrS) .* (v1_bot < qtrL)); % Indices of vertices of middle
                v1_bot = v1_bot(idx);
                v2_bot = v2_bot(idx);
                
                [mx_v_bot(2,i), Idx_v2_bot] = max(v2_bot);
                mx_v_bot(1,i) = v1_bot(Idx_v2_bot);
                Idx_mx_cvh = (v1h_bot == mx_v_bot(1,i));
%                 subplot(4,5,i); plot(v1_bot,v2_bot,'.'); hold on; %
% %                 plot(mx_v1_bot(i),mx_v2_bot(i),'*'); hold on; %
%                 plot(v1h_bot,v2h_bot,'.'); hold on; %
               
                if (i~=1)                
                    if (sum(Idx_mx_cvh)>0) % Check if max is part of convexhull
                        cnd_vector(i) = 0;
                    else
                        mid_v1 = (max(v1_bot)+min(v1_bot)) ./ 2;
                        
                        % Get v1 of min of bottom, convex hull, and right side in y direction 
                        r_Idx = v1h_bot > mid_v1;
                        [v2h_botR, IdxR] = min(v2h_bot(r_Idx));
                        v1h_botR = v1h_bot(r_Idx); 
                        v1h_botR = v1h_botR(IdxR);
                        
                        % Get v1 of min of bottom, convex hull, and left side in y direction
                        l_Idx = v1h_bot < mid_v1;
                        [v2h_botL, IdxL] = min(v2h_bot(l_Idx));
                        v1h_botL = v1h_bot(l_Idx); 
                        v1h_botL = v1h_botL(IdxL);  
                        
%                         delta_v1(i) = v1h_botR - v1h_botL;
                        if (isempty(v2h_botR))
                            mid_v2h = v2h_botL;
                        elseif (isempty(v2h_botL))
                            mid_v2h = v2h_botR;
                        else
                            mid_v2h = (v2h_botR + v2h_botL)./ 2;
                        end
                        if (isempty(v1h_botR))
                            mid_v1h = v1h_botL;
                        elseif (isempty(v1h_botL))
                            mid_v1h = v1h_botR;
                        else
                            mid_v1h = (v1h_botR + v1h_botL)./ 2;
                        end                        
%                         mid_v2h = (v2h_botR + v2h_botL)./ 2; % average of left and right in y-direction
%                         mid_v1h = (v1h_botR + v1h_botL)./ 2; % average of left and right in x-direction
%                         try
                        [~,IdxMid_v1_bot] = min(abs(v1_bot-mid_v1h));
%                         catch
%                             sima = 0;
%                         end
                        mid_v2_bot = v2_bot(IdxMid_v1_bot);
                        dlt_cnd = 1;
                        org_IdxMid_v1_bot = IdxMid_v1_bot;
                        org_v2_bot = v2_bot;
                        while (dlt_cnd)
                            v2_bot = v2_bot(1:end-1);
                            if (isempty(v2_bot))
                                dlt_cnd = 0;
                            else
                                dlt_cnd = mid_v2_bot<=v2_bot(end);
                            end
                        end
                        if (~isempty(v2_bot))
                            dlt_cnd = 1;
                        end
                        while (dlt_cnd)
                            v2_bot = v2_bot(2:end);
                            IdxMid_v1_bot = IdxMid_v1_bot - 1;                            
                            if (isempty(v2_bot))
                                dlt_cnd = 0;
                            else
                                dlt_cnd = mid_v2_bot<=v2_bot(1);
                            end
                        end
                        if (~isempty(v2_bot))
                            [mid_max_v2_bot,Idx_md_mx] = max(v2_bot);
    %                         Idx_md = find(v2_bot==mid_v2_bot);
                            Idx_md = IdxMid_v1_bot;
                            x_mid_max_v2_bot=v1_bot(org_v2_bot == mid_max_v2_bot);
    %                         cnd1 = (Idx_md_mx-Idx_md)>(length(v2_bot)-Idx_md_mx);
    %                         cnd2 = (Idx_md_mx-Idx_md)>(Idx_md_mx-1);
    %                         if (cnd1 || cnd2)
                            x_v1_bot = v1_bot(org_IdxMid_v1_bot);
                            if(abs(x_mid_max_v2_bot)>abs(x_v1_bot))
                                delta_v2(i) = mid_v2_bot - mid_v2h;      
    %                             plot(x_v1_bot,mid_v2_bot,'*'); 
    %                             plot(x_mid_max_v2_bot,mid_max_v2_bot,'*');
                            else
                                delta_v2(i) = mid_max_v2_bot - mid_v2h; 
    %                             plot(x_mid_max_v2_bot,mid_max_v2_bot,'*');
                            end
%     %                         plot(v1_bot(IdxMid_v1_bot),mid_v2_bot,'*'); 
%     %                         plot(x_mid_max_v2_bot_,mid_max_v2_bot,'*');
%     %                         plot(mid_v1h,mid_v2h,'*');
%                             plot(v1h_botL,v2h_botL,'*'); 
%                             plot(v1h_botR,v2h_botR,'*'); hold off; 
                        else
                            delta_v2(i) = 0;
                        end
                    end
                end
            end
            
%             [t1,~] = kmeans(delta_v1',2);
            [s1,~] = kmeans(delta_v2',2);  
%             [st3,~] = kmeans([delta_v1' delta_v2'],3);
%             [st,~] = kmeans([delta_v1' delta_v2'],2);
            
            i = 1;
            cnd = cnd_vector(i);
            while (i<N && cnd)
                i = i + 1;
%                 cnd = cnd_vector(i) * (t1(i)~=t1(1)) * (s1(i)~=s1(1));
%                 cnd = cnd_vector(i) * (st(i)~=st(1));
                cnd = cnd_vector(i) * (s1(i)~=s1(1));
            end
%             i3=1;
%             while (i3<N && cnd)
%                 i3 = i3 + 1;
%                 cnd = cnd_vector(i) * (st3(i)==st3(2));
%             end      
%             i3 = i3-1
%             k9_adj3(3) = zPoints(i3)
            
%             we want i-2 because of transition stages 
%             if (i>2)
%                 i = i-2;
%             else
%                 i = i-1;
%             end
%             % (for males it's larger because of genitals)
%             if (self.m==0)
%                 i = i-1;
%             elseif (self.m==1)
%                 if (i>4)
%                     i = i-4;
%                 else % i==2
%                     i = i-1;
%                 end
%             end
            i = i-1;
            k9_adj(3) = zPoints(i);
%             if (i>1)
%                 zPoints(i-1)
%             end
%             k9_adj = [mx_v1_bot(i),mx_v2_bot(i),zPoints(i)];
% i
% k9_adj(3)
        end
        
        function [k2, k4] = getArmpits(self)
        % Gets the armpits x and z coords
            v1 = self.v(:,1);
            v2 = self.v(:,2);
            v3 = self.v(:,3);
            
            r_side = [v1(v1(:,1)< self.crotch(1)), v3(v1(:,1)< self.crotch(1))]; %Finds right half of body
            l_side = [v1(v1(:,1)> self.crotch(1)), v3(v1(:,1)> self.crotch(1))]; %Finds left half of body

            test_k2 = NaN(3,3);
            test_k4 = NaN(3,3);
            alphas = [pi/8, pi/12, pi/24];
            for i = 1:3
                alpha = alphas(i);
                [new_rs_v1, new_rs_v3] = rotate_person(r_side(:,1), r_side(:,2), alpha); %rotates right side to apply find_minmax function 

                [new_ls_v1, new_ls_v3] = rotate_person(l_side(:,1), l_side(:,2), -alpha); %rotates left side to apply find_minmax function

                [m_V_Right, m_I_Right] = min(new_rs_v1);
                rh = [m_V_Right, new_rs_v3(m_I_Right)];  %gives rotated right hand

                [m_V_Left, m_I_Left] = max(new_ls_v1);
                lh = [m_V_Left, new_ls_v3(m_I_Left)];    %gives rotated left hand

                [n_m_x,~] = rotate_person(self.r_foot(1,1), self.r_foot(1,3), alpha);
                [n_M_x,~] = rotate_person(self.l_foot(1,1), self.l_foot(1,3), -alpha);

                [rotated_k2_1, rotated_k2_3] = find_minmax(new_rs_v1,new_rs_v3, rh(1), n_m_x, 111); %Finds right armpit rotated
                [rotated_k4_1, rotated_k4_3] = find_minmax(new_ls_v1,new_ls_v3, n_M_x, lh(1), 111); %Finds left armpit rotated

                [test_k2(i,1), test_k2(i,3)] = rotate_person(rotated_k2_1, rotated_k2_3, -alpha); %Actual right armpit
                [test_k4(i,1), test_k4(i,3)] = rotate_person(rotated_k4_1, rotated_k4_3, alpha); %Actual left armpit
            end
            
            % delete points that are not in the armpit area
                % right
            p = test_k2(:,[1,3]);
            IC = centroid(delaunay(p, Qt),1);
            dist = zeros(1,3);
            for i = 1:3
                dist(i) = norm(IC-p(i,:));
            end
            [maxDist,maxDistIdx] = max(dist);
            dist(maxDistIdx) = [];
            if mean(dist)*2<maxDist
                test_k2(maxDistIdx,:) = [];
            end
                % left
            p = test_k4(:,[1,3]);
            IC = centroid(delaunay(p, Qt),1);
            dist = zeros(1,3);
            for i = 1:3
                dist(i) = norm(IC-p(i,:));
            end
            [maxDist,maxDistIdx] = max(dist);
            dist(maxDistIdx) = [];
            if mean(dist)*2<maxDist
                test_k4(maxDistIdx,:) = [];
            end
            
            [~,max_k2Idx] = max(test_k2(:,3));
            [~,max_k4Idx] = max(test_k4(:,3));
            
            k2 = test_k2(max_k2Idx,:);
            k4 = test_k4(max_k4Idx,:);
            
            r_arm = [v1(v1(:,1)< k2(1)), v2(v1(:,1)< k2(1)) , v3(v1(:,1)< k2(1))]; %Finds right half of body
            allArmIdx = (1:length(self.v));            
            vOnLine = getVOnLine(self, self.v, k2(3), allArmIdx);
            r_arm_circ_armpit = r_arm(ismember(vOnLine,r_arm,'rows'),:);
            k2(2) = mean(max(r_arm_circ_armpit(:,2))+min(r_arm_circ_armpit(:,2)));
            
            l_arm = [v1(v1(:,1)> k4(1)), v2(v1(:,1)> k4(1)), v3(v1(:,1)> k4(1))]; %Finds right half of body
            vOnLine = getVOnLine(self, self.v, k4(3), allArmIdx);
            l_arm_circ_armpit = r_arm(ismember(vOnLine,l_arm,'rows'),:);
            k4(2) = mean(max(l_arm_circ_armpit(:,2))+min(l_arm_circ_armpit(:,2)));
        end
        
        function [rightArmpit, leftArmpit] = getArmpitsAlt(self)
        % [rightArmpit, leftArmpit] = getArmpitsAlt(self) returns the highest
        % point of the arch under the arms. Note that if the arms are
        % connected to the body, the points may need to be adjusted.
        %
        % Output:
        %   rightArmpit - [x,y,z], right armpit
        %   leftArmpit  - [x,y,z], left armpit
        
            maxv1 = max(self.v(:,1));
            minv1 = min(self.v(:,1));
            maxv3 = max(self.v(:,3));
            minv3 = min(self.v(:,3));
            
            step = (maxv1-minv1)*0.1; % step size
            
            % No need to look at head and legs
            notopIdx = self.v(:,3) < maxv3 - (maxv3-minv3)*0.15;
            nobottomIdx = self.v(:,3) > (maxv3-minv3)*0.35 + minv3;
            
            % Left side
            sideIdx = self.v(:,1) > (maxv1-minv1)*0.5*0.10;
            faces = getFaces(self.f, sideIdx & notopIdx & nobottomIdx);
            px = maxv1;
            pz = (maxv3-minv3)*0.35 + minv3;
            p = [px,pz]; % Starting point
            leftArmpit = self.getOneArmpit(faces,p,step,'left');
            
            % Right side
            sideIdx = self.v(:,1) < -(maxv1-minv1)*0.5*0.10;
            faces = getFaces(self.f, sideIdx & notopIdx & nobottomIdx);
            px = minv1;
            p = [px,pz]; % Starting point
            rightArmpit = self.getOneArmpit(faces,p,step,'right');
        end
        
        function p = getOneArmpit(self,faces,p,step,side)
        % p = getOneArmpit(self,faces,p,step,side) returns the point of one
        % of the armpits.
        %
        % Input:
        %   faces - faces to consider
        %   p - starting point
        %   step - step size
        %   side - 'right' or 'left'
        %
        % Output:
        %   p - [x,y,z], the armpit
        
            n = 20; % Number of angles
            multiplier = 1; % Step multiplier
            angle = linspace(0,7*pi/8,n);
            if strcmp(side,'left')
                angle = fliplr(angle);
            end
            
            x = reshape(self.v(faces(:,:),1), [], 3); % x values of all faces
            z = reshape(self.v(faces(:,:),3), [], 3); % z values of all faces
            
            faceArea = 0.5*(-z(:,2).*x(:,3) + z(:,1).*(-x(:,2)+x(:,3)) + x(:,1).*(z(:,2)-z(:,3)) + x(:,2).*z(:,3)); % area of all faces

            % Find point on body
            while true
                s = (z(:,1).*x(:,3) - x(:,1).*z(:,3) + (z(:,3)-z(:,1)).*p(1) + (x(:,1)-x(:,3)).*p(2))./(2*faceArea);
                t = (x(:,1).*z(:,2) - z(:,1).*x(:,2) + (z(:,1)-z(:,2)).*p(1) + (x(:,2)-x(:,1)).*p(2))./(2*faceArea);
                
                if sum(s>0 & t>0 & 1-s-t>0)>0 % check for intersection
                    if strcmp(side,'right')
                        p(1) = p(1)-step/4;
                    else
                        p(1) = p(1)+step/4;
                    end
                    break;
                end
                if strcmp(side,'right')
                    p(1) = p(1)+step/4;
                else
                    p(1) = p(1)-step/4;
                end
            end
            
            % Move up along the body
            armpitFound = false;
            while ~armpitFound
                i = 1;
                intersectAt = 0;
                nextFound = false;
                while ~nextFound
                    nextp = p + [cos(angle(i)),sin(angle(i))]*step*multiplier;

                    % s and t are used to check if nextp intersects with
                    % the person
                    s = (z(:,1).*x(:,3) - x(:,1).*z(:,3) + (z(:,3)-z(:,1)).*nextp(1) + (x(:,1)-x(:,3)).*nextp(2))./(2*faceArea);
                    t = (x(:,1).*z(:,2) - z(:,1).*x(:,2) + (z(:,1)-z(:,2)).*nextp(1) + (x(:,2)-x(:,1)).*nextp(2))./(2*faceArea);

                    if sum(s>0 & t>0 & 1-s-t>0)>0 % did intersect; go to next
                        intersectAt = i;
                        i = i + 1;
                    else % did not intersect
                        if intersectAt == 0
                            if i > 1
                                i = i - 1;
                            else
                                multiplier = multiplier*0.8;
                            end
                        else
                            nextFound = true;
                            multiplier = multiplier*0.9;
                            if i > 0 && i <= n
                                p = p + [cos(angle(i-1)),sin(angle(i-1))]*step*multiplier;
                            end
                        end
                    end
                    if i == n-1 % start over with smaller radius
                        multiplier = multiplier*0.5;
                        intersectAt = 0;
                        i = 1;
                    end
                    if multiplier <= 0.5^5
                        nextFound = true;
                        armpitFound = true;
                    end
                end
            end
            p = [p(1), NaN, p(2)];
        end
        
        function [chestCircumference] = getChestCircumference(self)
        %chest circumference
            zValue = median([self.r_armpit(3), self.l_armpit(3)]); 
            vOnLine = getVOnLine(self, self.v, zValue, self.trunkIdx);
            [chestCircumference.value,b] = getCircumference(vOnLine(:,1), vOnLine(:,2));
            chestCircumference.points = vOnLine(b,:);
            
            if(self.circ_ellipse)
                [semimajor_axis, semiminor_axis, x0, y0, ~] = ellipse_fit(vOnLine(:,1), vOnLine(:,2));
                chestCircumference.a_over_b = semiminor_axis/semimajor_axis;
                chestCircumference.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end 
            %plot_ellipse(x0,y0,semimajor_axis,semiminor_axis,vOnLine(b,:))
            
            if(self.circ_cpd)
                [chestCircumference.templatePoints,chestCircumference.templateValue] = templqte_circumference( vOnLine);
            end
        end

        function [armVertices, armVerticesIdx, faces] = armSearch(self, side)
        % Finds the vertices, their indices and the faces of arm
        % side = 'r' or 'l'
            armVertices = [];
            armVerticesIdx = [];
            faces = [];
            
            try
                if side == 'r'
                    [~,handIdx] = min(self.v(:,1));
                else
                	[~,handIdx] = max(self.v(:,1));
                end
                faces = [find(self.f(:,1) == handIdx); ...
                         find(self.f(:,2) == handIdx); ...
                         find(self.f(:,3) == handIdx)];
                armVerticesIdx = [];
                newFaces = faces; % faces to search with
                
                while(~isempty(newFaces))
                    newVerticesIdx = self.f(newFaces,:);
                    newVerticesIdx = newVerticesIdx(:);
                    newVerticesIdx = unique(newVerticesIdx);
                    newVertices = self.v(newVerticesIdx,:);
                    if side == 'r'
                        newVerticesIdx = newVerticesIdx(newVertices(:,1) < self.r_armpit(1));
                    elseif side == 'l'
                        newVerticesIdx = newVerticesIdx(newVertices(:,1) > self.l_armpit(1));
                    else
                        return
                    end
                    armVerticesIdx = [armVerticesIdx; newVerticesIdx]; % save vertices indices
                    
                    % delete duplicates
                    [newFaces,~] = find(ismember(self.f,newVerticesIdx));
                    newFaces = unique(newFaces);
                    [~, ia, ~] = intersect(newFaces, faces);
                    newFaces(ia) = [];
                    
                    faces = [faces; newFaces]; % add new faces to old faces
                end
                
                armVerticesIdx = unique(armVerticesIdx);
                armVertices = self.v(armVerticesIdx,:);
            catch
                disp('Error: armSearch has armpit_idx that is not a scalar');
            end
        end
        
        function [vIdx] = trunkSearch(self)
%             v = []; % trunk vertices
            vIdx = []; % trunk vertex indices
%             faces = []; % tunk faces
            
            % find a face of the trunk
            zMax = (self.crotch(3) + self.lShoulder(3))/2 + 0.1*(self.lShoulder(3) - self.crotch(3));
            zMin = (self.crotch(3) + self.lShoulder(3))/2 - 0.1*(self.lShoulder(3) - self.crotch(3));
            xMax = (self.rShoulder(1) + self.lShoulder(1))/2 + 0.1*(self.lShoulder(1) - self.rShoulder(1));
            xMin = (self.rShoulder(1) + self.lShoulder(1))/2 - 0.1*(self.lShoulder(1) - self.rShoulder(1));
            oneIdx = find(self.v(:,3)<zMax & self.v(:,3)>zMin & self.v(:,1)<xMax & self.v(:,1)>xMin,1);
            
            faces = [find(self.f(:,1) == oneIdx); ...
                     find(self.f(:,2) == oneIdx); ...
                     find(self.f(:,3) == oneIdx)];
            newFaces = faces; % faces to search with

            while(~isempty(newFaces))
                newVerticesIdx = self.f(newFaces,:);
                newVerticesIdx = newVerticesIdx(:);
                newVerticesIdx = unique(newVerticesIdx);
                newVertices = self.v(newVerticesIdx,:);
                
                % below shoulders and above crotch
                newVerticesIdx = newVerticesIdx(newVertices(:,3) < max(self.lShoulder(3), self.rShoulder(3)));
                newVertices = newVertices(newVertices(:,3) < max(self.lShoulder(3), self.rShoulder(3)),:);
                newVerticesIdx = newVerticesIdx(newVertices(:,3) > self.crotch(3));

                vIdx = [vIdx; newVerticesIdx]; % save vertices indices

                % delete duplicates
                [newFaces,~] = find(ismember(self.f,newVerticesIdx));
                newFaces = unique(newFaces);
                [~, ia, ~] = intersect(newFaces, faces);
                newFaces(ia) = [];

                faces = [faces; newFaces]; % add new faces to old faces
            end

            vIdx = unique(vIdx);
            vIdx(ismember(vIdx,self.rArmIdx)) = [];
            vIdx(ismember(vIdx,self.lArmIdx)) = [];
            vIdx(ismember(vIdx,self.legIdx)) = [];
%             v = self.v(vIdx,:); 
        end
        
        function [legIdx] = getLegs(self)
        % Get vertex indices of legs
            % right side
            r1 = [self.crotch(1) self.crotch(3)]; % crotch
            h1 = [self.r_hip(1) self.r_hip(3)]; % hip
            slopeR = (r1(2)-h1(2)) ./ (r1(1)-h1(1));
            y_interceptR = r1(2) - (slopeR * r1(1));
            v3 = self.v(:,3);
            v1 = self.v(:,1);
            wholeArea = v3 - (slopeR * v1) - y_interceptR;
            negAreaR = uint32(find(wholeArea<=0));
            
            % left side
            h1 = [self.l_hip(1) self.l_hip(3)];
            slopeL = (r1(2)-h1(2)) ./ (r1(1)-h1(1));
            y_interceptL = r1(2) - (slopeL * r1(1));
            wholeArea = v3 - (slopeL * v1) - y_interceptL;
            negAreaL = uint32(find(wholeArea<=0));
            
            % combine left and right
            legIdx = unique([negAreaL; negAreaR]);
            
            % we don't want parts of the arms
            armIdx = [self.lArmIdx; self.rArmIdx];
            [~, ia, ~] = intersect(legIdx, armIdx);
            legIdx(ia) = [];
        end
        
        function [rThighGirth, lThighGirth,...
                  rthigh_front,rthigh_back,rthigh_lateral,rthigh_medial,...
                  lthigh_front,lthigh_back,lthigh_lateral,lthigh_medial] = getThighGirth(self)
            v1 = self.v(:,1);
            rLegIdx = self.legIdx(v1(self.legIdx) < self.crotch(1,1));
            lLegIdx = self.legIdx(v1(self.legIdx) >= self.crotch(1,1));
            
            dist = self.r_hip(3) - self.r_ankle(3);
            zValue = 0.75*dist + self.r_ankle(3);
            vOnLine = getVOnLine(self, self.v, zValue, rLegIdx);
            [rThighGirth.value,b] = getCircumference(vOnLine(:,1), vOnLine(:,2));
            rThighGirth.points = vOnLine(b,:);
            
            [~,idx] = min(rThighGirth.points(:,2)); rthigh_front   = rThighGirth.points(idx,:);
            [~,idx] = max(rThighGirth.points(:,2)); rthigh_back    = rThighGirth.points(idx,:);
            [~,idx] = min(rThighGirth.points(:,1)); rthigh_lateral = rThighGirth.points(idx,:);
            [~,idx] = max(rThighGirth.points(:,1)); rthigh_medial  = rThighGirth.points(idx,:);
            
            if(self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(vOnLine(:,1), vOnLine(:,2));
                rThighGirth.a_over_b = semiminor_axis/semimajor_axis;
                rThighGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end 
             
            if(self.circ_cpd)
                [rThighGirth.templatePoints,rThighGirth.templateValue] = template_circumference(vOnLine);
            end
            
            dist = self.l_hip(3) - self.l_ankle(3);
            zValue = 0.75*dist + self.l_ankle(3);
            vOnLine = getVOnLine(self, self.v, zValue, lLegIdx);
            [lThighGirth.value,b] = getCircumference(vOnLine(:,1), vOnLine(:,2));
            lThighGirth.points = vOnLine(b,:);
            
            [~,idx] = min(lThighGirth.points(:,2)); lthigh_front   = lThighGirth.points(idx,:);
            [~,idx] = max(lThighGirth.points(:,2)); lthigh_back    = lThighGirth.points(idx,:);
            [~,idx] = min(lThighGirth.points(:,1)); lthigh_medial  = lThighGirth.points(idx,:);
            [~,idx] = max(lThighGirth.points(:,1)); lthigh_lateral = lThighGirth.points(idx,:);

            
            if (self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(vOnLine(:,1), vOnLine(:,2));
                lThighGirth.a_over_b = semiminor_axis/semimajor_axis;
                lThighGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            
            if(self.circ_cpd)
                [lThighGirth.templatePoints,lThighGirth.templateValue] = template_circumference(vOnLine);
            end
        end
        
        function [headIdx] = getHead(self)
        % Get vertex indices of head
            headIdx = [];
            
            try
            maxShoulderZ = max(self.rShoulder(3), self.lShoulder(3));
            headIdx = uint32(find(self.v(:,3) > maxShoulderZ));
            
            % we don't want parts of the arms
            armIdx = [self.lArmIdx; self.rArmIdx];
            [~, ia, ~] = intersect(headIdx, armIdx);
            headIdx(ia) = [];
            catch
                disp('Error: headIdx not found');
            end
        end
        
        function [lShoulder, rShoulder] = getShoulders(self) 
            lArmVertices = self.v(self.lArmIdx,:);
            [~, lShoulderZIdx] = max(lArmVertices(:,3));
            lShoulder = lArmVertices(lShoulderZIdx,:);
            
            rArmVertices = self.v(self.rArmIdx,:);
            [~, rShoulderZIdx] = max(rArmVertices(:,3));
            rShoulder = rArmVertices(rShoulderZIdx,:);
        end
        
        function [leftArmLength, rightArmLength,armMaxR, armMaxL] = getArmLength(self)
            leftArmLength = 0;
            rightArmLength = 0;
            try
                armMaxR = [(self.r_armpit(1)+self.rShoulder(1))/2,(self.r_armpit(3)+self.rShoulder(3))/2];
                armMaxL = [(self.l_armpit(1)+self.lShoulder(1))/2,(self.l_armpit(3)+self.lShoulder(3))/2];
                
                wristR = self.r_wrist;
                wristL = self.l_wrist;
                
                rightArmLength = sqrt((armMaxR(1,1)-wristR(1,1))^2+(armMaxR(1,2)-wristR(1,3))^2);
                leftArmLength = sqrt((armMaxL(1,1)-wristL(1,1))^2+(armMaxL(1,2)-wristL(1,3))^2);
            catch
                disp('Error: arm lengths not possible');
            end
        end
        
        function [collar] = getCollar(self)
            collar = (self.lShoulder+self.rShoulder)/2;
        end
        
        function [collarScalpLength] = getCollarScalpLength(self)
            v1 = self.v(:,1);
            v3 = self.v(:,3);
            [HeadZ, Head_I] = max(v3);
            HeadX = v1(Head_I);
            collarScalpLength = sqrt((HeadX - self.collar(1,1))^2 + (HeadZ - self.collar(1,2))^2);
        end
        
        function [trunkLength] = getTrunkLength(self)
            trunkLength = sqrt((self.crotch(1)-self.collar(1,1))^2 + (self.crotch(3)-self.collar(1,2))^2);
        end
        
        function [lLegLength, rLegLength] = getLegLength(self)
            % 'New Code'
            V = self.v;
%             rlegMaxZ = (self.k9(3)+self.k3(3))/2;
%             llegMaxZ = (self.k9(3)+self.k6(3))/2;
            rlegMaxZ = self.r_hip(3);
            llegMaxZ = self.l_hip(3);
            rnewV=V(V(:,1)>0,:);
            lnewV=V(V(:,1)<0,:);            
            [rlegMinZ, rlegMinId] = min(rnewV(:,3));
            [llegMinZ, llegMinId] = min(lnewV(:,3));
            rlegMinX=rnewV(rlegMinId,1);
            llegMinX=lnewV(llegMinId,1);
%             rlegMinZ = self.r_ankle(:,3);
%             llegMinZ = self.l_ankle(:,3);
%             rlegMinX=self.r_ankle(:,1);
%             llegMinX=self.l_ankle(:,1);

            cMinX=(rlegMinX-llegMinX)/2;
            deltaX=rlegMinX-cMinX;
            ldeltaZ=llegMaxZ - llegMinZ;
            rdeltaZ=rlegMaxZ - rlegMinZ;
            lLegLength = sqrt(deltaX^2+ldeltaZ^2);
            rLegLength = sqrt(deltaX^2+rdeltaZ^2);
            
           % Old Code
%             legMaxZ = (self.k9(3)+self.k3(3))/2;
%             legMinZ = min(self.v(:,3));
%             rLegLength = legMaxZ - legMinZ;
%             lLegLength = rLegLength;
        end
        
        function [crotchHeight] = getCrotchHeight(self)
            crotchHeight = self.crotch(3) - min(self.v(:,3));
        end
           
        function [trunkIdx] = getTrunk(self)
        % Get vertex indices of trunk
            trunkIdx = (uint32(1:size(self.v,1)))';
            nonTrunkIdx = [self.lArmIdx; self.rArmIdx; ...
                           self.legIdx; self.headIdx];
            trunkIdx(nonTrunkIdx) = [];
        end
        
        function [circumference, end_points] = getHip(self)
        % For crotch and armpit points, may have to consider cases where
     
            % get all vertices without arms
            armIdx = [self.lArmIdx; self.rArmIdx];
            keepIdx = 1:length(self.v);
            keepIdx(armIdx) = [];
            
            zStart = self.crotch(3);
            zEnd = (self.r_armpit(3)+self.crotch(3))/2;
            
            % Slice once then find the max slice
            % then slice that slice again
            
            [x,y,z] = slice_n_dice(self, 3, 10, zStart,zEnd, keepIdx);

            % Finds end_points
            [x_min, x_min_idx] = min(x); %mins give right side point, maxes give left side point
            [x_max, x_max_idx] = max(x);
            
%                 [y_backPoint,indx_backPoint] = max(y);
%                 x_backPoint = x(indx_backPoint);
%                 z_backPoint = z(indx_backPoint);
            back_y = y(y>0);
            back_x = x(y>0);
            back_z = z(y>0);
            
            [~,idx_min_xDistCrHip]=min(abs(abs(back_x)-abs(self.crotch(1))));
            x_backPoint = back_x(idx_min_xDistCrHip);
            y_backPoint = back_y(idx_min_xDistCrHip);
            z_backPoint = back_z(idx_min_xDistCrHip);
                
            y_min = y(x_min_idx);
            y_max = y(x_max_idx);
            mnZ = mean(z);
            end_points = [x_min, y_min(1), mnZ; x_max, y_max(1), mnZ;x_backPoint,y_backPoint,z_backPoint];
            
            % Finds circumference of hip
            [circumference.value,b] = getCircumference(x,y);
            circumference.points = [x(b) y(b) z(b)];
            
            if(self.circ_ellipse)
            [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(x,y);
            circumference.a_over_b = semiminor_axis/semimajor_axis;
            circumference.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            
            if(self.circ_cpd)
                [circumference.templatePoints,circumference.templateValue] = template_circumference( vOnLine);
            end
        end
        
        function [circumference, end_points] = getWaist(self)
            z_mid = mean([self.r_armpit(3),self.r_hip(3)]); 
            vOnLine = getVOnLine(self, self.v, z_mid, self.trunkIdx);
            [circumference.value,b] = getCircumference(vOnLine(:,1), vOnLine(:,2));
            circumference.points = [vOnLine(b,1),vOnLine(b,2),vOnLine(b,3)];
            
            if (self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(vOnLine(:,1), vOnLine(:,2));
               circumference.a_over_b = semiminor_axis/semimajor_axis;
               circumference.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(self.circ_cpd)
                [circumference.templatePoints,circumference.templateValue] = template_circumference( vOnLine);
            end            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Finds end_points
            [x_min, x_min_idx] = min(vOnLine(:,1)); %mins give right side point, maxes give left side point
            [x_max, x_max_idx] = max(vOnLine(:,1));
            y_min = vOnLine(x_min_idx,2);
            y_max = vOnLine(x_max_idx,2);
            mnZ = mean(vOnLine(:,3));
            end_points = [x_min, y_min(1), mnZ; x_max, y_max(1), mnZ];            
        end    
        
        function nose_tip = getNoseTip(self)
            v_aboveShoulder = self.v(self.v(:,3)>max(self.rShoulder(3),self.lShoulder(3)),:);
            [val_HighestShoulder,idx_HighestShoulder] = min(v_aboveShoulder(:,3));
            [~,idx_TipHead] = max(v_aboveShoulder(:,3));
            %[~,idx_HighestShoulder] = max(self.rShoulder(3),self.lShoulder(3));
            HeadShoulder_dist = v_aboveShoulder(idx_TipHead,:)-v_aboveShoulder(idx_HighestShoulder,:);
            %HeadShoulder_dist = self.v(max(self.v(:,3)),:)-self.v(self.v(:,3)==max(self.rShoulder(3),self.lShoulder(3)),:);
           % HeadShoulder_dist = self.v(max(self.v(:,3),2),:)-v_aboveShoulder(idx_HighestShoulder,:);
            %dist = sqrt(HeadShoulder_dist(:,1)^2+HeadShoulder_dist(:,2)^2+HeadShoulder_dist(:,3)^2);
            dist = v_aboveShoulder(idx_TipHead,3)-v_aboveShoulder(idx_HighestShoulder,3);

            tmp = v_aboveShoulder(v_aboveShoulder(:,3)>val_HighestShoulder+dist*0.3,:);
            tmp = tmp(tmp(:,3)<val_HighestShoulder+0.6*dist,:);
            
            [y_noseTip,idx_noseTip] = min(tmp(:,2));
            x_noseTip = tmp(idx_noseTip,1);
            z_noseTip = tmp(idx_noseTip,3);
            nose_tip = [x_noseTip,y_noseTip,z_noseTip];
        end  
        
        function [rtoe_tip, rheel_tip, ltoe_tip, lheel_tip] = getFeetTips(self)
            
            % Right Foot
            v_under_rAnkle = self.v(self.v(:,3)<self.r_ankle(3),:);
            v_under_rAnkle = v_under_rAnkle(v_under_rAnkle(:,1)<0,:);
            
            % Right Heel
            [~,Idx_y_rheel_tip] = max(v_under_rAnkle(:,2));
            rheel_tip = v_under_rAnkle(Idx_y_rheel_tip,:);
            
            % Right Toe Tip
            alpha = -pi/8;
            v_under_rAnkle_rotated = v_under_rAnkle;
            [v_under_rAnkle_rotated(:,1),v_under_rAnkle_rotated(:,2)] = rotate_person(v_under_rAnkle(:,1),v_under_rAnkle(:,2),alpha);
            [~,Idx_y_rtoe_tip] = min(v_under_rAnkle_rotated(:,2));
            rtoe_tip = v_under_rAnkle(Idx_y_rtoe_tip,:);
            
            % Left Foot
            v_under_lAnkle = self.v(self.v(:,3)<self.l_ankle(3),:);
            v_under_lAnkle = v_under_lAnkle(v_under_lAnkle(:,1)>0,:);
            
            % Left Heel
            [~,Idx_y_lheel_tip] = max(v_under_lAnkle(:,2));
            lheel_tip = v_under_lAnkle(Idx_y_lheel_tip,:);
            
            % Left Toe Tip
            alpha = pi/8;
            v_under_lAnkle_rotated = v_under_lAnkle;
            [v_under_lAnkle_rotated(:,1),v_under_lAnkle_rotated(:,2)] = rotate_person(v_under_lAnkle(:,1),v_under_lAnkle(:,2),alpha);
            [~,Idx_y_ltoe_tip] = min(v_under_lAnkle_rotated(:,2));
            ltoe_tip = v_under_lAnkle(Idx_y_ltoe_tip,:);
        end
        
        function bodyType = getBodyType(self)
            hips = self.hipCircumference.value;
            waist = self.waistCircumference.value;
            chest = self.chestCircumference.value;
            if (waist < hips)
                if (waist < chest)
                    bodyType = 'Hourglass';
                else
                    bodyType = 'Triangle';
                end
            else
                if (waist > chest)
                    bodyType = 'Round';
                else
                    bodyType = 'Inverted Triangle';
                end
            end
        end
            
        
        function [surface] = getWBSurfaceArea(self)
            v1 = self.v(self.f(:,1),:);
            v2 = self.v(self.f(:,2),:);
            v3 = self.v(self.f(:,3),:);
            surface.total = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
        end
        
        function [volume,surface] = getSurfaceArea(self)
            v1 = self.v(self.f(:,1),:);
            v2 = self.v(self.f(:,2),:);
            v3 = self.v(self.f(:,3),:);
            volume.total = sum(SignedVolumeOfTriangle(v1,v2,v3));
            surface.total = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
            
            %self.f = fixFaceOrientation(self.f,self.v);

            
            %% if we want to get only the SA: 
            
            [surface.trunk] = getSurfaceAreaPartial(self,self.trunkIdx,0);
            [surface.lleg] = getSurfaceAreaPartial(self,self.legIdx,1);
            [surface.rleg] = getSurfaceAreaPartial(self,self.legIdx,2);
            [surface.legs] = getSurfaceAreaPartial(self,self.legIdx,3);
            [surface.head] = getSurfaceAreaPartial(self,self.headIdx,4);
            [surface.rArm] = getSurfaceAreaPartial(self,self.lArmIdx,5);
            [surface.lArm] = getSurfaceAreaPartial(self,self.lArmIdx,6);
            
        end
        
        function [volume,surface] = getSurfaceAreaAndVolume(self)
            v1 = self.v(self.f(:,1),:);
            v2 = self.v(self.f(:,2),:);
            v3 = self.v(self.f(:,3),:);
            volume.total = sum(SignedVolumeOfTriangle(v1,v2,v3));
            surface.total = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
            
            %self.f = fixFaceOrientation(self.f,self.v);
     
            %% for Vol and SA: 
            [test_vol.lArm,volume.lArm, surface.lArm] = getSurfaceAreaAndVolumePartial(self,self.lArmIdx,6,volume.total);
            [test_vol.rArm,volume.rArm, surface.rArm] = getSurfaceAreaAndVolumePartial(self,self.rArmIdx,5,volume.total);
            [test_vol.head,volume.head, surface.head] = getSurfaceAreaAndVolumePartial(self,self.headIdx,4,volume.total);
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            %%% this is incorrect:
            lLegIdx = self.legIdx(self.v(self.legIdx,1)>self.crotch(1));
            rLegIdx = self.legIdx(self.v(self.legIdx,1)<=self.crotch(1));            
             [test_vol.lleg,volume.lleg, surface.lleg] = getSurfaceAreaAndVolumePartial(self,self.legIdx,1,volume.total); % mistake of face closing to the other hople
             [test_vol.rleg,volume.rleg, surface.rleg] = getSurfaceAreaAndVolumePartial(self,self.legIdx,2,volume.total);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% this will give you correct volume for both legs together:
            [test_vol.legs, volume.legs, surface.legs] = getSurfaceAreaAndVolumePartial(self,self.legIdx,3,volume.total);
            [surface.lleg] = getSurfaceAreaPartial(self,self.legIdx,1);
            [surface.rleg] = getSurfaceAreaPartial(self,self.legIdx,2);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [~,~, surface.trunk] = getSurfaceAreaAndVolumePartial(self,self.trunkIdx,0,volume.total);
            volume.trunk = volume.total - volume.lArm - volume.rArm - volume.head - volume.legs; %- abs(volume.rleg)- abs(volume.lleg);%    
            
        end
        
        
        function [surface] = getSurfaceAreaPartial(self,indices,isleg)
            
            %self.f = fixFaceOrientation(self.f,self.v);
            
            [faces,~] = getFaces(self.f,indices);
            v1 = self.v(faces(:,1),:);
            v2 = self.v(faces(:,2),:);
            v3 = self.v(faces(:,3),:);
            surface = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
            LegCrotchIdx=[];
            
            if isleg ~=0
                if (isleg==1)
                    LegCrotchIdx = indices(self.v(indices,1)>self.crotch(1));
                    [faces_surface,~] = getFaces(self.f,LegCrotchIdx);
                    v1 = self.v(faces_surface(:,1),:);
                    v2 = self.v(faces_surface(:,2),:);
                    v3 = self.v(faces_surface(:,3),:);
                    surface = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
                elseif (isleg==2)
                    LegCrotchIdx = indices(self.v(indices,1)<=self.crotch(1));
                    [faces_surface,~] = getFaces(self.f,LegCrotchIdx);
                    v1 = self.v(faces_surface(:,1),:);
                    v2 = self.v(faces_surface(:,2),:);
                    v3 = self.v(faces_surface(:,3),:);
                    surface = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
                end
            
            end
        end 
        
        function [test_vol, volume,surface] = getSurfaceAreaAndVolumePartial(self,indices,isleg,TotalBodyVolume)

            %self.f = fixFaceOrientation(self.f,self.v);
            
            [faces,~] = getFaces(self.f,indices);
            v1 = self.v(faces(:,1),:);
            v2 = self.v(faces(:,2),:);
            v3 = self.v(faces(:,3),:);
            surface = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
            
            LegCrotchIdx=[];
            
            if isleg ~=0
                if (isleg==1)
                    LegCrotchIdx = indices(self.v(indices,1)>self.crotch(1));
                    [faces_surface,~] = getFaces(self.f,LegCrotchIdx);
                    v1 = self.v(faces_surface(:,1),:);
                    v2 = self.v(faces_surface(:,2),:);
                    v3 = self.v(faces_surface(:,3),:);
                    surface = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
                elseif (isleg==2)
                    LegCrotchIdx = indices(self.v(indices,1)<=self.crotch(1));
                    [faces_surface,~] = getFaces(self.f,LegCrotchIdx);
                    v1 = self.v(faces_surface(:,1),:);
                    v2 = self.v(faces_surface(:,2),:);
                    v3 = self.v(faces_surface(:,3),:);
                    surface = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
                end
                
                if (~isempty(LegCrotchIdx))
                    AllEdges = getBoundaryEdges(faces);
                    AllEdgesIdx = ([AllEdges(:,1);AllEdges(:,2)]);
                    [facesLeg,~] = getFaces(self.f,LegCrotchIdx);
                    bdyEdges = getBoundaryEdges(facesLeg);
                    
                    TopHole101Idx = ismember(sort(bdyEdges,2),sort(AllEdges,2));
                    
                    SetOfVert_TopHole =  bdyEdges(TopHole101Idx(:,1)==1,:);
                    [EdgesRep,SortedSetOfVert] = hist(double([SetOfVert_TopHole(:,1);SetOfVert_TopHole(:,2)]),unique(double([SetOfVert_TopHole(:,1);SetOfVert_TopHole(:,2)])));
					SortedSetOfVert = SortedSetOfVert';
                    StartEndVert = SortedSetOfVert(EdgesRep==1);
                    faces = facesLeg;
                    v1 = self.v(faces(:,1),:);
                    v2 = self.v(faces(:,2),:);
                    v3 = self.v(faces(:,3),:);
                end
                
                %[list,bdyEdges,~,~] = getListOfHoles(faces,self.v);
                
                bdyEdges = getBoundaryEdges(faces);
                
                list = zeros(1,length(bdyEdges));
                list_old = list;
                holeNum = 1;
                row = 1;
                col = 1;
                if (~isempty(bdyEdges))
                    list(row) = holeNum;
                    start = row;
                    while true
                        col = mod(col,2)+1; % get other col
                        % Find next edge
                        [a,b] = find(bdyEdges == bdyEdges(row,col));
                        if sum([a(1),b(1)] == [row,col]) == 2
                            row = a(2);
                            col = b(2);
                        else
                            row = a(1);
                            col = b(1);
                        end
                        % Update list
                        if row ~= start
                            list(row) = holeNum;
                        else
                            % if row == start, continue with next hole
                            holeNum = holeNum + 1;
                            col = 1;
                            row = find(list==0, 1); % next row not assigned to a hole
                            if ~isempty(row)
                                list(row) = holeNum;
                                start = row;
                            else
                                break; % Break if no edge is left
                            end
                        end
                        if(sum(list == 0) == sum(list_old == 0))
                            bdyEdges(list==0,:) = [];
                            list(list == 0) = [];
                            [~,loc]=min([sum(list == 1),sum(list == 2)]);
                            bdyEdges(list==loc,:) = [];
                            list = zeros(1,length(bdyEdges));
                            holeNum = 1;
                            row = 1;
                            col = 1;
                            list(row) = holeNum;
                            start = row;
                        end
                        list_old = list;
                    end
                end
                
                
                vA_hatL = cell(1,max(list));
                for i = 1:max(list)
                    holeEdges = bdyEdges(list==i,:);
                    
                    indx_r = 1;
                    
                    find_vIdx_edge1 = ismember(self.f,holeEdges(1,1)) + ismember(self.f,holeEdges(1,2));
                    find_edge1 = sum(find_vIdx_edge1,2);
                    [~,idx_edge1] = ismember(2,find_edge1);
                    edge1 = self.f(idx_edge1,:);
                    idx_101 = ismember(edge1,holeEdges(1,:));
                    if (idx_101(1)==idx_101(2))
                        ver1 = edge1(2);
                        ver2 = edge1(1);
                        %    closing_faces = [self.f(idx_edge1,2),self.f(idx_edge1,1),c];
                    elseif(idx_101(3)==idx_101(2))
                        ver1 = edge1(3);
                        ver2 = edge1(2);
                        %     closing_faces = [self.f(idx_edge1,3),self.f(idx_edge1,2),c];
                    else
                        ver1 = edge1(1);
                        ver2 = edge1(3);
                        %     closing_faces = [self.f(idx_edge1,1),self.f(idx_edge1,3),c];
                    end
                    holeEdges(1,:)=[ver1 ver2];
                    srtd_i = holeEdges;
                    
                    for j = 1:size(holeEdges,1)-1
                        
                        temp = [holeEdges(1:indx_r-1,:);[NaN NaN];holeEdges(indx_r+1:end,:)];
                        [indx_r,indx_c] = find(temp==srtd_i(j,2));
                        srtd_i(j+1,:) = holeEdges(indx_r,:);
                        
                        if (indx_c == 2)
                            srtd_i(j+1,:) = flip(srtd_i(j+1,:));
                        end
                    end
                    if (srtd_i(end,1)==srtd_i(1,1))
                        srtd_i(end,:) = flip(srtd_i(end,:));
                    end
                    vA_hatL{1,i} = zeros(size(srtd_i,1),3);
                    vA_hatL{1,i}(:,1:2) = srtd_i ;
                end
                
                if (~isempty(LegCrotchIdx))
                    hole = vA_hatL{1,1};	
                    [~,OrderStartEnd] = ismember(hole,StartEndVert);
                    if (sum(OrderStartEnd(:,1)==2)==1)
                        StartEndVert_tmp(1) = StartEndVert(2);
                        StartEndVert_tmp(2) = StartEndVert(1);
                        StartEndVert = StartEndVert_tmp;
                        clear StartEndVert_tmp
                    end
                    
                    Idx_Start = hole(ismember(hole(:,1),StartEndVert(1,1))==1,1);
                    Idx_End = hole(ismember(hole(:,2),StartEndVert(1,2))==1,2);
					hole_startIdx = getIndex(Idx_Start, hole);
					hole_endIdx = getIndex(Idx_End,hole);
                    if hole_startIdx<hole_endIdx
                        HoleLeg = [StartEndVert(1,2),StartEndVert(1,1);hole(hole_startIdx:hole_endIdx-1,[1:2])];
                    else
                        HoleLeg = [StartEndVert(1,2),StartEndVert(1,1);hole([hole_startIdx:end,1:hole_endIdx-1],[1:2])];
                    end
                    hole(hole_startIdx+1,[1,2])=StartEndVert;
                    HoleTop_idx = ismember(hole(:,[1:2]),HoleLeg);
                    HoleTop_idx = HoleTop_idx (:,2);
                    HoleTop(1,[1,2]) = [StartEndVert(1),StartEndVert(2)];
                    for l = 1:size(HoleTop_idx)
                        if HoleTop_idx(l)==0
                            HoleTop = [HoleTop;hole(l,[1,2])];
                        end
                    end
                    
                    vA_hatL{1,1} = HoleLeg(:,[1,2]);
                    vA_hatL{1,2} = HoleTop(:,[1,2]);
                    
                    if isleg == 2 || isleg == 1
                        HoleTop(end+1,2) = HoleTop(1,1);
                        HoleTop(end,1) = NaN;
%                         HoleTop(end+1,1) = HoleTop(1,2);
%                         HoleTop(end,2) = NaN;
                        HoleTopUniquetmp=unique(HoleTop);
                        
                        for m = 1:size(HoleTopUniquetmp)
                            u=ismember(HoleTop,HoleTopUniquetmp(m));
                            occurence = sum(sum(u));
                            HoleTopUniquetmp(m,2) = occurence;
                        end
                        
                        uniqueVertex = find(HoleTopUniquetmp(:,2)==1);
                        HoleTop(end,1) = HoleTopUniquetmp(uniqueVertex,1);
%                         UniqueVertexIdx = HoleTopUniquetmp(uniqueVertex,1);
%                         SumForPosition = sum(ismember(HoleTop,UniqueVertexIdx(1,1)),1);
%                         if SumForPosition(1,1)==1
%                             HoleTop(end,2)=UniqueVertexIdx(1,1);
%                             HoleTop(end-1,1)=UniqueVertexIdx(2,1);
%                         else
%                             HoleTop(end,2)=UniqueVertexIdx(2,1);
%                             HoleTop(end-1,1)=UniqueVertexIdx(1,1);
%                         end
                        vA_hatL{1,2} = HoleTop(:,[1,2]);
                    end
                end
                
                center = zeros(size(vA_hatL,2),3);
                
                for k=1:size(vA_hatL,2)
                    srtd_i = vA_hatL{1,k};
                    vert1 = self.v(srtd_i(:,1),:);
                    vert2 = self.v(srtd_i(:,2),:);
                    center(k,1) = (max(vert1(:,1)) + min(vert1(:,1)))/2;
                    center(k,2) = (max(vert1(:,2)) + min(vert1(:,2)))/2;
                    center(k,3) = (max(vert1(:,3)) + min(vert1(:,3)))/2;
                    
                    if isleg==3 % legs
                        v1 = [v1; vert1];
                        v2 = [v2; vert2];
                        
                    elseif  (isleg==1) %leftleg
                        v1 = [v1; vert2];
                        v2 = [v2; vert1];
                        
                    elseif  (isleg==2) %rightleg
                        v1 = [v1; vert1];
                        v2 = [v2; vert2];
                        
                    elseif  (isleg==4) %head
                        v1 = [v1; vert2];
                        v2 = [v2; vert1];
                    else
                        v1 = [v1; vert2];
                        v2 = [v2; vert1];
                        
                    end
                    v3 = [v3; repmat(center(k,:),length(vert1),1)];
                end
                
                vv = [self.v;center(1,:)];
                vv_first = [self.v;center(1,:)];
                
                %%% plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                %figure;
                ff = [faces;[vA_hatL{1,1}(:,[1,2]) length(vv)*ones(length(vA_hatL{1,1}),1)]];
                % hold on; patch('vertices',vv,'faces',ff,'FaceColor', 'y');
                
                if size(vA_hatL,2)>1
                    vv = [self.v;center(1,:);center(2,:)];
                    ff=[faces;[vA_hatL{1,2}(:,[1,2]) length(vv)*ones(length(vA_hatL{1,2}),1)]];
                    %  hold on; patch('vertices',vv,'faces',ff, 'FaceColor', 'y');
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                %             f_tmp = [faces;[vA_hatL{1,1}(:,[1,2]) length(vv)*ones(length(vA_hatL{1,1}),1)]];
                %             if size(vA_hatL,2)>1
                %                 f_tmp = [faces;[vA_hatL{1,1}(:,[1,2]) length(vv)*ones(length(vA_hatL{1,1}),1)];[vA_hatL{1,2}(:,[1,2]) length(vv)*ones(length(vA_hatL{1,2}),1)]];
                %             end
                
                %%%%%%%%%%%%%%%%%%%%% SIMA
                
                new_faces1 = [vA_hatL{1,1}(:,[1,2]) length(vv_first)*ones(length(vA_hatL{1,1}),1)];
                
                %%% checking the oeirntation of the newly created faces and fixing it
                %             [w,~]=ismember(faces,new_faces(3,[1,2]));
                %%%%% Sima v1
                %             edges = [faces(:,[1,2]);faces(:,[2,3]);faces(:,[3,1])];
                %             wOrt_faces = ismember(new_faces(:,[1,2]),edges,'rows')==1;
                %             new_faces(wOrt_faces,[1,2])=new_faces(wOrt_faces,[2,1]);
                %%%%% Sima v2
                edges = [faces(3,[1,2]);faces(3,[2,3]);faces(3,[3,1])];
                flip_newF = ismember(new_faces1(:,[1,2]),edges,'rows')==1;
                new_faces1(:,[1,2])=new_faces1(:,[2,1]);
                %%%%%%%%%%%%%%
                %             sumU = sum(w,2);
                %             [~,e]=ismember(2,sumU);
                %             [~,q]=ismember(new_faces(3,1),faces(e,:));
                %             [~,g]= ismember(new_faces(3,2),faces(e,:));
                %             if q+1 == g || g == q-2
                %                 new_faces_tmp = [new_faces(:,2),new_faces(:,1),new_faces(:,3)];
                %                 new_faces = new_faces_tmp;
                %             end
                
                f_tmp = [faces; new_faces1];
                new_faces2 = [];
                
                if size(vA_hatL,2)>1
                    if (isleg==1 || isleg==2)
                        new_faces2 = [vA_hatL{1,2}(:,[1,2]) length(vv)*ones(length(vA_hatL{1,2}),1)];
                        
                        %%%% checking the orientation of the new faces
                        %                     [w,~]=ismember(faces,new_faces2(3,[1,2]));
                        %                     sumU = sum(w,2);
                        %                     [~,e]=ismember(2,sumU);
                        %                     [~,q]=ismember(new_faces2(3,1),faces(e,:));
                        %                     [~,g]= ismember(new_faces2(3,2),faces(e,:));
                        %                     if q+1 == g || g == q-2
                        %                         new_faces_tmp = [new_faces2(:,2),new_faces2(:,1),new_faces2(:,3)];
                        %                         new_faces2 = new_faces_tmp;
                        %                     end
                        edges = [faces(3,[1,2]);faces(3,[2,3]);faces(3,[3,1])];
                        flip_newF = ismember(new_faces2(:,[1,2]),edges,'rows')==1;
                        new_faces2(:,[1,2])=new_faces2(:,[2,1]);
                        
                        %  new_faces = [new_faces(2:end,:);new_faces2(2:end,:)];
                        new_faces = [new_faces1;new_faces2];
                    else
                        new_faces = [[vA_hatL{1,1}(:,[1,2]) length(vv)*ones(length(vA_hatL{1,1}),1)];[vA_hatL{1,2}(:,[1,2]) length(vv)*ones(length(vA_hatL{1,2}),1)]];
                    end
                    
                    f_tmp = [faces;new_faces];
                end
                
                %figure
                %patch('vertices', vv, 'faces', [faces;new_faces1], 'FaceColor', 'y');
                
                %%%%%%%%%%%%%%%%%%%%%%%%
                
                %  bdyEdges = getBoundaryEdges(f_tmp);
                
                %             X = [v1(:,1);v2(:,1);v3(:,1)];
                %             Y = [v1(:,2);v2(:,2);v3(:,2)];
                %             Z = [v1(:,3);v2(:,3);v3(:,3)];
                
                
                % [TriIdx, volume] = convhull(100+X,100+Y,100+Z);
                %vv = vv;
                
                
                %%%%%%%%%%%plotting
                %              if isleg == 1
                %                 figure; patch('vertices',vv,'faces',faces,'FaceColor', 'y');
                %                 hold on; patch('vertices',vv,'faces',new_faces,'FaceColor', 'm');
                %                 hold on; patch('vertices',vv,'faces',new_faces2,'FaceColor', 'g');
                %                 %hold on; patch('vertices',vv,'faces',f_tmp);
                %
                %              elseif isleg == 2
                %                 figure; patch('vertices',vv,'faces',faces,'FaceColor', 'y');
                %                 hold on; patch('vertices',vv,'faces',new_faces,'FaceColor', 'm');
                %                 hold on; patch('vertices',vv,'faces',new_faces2,'FaceColor', 'b');
                %                 %hold on; patch('vertices',vv,'faces',f_tmp);
                %
                %              end
                %%%%%%%%%%%%
                
                %              f_tmp = fixFaceOrientation2(f_tmp,vv);
                
                volume = vol_calc( vv, f_tmp, 1000000);
                
                %%%% may need to do the same for legs???
                
                %if  volume > (TotalBodyVolume)
                if ~(isleg ==1 || isleg ==2)
                    f_tmp1 = [faces; new_faces1(:,[2,1,3])];
                    volume1 = vol_calc( vv, f_tmp1, 1000000 );
                    [volumeShift,vol_case] = min([volume1,volume]);
                    test_vol = volumeShift;
                    switch vol_case
                        case 1
                            volumeNonShift = vol_calc( vv, f_tmp1, 0 );
                        case 2
                            volumeNonShift = vol_calc( vv, f_tmp, 0 );
                    end
                    %VolDifference  = [abs(volumeNonShift-volume1),abs(volumeNonShift-volume)];
                    [~,volDiff_case] = min([abs(volumeNonShift-volume1),abs(volumeNonShift-volume)]);
                    
                    switch volDiff_case
                        case 1
                            volume = volume1;
                        case 2
                            volume = volume;
                    end
                    
                else
                    % 1 flipped
                    f_tmp1 = [faces;new_faces1(:,[2,1,3]);new_faces2];
                    volume1 = vol_calc( vv, f_tmp1, 1000000 );
                    
                    % 2 flipped
                    f_tmp2 = [faces;new_faces1;new_faces2(:,[2,1,3])];
                    volume2 = vol_calc( vv, f_tmp2, 1000000 );
                    
                    % 1&2 flipped
                    f_tmp3 = [faces;new_faces1(:,[2,1,3]);new_faces2(:,[2,1,3])];
                    volume3 = vol_calc( vv, f_tmp3, 1000000 );
                    
                    [volume,vol_case] = min([volume1,volume2,volume3,volume]);
                    
                    test_vol = volume;
                    switch vol_case
                        case 1
                            volume = vol_calc( vv, f_tmp1, 0 );
                        case 2
                            volume = vol_calc( vv, f_tmp2, 0 );
                        case 3
                            volume = vol_calc( vv, f_tmp3, 0 );
                        case 4
                            volume = vol_calc( vv, f_tmp, 0 );
                    end
                    
                end
                %                  else
                %                      test_vol = volume;
                %                      volume = vol_calc( vv, f_tmp, 0 );
                %                  end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
            else
                volume = NaN;
                test_vol = NaN;
            end
            
            
            %volume = abs(sum(SignedVolumeOfTriangle(v1,v2,v3)));
            
            %             if isleg == 4 % head
            %                 if volume > (0.2*TotalBodyVolume)
            %                     v1 = [self.v(faces(:,1),:); vert1];
            %                     v2 = [self.v(faces(:,2),:); vert2];
            %                     volume = abs(sum(SignedVolumeOfTriangle(100+v1,100+v2,v3+100)));
            %                 end
            %             elseif isleg == 2 % rLeg
            %                  if volume > (0.35*TotalBodyVolume)
            %                         srtd_i = vA_hatL{1,1};
            %                         vert11 = self.v(srtd_i(:,1),:);
            %                         vert12 = self.v(srtd_i(:,2),:);
            %                         srtd_i = vA_hatL{1,2};
            %                         vert21 = self.v(srtd_i(:,1),:);
            %                         vert22 = self.v(srtd_i(:,2),:);
            %                         v1 = [self.v(faces(:,1),:); vert12 ; vert22];
            %                         v2 = [self.v(faces(:,2),:); vert11 ; vert21];
            %                         v3 = [self.v(faces(:,3),:); repmat(center(1,:),length(vert11),1); repmat(center(2,:),length(vert21),1)];
            %                         volume = abs(sum(SignedVolumeOfTriangle(100+v1,100+v2,100+v3)));
            %                  end
            %              elseif isleg == 1 % lLeg
            %                  if volume > (0.35*TotalBodyVolume)
            %                         srtd_i = vA_hatL{1,1};
            %                         vert11 = self.v(srtd_i(:,1),:);
            %                         vert12 = self.v(srtd_i(:,2),:);
            %                         srtd_i = vA_hatL{1,2};
            %                         vert21 = self.v(srtd_i(:,1),:);
            %                         vert22 = self.v(srtd_i(:,2),:);
            %                         v2 = [self.v(faces(:,2),:); vert12 ; vert22];
            %                         v1 = [self.v(faces(:,1),:); vert11 ; vert21];
            %                         v3 = [self.v(faces(:,3),:); repmat(center(1,:),length(vert11),1); repmat(center(2,:),length(vert21),1)];
            %                         volume = abs(sum(SignedVolumeOfTriangle(100+v1,100+v2,100+v3)));
            %                  end
            %             elseif isleg == 5  % rArm
            %                 if volume > (0.075*TotalBodyVolume)
            %                     v1 = [self.v(faces(:,1),:); vert1];
            %                     v2 = [self.v(faces(:,2),:); vert2];
            %                     volume = abs(sum(SignedVolumeOfTriangle(100+v1,100+v2,100+v3)));
            %                 end
            %             end
			function index = getIndex(value,matrix)
				%Linear search
				[m,n] = size(matrix);
				maxIndex = m*n;
				for i = 1:maxIndex
					if matrix(i) == value
						index = i;
						break
					end
				end
				if matrix(index) ~= value
					index = -1;
				end
			end
        end
				
        function [rwrist, lwrist,...
                  rWristGirth,lWristGirth,...
                  rwrist_front,rwrist_back,rwrist_lateral,rwrist_medial,...
                  lwrist_front,lwrist_back,lwrist_lateral,lwrist_medial] = getWrist(self)
%         function [rwrist, lwrist,rWristGirth,lWristGirth,rwrist_ulnar,rwrist_radial,lwrist_ulnar,lwrist_radial] = getWrist(self)

            % finding hands
            [~,rHandIdx] = min(self.v(:,1));
            right_hand = self.v(rHandIdx,:); %Gives the point on the right hand that is furthest to the right
            [~,lHandIdx] = max(self.v(:,1));
            left_hand =  self.v(lHandIdx,:); %Gives the point on the left hand that is furthest to the left

            % Establish rotation angle for arms
            hypotenuse = norm([self.lShoulder(1),self.lShoulder(3)] - [left_hand(1),left_hand(3)]);
            adjacent = self.lShoulder(3) - left_hand(3);
            theta_l = acos(adjacent/hypotenuse);
            
            hypotenuse = norm([self.rShoulder(1),self.rShoulder(3)] - [right_hand(1),right_hand(3)]);
            adjacent = self.rShoulder(3) - right_hand(3);
            theta_r = acos(adjacent/hypotenuse);
            
            % right arm
            [x,z] = rotate_person(self.v(:,1),self.v(:,3),theta_r);
            y = self.v(:,2);
            zStart = min(z(self.rArmIdx));
            zEnd = (2*zStart+max(z(self.rArmIdx)))/3;
            zStart = (3*zStart+zEnd)/4;
            %z = self.v(:,3);
            %%%temp%%%%
% %             figure;
% %             rightxyz = [x,y,z];
% %             rightxyz(:,1) = rightxyz(:,1) - max(x(self.rArmIdx));
% %             rightxyz(:,3) = rightxyz(:,3) - min(z(self.rArmIdx));
% %             rArmFaces = getFaces(self,self.rArmIdx);
% %             h_rArm = patch('vertices', rightxyz, 'faces', rArmFaces, 'FaceColor', 'm');
% %             hold on;
            %%%%%%%
            
            n = 20;
            zValue = linspace(zStart,zEnd,n);
            [vOnLine,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue, self.rArmIdx);
            maj_axis = zeros(1,n);
            for i = 1:n
                x_i = x(vIdxOnLine{i}); y_i = y(vIdxOnLine{i});
                dist=sqrt((repmat(x_i,size(x_i,2),size(x_i,1))-repmat(x_i',size(x_i,1),size(x_i,2))).^2+(repmat(y_i,size(y_i,2),size(y_i,1))-repmat(y_i',size(y_i,1),size(y_i,2))).^2);
                maj_axis(i)=max(max(dist));
            end
            
            [~, idx] = min(maj_axis);
            [rWristGirth.value,b] = getCircumference(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));            
            rWristGirth.points = [vOnLine{idx}(b,1),vOnLine{idx}(b,2),vOnLine{idx}(b,3)];
            rwrist = [mean(rWristGirth.points(:,1)), mean(rWristGirth.points(:,2)), mean(rWristGirth.points(:,3))];
            
            [~,idx] = min(rWristGirth.points(:,2)); rwrist_front   = rWristGirth.points(idx,:);
            [~,idx] = max(rWristGirth.points(:,2)); rwrist_back    = rWristGirth.points(idx,:);
            [~,idx] = min(rWristGirth.points(:,1)); rwrist_lateral = rWristGirth.points(idx,:);
            [~,idx] = max(rWristGirth.points(:,1)); rwrist_medial  = rWristGirth.points(idx,:);
                        
            if(self.circ_ellipse)
            [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));
            rWristGirth.a_over_b = semiminor_axis/semimajor_axis;
            rWristGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            
            if(self.circ_cpd)
                [rWristGirth.templatePoints,rWristGirth.templateValue] = template_circumference( vOnLine);
            end
            
% %             IdxVert = 0;
% %             m=1;
%             %[semimajor_axis, ~, ~, ~, ~] = ellipse_fit(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));
%             PointsDiff = zeros(size(vIdxOnLine{idx},1),size(vIdxOnLine{idx},1));
%             for m = 1:size(vIdxOnLine{idx},1)
%                 PointsDiff(:,m) = sqrt(x(vIdxOnLine{idx}).^2+y(vIdxOnLine{idx}).^2+z(vIdxOnLine{idx}).^2) - sqrt(x(vIdxOnLine{idx}(m))^2+y(vIdxOnLine{idx}(m))^2+z(vIdxOnLine{idx}(m))^2);
%                
%                 %[~,IdxVert] = ismember(2*semimajor_axis, PointsDiff);
%                 %m=m+1; 
%             end 
%             [IdxVert,m] = find(PointsDiff == max(max(PointsDiff)));
%             Point1 = vIdxOnLine{idx}(m);
%             Point2 = vIdxOnLine{idx}(IdxVert);
%             if self.v(Point1,2) > self.v(Point2,2)
%                 rwrist_ulnar = [self.v(Point1,1),self.v(Point1,2),self.v(Point1,3)];
%                 rwrist_radial = [self.v(Point2,1),self.v(Point2,2),self.v(Point2,3)];
%             else
%                 rwrist_radial = [self.v(Point1,1),self.v(Point1,2),self.v(Point1,3)];
%                 rwrist_ulnar = [self.v(Point2,1),self.v(Point2,2),self.v(Point2,3)];
%             end 
            
            %%%temp%%%
%             rightVOnLine = [x(vIdxOnLine{idx}),y(vIdxOnLine{idx})];
%             rightb = b;
            %%%%%%
            
            % left arm
            [x,z] = rotate_person(self.v(:,1),self.v(:,3),-theta_l);
            zStart = min(z(self.lArmIdx));
            zEnd = (2*zStart+max(z(self.lArmIdx)))/3;
            zStart = (3*zStart+zEnd)/4;
            
            %%%temp%%%%
%             leftxyz = [x,y,z];
%             leftxyz(:,1) = leftxyz(:,1) - min(x(self.lArmIdx));
%             leftxyz(:,3) = leftxyz(:,3) - min(z(self.lArmIdx));
%             lArmFaces = getFaces(self,self.lArmIdx);
%             h_lArm = patch('vertices', leftxyz, 'faces', lArmFaces, 'FaceColor', 'y');
%             set(h_lArm,'LineStyle','none')
%             set(h_rArm,'LineStyle','none')
%             axis equal;
%             light('Position',[-50,-50,50],'Style','infinite');
%             light('Position',[50,50,50],'Style','infinite');
%             lighting phong;
%             view([0 0]);
            %%%%%%%
            
            n = 20;
            zValue = linspace(zStart,zEnd,n);
            [vOnLine,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue, self.lArmIdx);
            maj_axis = zeros(1,n);
            for i = 1:n
                x_i = x(vIdxOnLine{i}); y_i = y(vIdxOnLine{i});
                dist=sqrt((repmat(x_i,size(x_i,2),size(x_i,1))-repmat(x_i',size(x_i,1),size(x_i,2))).^2+(repmat(y_i,size(y_i,2),size(y_i,1))-repmat(y_i',size(y_i,1),size(y_i,2))).^2);
                maj_axis(i)=max(max(dist));
            end
            
            [~, idx] = min(maj_axis);
            [lWristGirth.value,b] = getCircumference(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));  
            lWristGirth.points = [vOnLine{idx}(b,1),vOnLine{idx}(b,2),vOnLine{idx}(b,3)];
            lwrist = [mean(lWristGirth.points(:,1)), mean(lWristGirth.points(:,2)), mean(lWristGirth.points(:,3))];
            
            [~,idx] = min(lWristGirth.points(:,2)); lwrist_front   = lWristGirth.points(idx,:);
            [~,idx] = max(lWristGirth.points(:,2)); lwrist_back    = lWristGirth.points(idx,:);
            [~,idx] = min(lWristGirth.points(:,1)); lwrist_medial  = lWristGirth.points(idx,:);
            [~,idx] = max(lWristGirth.points(:,1)); lwrist_lateral = lWristGirth.points(idx,:);
            
            if(self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));
                lWristGirth.a_over_b = semiminor_axis/semimajor_axis;
                lWristGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            
            if(self.circ_cpd)
                [lWristGirth.templatePoints,lWristGirth.templateValue] = template_circumference( vOnLine);
            end
            
%               PointsDiff = zeros(size(vIdxOnLine{idx},1),size(vIdxOnLine{idx},1));
%             for m = 1:size(vIdxOnLine{idx},1)
%                 PointsDiff(:,m) = sqrt(x(vIdxOnLine{idx}).^2+y(vIdxOnLine{idx}).^2+z(vIdxOnLine{idx}).^2) - sqrt(x(vIdxOnLine{idx}(m))^2+y(vIdxOnLine{idx}(m))^2+z(vIdxOnLine{idx}(m))^2);
%
%                 %[~,IdxVert] = ismember(2*semimajor_axis, PointsDiff);
%                 %m=m+1;
%             end
%             [IdxVert,m] = find(PointsDiff == max(max(PointsDiff)));
%             Point1 = vIdxOnLine{idx}(m);
%             Point2 = vIdxOnLine{idx}(IdxVert);
%             if self.v(Point1,2) > self.v(Point2,2)
%                 lwrist_ulnar = [self.v(Point1,1),self.v(Point1,2),self.v(Point1,3)];
%                 lwrist_radial = [self.v(Point2,1),self.v(Point2,2),self.v(Point2,3)];
%             else
%                 lwrist_radial = [self.v(Point1,1),self.v(Point1,2),self.v(Point1,3)];
%                 lwrist_ulnar = [self.v(Point2,1),self.v(Point2,2),self.v(Point2,3)];
%             end
            %%%temp%%%
%             leftVOnLine = [x(vIdxOnLine{idx}),y(vIdxOnLine{idx})];
%             leftb = b;
%             figure;
%             x = rightVOnLine(:,1); y = rightVOnLine(:,2);
%             x = abs(x - max(x)); y = y - min(y);
%             plot(x,y,'.','color','r');
%             hold on;
%             plot(x(rightb),y(rightb),'-','color','r');
%             x = leftVOnLine(:,1); y = leftVOnLine(:,2);
%             x = x - min(x); y = y - min(y);
%             plot(x,y,'.','color','b');
%             plot(x(leftb),y(leftb),'-','color','b');
%             axis equal;
            %%%%%%
            
        end
        
        function [rwrist, lwrist,...
                  rWristGirth,lWristGirth,...
                  rwrist_front,rwrist_back,rwrist_lateral,rwrist_medial,...
                  lwrist_front,lwrist_back,lwrist_lateral,lwrist_medial] = getWrist_old(self)
%         function [rwrist, lwrist,rWristGirth,lWristGirth,rwrist_ulnar,rwrist_radial,lwrist_ulnar,lwrist_radial] = getWrist(self)

            % finding hands
            [~,rHandIdx] = min(self.v(:,1));
            right_hand = self.v(rHandIdx,:); %Gives the point on the right hand that is furthest to the right
            [~,lHandIdx] = max(self.v(:,1));
            left_hand =  self.v(lHandIdx,:); %Gives the point on the left hand that is furthest to the left

            % Establish rotation angle for arms
            hypotenuse = norm([self.lShoulder(1),self.lShoulder(3)] - [left_hand(1),left_hand(3)]);
            adjacent = self.lShoulder(3) - left_hand(3);
            theta_l = acos(adjacent/hypotenuse);
            
            hypotenuse = norm([self.rShoulder(1),self.rShoulder(3)] - [right_hand(1),right_hand(3)]);
            adjacent = self.rShoulder(3) - right_hand(3);
            theta_r = acos(adjacent/hypotenuse);
            
            % right arm
            [x,z] = rotate_person(self.v(:,1),self.v(:,3),theta_r);
            y = self.v(:,2);
            zStart = min(z(self.rArmIdx));
            zEnd = (3*zStart+max(z(self.rArmIdx)))/4;
            zStart = (3*zStart+zEnd)/4;
            %z = self.v(:,3);
            %%%temp%%%%
% %             figure;
% %             rightxyz = [x,y,z];
% %             rightxyz(:,1) = rightxyz(:,1) - max(x(self.rArmIdx));
% %             rightxyz(:,3) = rightxyz(:,3) - min(z(self.rArmIdx));
% %             rArmFaces = getFaces(self,self.rArmIdx);
% %             h_rArm = patch('vertices', rightxyz, 'faces', rArmFaces, 'FaceColor', 'm');
% %             hold on;
            %%%%%%%
            
            n = 20;
            zValue = linspace(zStart,zEnd,n);
            [vOnLine,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue, self.rArmIdx);
            circumference = zeros(1,n);
            for i = 1:n
                circumference(i) = getCircumference(x(vIdxOnLine{i}),y(vIdxOnLine{i}));
            end
            
            [rWristGirth.value, idx] = min(circumference);
            [~,b] = getCircumference(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));
            rWristGirth.points = [vOnLine{idx}(b,1),vOnLine{idx}(b,2),vOnLine{idx}(b,3)];
            rwrist = [mean(rWristGirth.points(:,1)), mean(rWristGirth.points(:,2)), mean(rWristGirth.points(:,3))];
            
            [~,idx] = min(rWristGirth.points(:,2)); rwrist_front   = rWristGirth.points(idx,:);
            [~,idx] = max(rWristGirth.points(:,2)); rwrist_back    = rWristGirth.points(idx,:);
            [~,idx] = min(rWristGirth.points(:,1)); rwrist_lateral = rWristGirth.points(idx,:);
            [~,idx] = max(rWristGirth.points(:,1)); rwrist_medial  = rWristGirth.points(idx,:);
                        
            if(self.circ_ellipse)
            [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));
            rWristGirth.a_over_b = semiminor_axis/semimajor_axis;
            rWristGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            
            if(self.circ_cpd)
                [rWristGirth.templatePoints,rWristGirth.templateValue] = template_circumference( vOnLine);
            end
            
% %             IdxVert = 0;
% %             m=1;
%             %[semimajor_axis, ~, ~, ~, ~] = ellipse_fit(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));
%             PointsDiff = zeros(size(vIdxOnLine{idx},1),size(vIdxOnLine{idx},1));
%             for m = 1:size(vIdxOnLine{idx},1)
%                 PointsDiff(:,m) = sqrt(x(vIdxOnLine{idx}).^2+y(vIdxOnLine{idx}).^2+z(vIdxOnLine{idx}).^2) - sqrt(x(vIdxOnLine{idx}(m))^2+y(vIdxOnLine{idx}(m))^2+z(vIdxOnLine{idx}(m))^2);
%                
%                 %[~,IdxVert] = ismember(2*semimajor_axis, PointsDiff);
%                 %m=m+1; 
%             end 
%             [IdxVert,m] = find(PointsDiff == max(max(PointsDiff)));
%             Point1 = vIdxOnLine{idx}(m);
%             Point2 = vIdxOnLine{idx}(IdxVert);
%             if self.v(Point1,2) > self.v(Point2,2)
%                 rwrist_ulnar = [self.v(Point1,1),self.v(Point1,2),self.v(Point1,3)];
%                 rwrist_radial = [self.v(Point2,1),self.v(Point2,2),self.v(Point2,3)];
%             else
%                 rwrist_radial = [self.v(Point1,1),self.v(Point1,2),self.v(Point1,3)];
%                 rwrist_ulnar = [self.v(Point2,1),self.v(Point2,2),self.v(Point2,3)];
%             end 
            
            %%%temp%%%
%             rightVOnLine = [x(vIdxOnLine{idx}),y(vIdxOnLine{idx})];
%             rightb = b;
            %%%%%%
            
            % left arm
            [x,z] = rotate_person(self.v(:,1),self.v(:,3),-theta_l);
            zStart = min(z(self.lArmIdx));
            zEnd = (3*zStart+max(z(self.lArmIdx)))/4;
            zStart = (3*zStart+zEnd)/4;
            
            %%%temp%%%%
%             leftxyz = [x,y,z];
%             leftxyz(:,1) = leftxyz(:,1) - min(x(self.lArmIdx));
%             leftxyz(:,3) = leftxyz(:,3) - min(z(self.lArmIdx));
%             lArmFaces = getFaces(self,self.lArmIdx);
%             h_lArm = patch('vertices', leftxyz, 'faces', lArmFaces, 'FaceColor', 'y');
%             set(h_lArm,'LineStyle','none')
%             set(h_rArm,'LineStyle','none')
%             axis equal;
%             light('Position',[-50,-50,50],'Style','infinite');
%             light('Position',[50,50,50],'Style','infinite');
%             lighting phong;
%             view([0 0]);
            %%%%%%%
            
            n = 20;
            zValue = linspace(zStart,zEnd,n);
            [vOnLine,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue, self.lArmIdx);
            circumference = zeros(1,n);
            for i = 1:n
                circumference(i) = getCircumference(x(vIdxOnLine{i}),y(vIdxOnLine{i}));
            end
            
            [lWristGirth.value, idx] = min(circumference);
            [~,b] = getCircumference(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));
            lWristGirth.points = [vOnLine{idx}(b,1),vOnLine{idx}(b,2),vOnLine{idx}(b,3)];
            lwrist = [mean(lWristGirth.points(:,1)), mean(lWristGirth.points(:,2)), mean(lWristGirth.points(:,3))];
            
            [~,idx] = min(lWristGirth.points(:,2)); lwrist_front   = lWristGirth.points(idx,:);
            [~,idx] = max(lWristGirth.points(:,2)); lwrist_back    = lWristGirth.points(idx,:);
            [~,idx] = min(lWristGirth.points(:,1)); lwrist_medial  = lWristGirth.points(idx,:);
            [~,idx] = max(lWristGirth.points(:,1)); lwrist_lateral = lWristGirth.points(idx,:);
            
            if(self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(x(vIdxOnLine{idx}),y(vIdxOnLine{idx}));
                lWristGirth.a_over_b = semiminor_axis/semimajor_axis;
                lWristGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            
            if(self.circ_cpd)
                [lWristGirth.templatePoints,lWristGirth.templateValue] = template_circumference( vOnLine);
            end
            
%               PointsDiff = zeros(size(vIdxOnLine{idx},1),size(vIdxOnLine{idx},1));
%             for m = 1:size(vIdxOnLine{idx},1)
%                 PointsDiff(:,m) = sqrt(x(vIdxOnLine{idx}).^2+y(vIdxOnLine{idx}).^2+z(vIdxOnLine{idx}).^2) - sqrt(x(vIdxOnLine{idx}(m))^2+y(vIdxOnLine{idx}(m))^2+z(vIdxOnLine{idx}(m))^2);
%
%                 %[~,IdxVert] = ismember(2*semimajor_axis, PointsDiff);
%                 %m=m+1;
%             end
%             [IdxVert,m] = find(PointsDiff == max(max(PointsDiff)));
%             Point1 = vIdxOnLine{idx}(m);
%             Point2 = vIdxOnLine{idx}(IdxVert);
%             if self.v(Point1,2) > self.v(Point2,2)
%                 lwrist_ulnar = [self.v(Point1,1),self.v(Point1,2),self.v(Point1,3)];
%                 lwrist_radial = [self.v(Point2,1),self.v(Point2,2),self.v(Point2,3)];
%             else
%                 lwrist_radial = [self.v(Point1,1),self.v(Point1,2),self.v(Point1,3)];
%                 lwrist_ulnar = [self.v(Point2,1),self.v(Point2,2),self.v(Point2,3)];
%             end
            %%%temp%%%
%             leftVOnLine = [x(vIdxOnLine{idx}),y(vIdxOnLine{idx})];
%             leftb = b;
%             figure;
%             x = rightVOnLine(:,1); y = rightVOnLine(:,2);
%             x = abs(x - max(x)); y = y - min(y);
%             plot(x,y,'.','color','r');
%             hold on;
%             plot(x(rightb),y(rightb),'-','color','r');
%             x = leftVOnLine(:,1); y = leftVOnLine(:,2);
%             x = x - min(x); y = y - min(y);
%             plot(x,y,'.','color','b');
%             plot(x(leftb),y(leftb),'-','color','b');
%             axis equal;
            %%%%%%
            
        end
        
        function [distl,distr,distrl] = getCurve(self, d, n)
        % Gets side or front curve
            % d = 1 for side curve
            % d = 2 for front curve
            if d ~= 1 && d ~= 2
                disp('Error: The value for d needs to be 1 or 2')
                return
            end
            zPoints = linspace(self.crotch(3), min(self.r_armpit(3),self.l_armpit(3)), n);
            
            distrl = zeros(n,1); %distance of pos - neg
            distr = zeros(n,1); %distance of pos to y=0
            distl = zeros(n,1); %distance of neg to y=0
            
            a = reshape(self.v(self.f(:,:),3), [], 3);% z values for faces
            
            for i = 1:n
                facesOnLine = (a(:,1) >= zPoints(i) & (a(:,2) <= zPoints(i) | a(:,3) <= zPoints(i))) |...
                                   (a(:,2) >= zPoints(i) & (a(:,1) <= zPoints(i) | a(:,3) <= zPoints(i))) |...
                                   (a(:,3) >= zPoints(i) & (a(:,1) <= zPoints(i) | a(:,2) <= zPoints(i)));
                vIdxOnLine = reshape(self.f(facesOnLine,:),[],1);
                vIdxOnLine = unique(vIdxOnLine);
                trunkPlusLegs = unique([self.trunkIdx; self.legIdx]);
                vIdxOnLine = intersect(trunkPlusLegs,vIdxOnLine); % only vertices in trunk (no arms)
                vOnLine = self.v(vIdxOnLine,:);
                distrl(i) = max(vOnLine(:,d)) - min(vOnLine(:,d)); %range of RHS-LHS
                distr(i) = max(vOnLine(:,d)); %RHS >0
                distl(i) = min(vOnLine(:,d)); %LHS <0
            end
           
        end
        
        function [vOnLine,vIndex] = getVOnLine(self, v, zValue, keepIdx)
        % Finds the vertices at a given z value
            % v are all vertices (they need to be rotated if necessary)
            % zValue is the given z value (can be multiple values in an array
            % keepIdx are the indices that should be keept (needed for
            %         chopping of)
            
            n = length(zValue);
            vOnLine = cell(1,n);
            vIndex = cell(1,n);
            z = reshape(v(self.f(:,:),3), [], 3); % z values of all faces
            for i = 1:n
                facesOnLine = (z(:,1) >= zValue(i) & (z(:,2) <= zValue(i) | z(:,3) <= zValue(i))) |...
                              (z(:,2) >= zValue(i) & (z(:,1) <= zValue(i) | z(:,3) <= zValue(i))) |...
                              (z(:,3) >= zValue(i) & (z(:,1) <= zValue(i) | z(:,2) <= zValue(i)));  
                vIdxOnLine = reshape(self.f(facesOnLine,:),[],1);
                vIdxOnLine = unique(vIdxOnLine);
                vIdxOnLine = intersect(keepIdx,vIdxOnLine); % only vertices that are in keepIdx
                vOnLine(i) = {self.v(vIdxOnLine,:)};
                vIndex(i) = {vIdxOnLine};
            end
            if n == 1
                vOnLine = vOnLine{1};
                vIndex = vIndex{1};
            end
        end
          
        function [rightforearmgirth,leftforearmgirth,...
                  rightbicepgirth,leftbicepgirth,...
                  rforearm_front,rforearm_back,rforearm_lateral,rforearm_medial,...
                  lforearm_front,lforearm_back,lforearm_lateral,lforearm_medial,...
                  rbicep_front,rbicep_back,rbicep_lateral,rbicep_medial,...
                  lbicep_front,lbicep_back,lbicep_lateral,lbicep_medial] = getArmGirth(self)
            A = [0;self.r_wrist(3) - self.armMaxR(2)];
            B = [self.r_wrist(1)-self.armMaxR(1); self.r_wrist(3)-self.armMaxL(2)];
            C = [0;self.l_wrist(3) - self.armMaxL(2)];
            D = [self.l_wrist(1)-self.armMaxL(1); self.l_wrist(3)-self.armMaxL(2)];
            
            right_arm_theta = acos(dot(A,B)/(norm(A)*norm(B)));
            left_arm_theta = acos(dot(C,D)/(norm(C)*norm(D)));
            
            % Rotate arms
            [rotated_rarm_x, rotated_rarm_z] = rotate_person(self.v(:,1), self.v(:,3), right_arm_theta);
            [rotated_larm_x, rotated_larm_z] = rotate_person(self.v(:,1), self.v(:,3), -left_arm_theta);
            
            % Forearm right
            rightforearm = [((3*self.r_wrist(1))+self.armMaxR(1))/4, ((3*self.r_wrist(3))+self.armMaxR(2))/4];
            [~, rotated_right_forearm_z] = rotate_person(rightforearm(1),rightforearm(2), right_arm_theta);
            newV = [rotated_rarm_x, self.v(:,2), rotated_rarm_z];
            [vOnLine,vIdxOnLine] = self.getVOnLine(newV, rotated_right_forearm_z, self.rArmIdx);
            [rightforearmgirth.value,b] = getCircumference(newV(vIdxOnLine,1), newV(vIdxOnLine,2));
            rightforearmgirth.points = vOnLine(b,:);
            
            [~,idx] = min(rightforearmgirth.points(:,2)); rforearm_front   = rightforearmgirth.points(idx,:);
            [~,idx] = max(rightforearmgirth.points(:,2)); rforearm_back    = rightforearmgirth.points(idx,:);
            [~,idx] = min(rightforearmgirth.points(:,1)); rforearm_lateral = rightforearmgirth.points(idx,:);
            [~,idx] = max(rightforearmgirth.points(:,1)); rforearm_medial  = rightforearmgirth.points(idx,:);
            
            if(self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(newV(vIdxOnLine,1), newV(vIdxOnLine,2));
                rightforearmgirth.a_over_b = semiminor_axis/semimajor_axis;
                rightforearmgirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end 
            
            if(self.circ_cpd)
                [rightforearmgirth.templatePoints,rightforearmgirth.templateValue] = template_circumference( vOnLine);
            end
            
            config.scene = [newV(vIdxOnLine,1), newV(vIdxOnLine,2)]; % same input as the input to ellipse_fits
            
            % Forearm left
            leftforearm = [(3*self.l_wrist(1)+self.armMaxL(1))/4, (3*self.l_wrist(3)+self.armMaxL(2))/4];
            [~, rotated_left_forearm_z] = rotate_person(leftforearm(1),leftforearm(2), -left_arm_theta);
            newV = [rotated_larm_x, self.v(:,2), rotated_larm_z];
            [vOnLine,vIdxOnLine] = self.getVOnLine(newV, rotated_left_forearm_z, self.lArmIdx);
            [leftforearmgirth.value,b] = getCircumference(newV(vIdxOnLine,1), newV(vIdxOnLine,2));
            leftforearmgirth.points = vOnLine(b,:);
            
            [~,idx] = min(leftforearmgirth.points(:,2)); lforearm_front   = leftforearmgirth.points(idx,:);
            [~,idx] = max(leftforearmgirth.points(:,2)); lforearm_back    = leftforearmgirth.points(idx,:);
            [~,idx] = min(leftforearmgirth.points(:,1)); lforearm_medial  = leftforearmgirth.points(idx,:);
            [~,idx] = max(leftforearmgirth.points(:,1)); lforearm_lateral = leftforearmgirth.points(idx,:);

            if (self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(newV(vIdxOnLine,1), newV(vIdxOnLine,2));
                leftforearmgirth.a_over_b = semiminor_axis/semimajor_axis;
                leftforearmgirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            if(self.circ_cpd)
                [leftforearmgirth.templatePoints,leftforearmgirth.templateValue] = template_circumference( vOnLine);
            end
            
            % Bicep right
            rightbicep = [(self.r_wrist(1)+3*self.armMaxR(1))/4, (self.r_wrist(3)+3*self.armMaxR(2))/4];
            [~, rotated_right_bicep_z] = rotate_person(rightbicep(1),rightbicep(2), right_arm_theta);
            newV = [rotated_rarm_x, self.v(:,2), rotated_rarm_z];
            [vOnLine,vIdxOnLine] = self.getVOnLine(newV, rotated_right_bicep_z, self.rArmIdx);
            [rightbicepgirth.value,b] = getCircumference(newV(vIdxOnLine,1), newV(vIdxOnLine,2));
            rightbicepgirth.points = vOnLine(b,:);
            
            [~,idx] = min(rightbicepgirth.points(:,2)); rbicep_front   = rightbicepgirth.points(idx,:);
            [~,idx] = max(rightbicepgirth.points(:,2)); rbicep_back    = rightbicepgirth.points(idx,:);
            [~,idx] = min(rightbicepgirth.points(:,1)); rbicep_lateral = rightbicepgirth.points(idx,:);
            [~,idx] = max(rightbicepgirth.points(:,1)); rbicep_medial  = rightbicepgirth.points(idx,:);

            if (self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(newV(vIdxOnLine,1), newV(vIdxOnLine,2));
                rightbicepgirth.a_over_b = semiminor_axis/semimajor_axis;
                rightbicepgirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            if(self.circ_cpd)
                [rightbicepgirth.templatePoints,rightbicepgirth.templateValue] = template_circumference( vOnLine);
            end
                
            % Bicep left
            leftbicep = [(self.l_wrist(1)+3*self.armMaxL(1))/4, (self.l_wrist(3)+3*self.armMaxL(2))/4];
            [~, rotated_left_bicep_z] = rotate_person(leftbicep(1),leftbicep(2), -left_arm_theta);
            newV = [rotated_larm_x, self.v(:,2), rotated_larm_z];
            [vOnLine,vIdxOnLine] = self.getVOnLine(newV, rotated_left_bicep_z, self.lArmIdx);
            [leftbicepgirth.value,b] = getCircumference(newV(vIdxOnLine,1), newV(vIdxOnLine,2));
            leftbicepgirth.points = vOnLine(b,:);
            
            [~,idx] = min(leftbicepgirth.points(:,2)); lbicep_front   = leftbicepgirth.points(idx,:);
            [~,idx] = max(leftbicepgirth.points(:,2)); lbicep_back    = leftbicepgirth.points(idx,:);
            [~,idx] = min(leftbicepgirth.points(:,1)); lbicep_medial  = leftbicepgirth.points(idx,:);
            [~,idx] = max(leftbicepgirth.points(:,1)); lbicep_lateral = leftbicepgirth.points(idx,:);
            
            if (self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(newV(vIdxOnLine,1), newV(vIdxOnLine,2));
                leftbicepgirth.a_over_b = semiminor_axis/semimajor_axis;
                leftbicepgirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end     
            if(self.circ_cpd)
                [leftbicepgirth.templatePoints,leftbicepgirth.templateValue] = template_circumference( vOnLine);
            end
        end
        
        function [lAnkle, rAnkle, lAnkleGirth, rAnkleGirth, rankle_medialPoint, rankle_lateralPoint, lankle_medialPoint, lankle_lateralPoint] = getAnkleGirth(self)
            n = 20;
            
            % right leg
            zStart = min(self.v(self.v(:,1)<0,3));
            zEnd = (3*zStart + self.crotch(3))/4;
            zStart = (7*zStart + zEnd)/8;
            
            rLegIdx = intersect(find(self.v(:,1)<self.crotch(1)),self.legIdx);
            zValue = linspace(zStart,zEnd,n);
            vOnLine = getVOnLine(self, self.v, zValue, rLegIdx);
            circumference = zeros(1,n);
            for i = 1:n
                circumference(i) = getCircumference(vOnLine{i}(:,1),vOnLine{i}(:,2));
            end
            
            [rAnkleGirth.value, idx] = min(circumference);
            [~,b] = getCircumference(vOnLine{idx}(:,1),vOnLine{idx}(:,2));
            rAnkleGirth.points = [vOnLine{idx}(b,1),vOnLine{idx}(b,2),vOnLine{idx}(b,3)];
            rAnkle = [mean(rAnkleGirth.points(:,1)), mean(rAnkleGirth.points(:,2)), mean(rAnkleGirth.points(:,3))];
            if(self.circ_ellipse)
            [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(vOnLine{idx}(:,1),vOnLine{idx}(:,2));
            rAnkleGirth.a_over_b = semiminor_axis/semimajor_axis;
            rAnkleGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            if(self.circ_cpd)
                [rAnkleGirth.templatePoints,rAnkleGirth.templateValue] = template_circumference( vOnLine);
            end
            
            [~, Idx_rankle_medial] = max(rAnkleGirth.points(:,1));
            [~, Idx_rankle_lateral] = min(rAnkleGirth.points(:,1));
            rankle_medialPoint = rAnkleGirth.points(Idx_rankle_medial,:);
            rankle_lateralPoint = rAnkleGirth.points(Idx_rankle_lateral,:); 
            
            % left leg
            zStart = min(self.v(self.v(:,1)>0,3));
            zEnd = (3*zStart + self.crotch(3))/4;
            zStart = (7*zStart + zEnd)/8;
            
            lLegIdx = intersect(find(self.v(:,1)>self.crotch(1)),self.legIdx);
            zValue = linspace(zStart,zEnd,n);
            vOnLine = getVOnLine(self, self.v, zValue, lLegIdx);
            circumference = zeros(1,n);
            for i = 1:n
                circumference(i) = getCircumference(vOnLine{i}(:,1),vOnLine{i}(:,2));
            end
            
            [lAnkleGirth.value, idx] = min(circumference);
            [~,b] = getCircumference(vOnLine{idx}(:,1),vOnLine{idx}(:,2));
            lAnkleGirth.points = [vOnLine{idx}(b,1),vOnLine{idx}(b,2),vOnLine{idx}(b,3)];
            lAnkle = [mean(lAnkleGirth.points(:,1)), mean(lAnkleGirth.points(:,2)), mean(lAnkleGirth.points(:,3))];
            if(self.circ_ellipse)
            [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(vOnLine{idx}(:,1),vOnLine{idx}(:,2));
            lAnkleGirth.a_over_b = semiminor_axis/semimajor_axis;
            lAnkleGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            if(self.circ_cpd)
                [lAnkleGirth.templatePoints,lAnkleGirth.templateValue] = template_circumference( vOnLine);
            end
            
            [~, Idx_lankle_medial] = min(lAnkleGirth.points(:,1));
            [~, Idx_lankle_lateral] = max(lAnkleGirth.points(:,1));
            lankle_medialPoint = lAnkleGirth.points(Idx_lankle_medial,:);
            lankle_lateralPoint = lAnkleGirth.points(Idx_lankle_lateral,:);  
            
        end
        
        function [lCalfGirth, rCalfGirth, rcalf_backPoint, rcalf_medialPoint, rcalf_lateralPoint, lcalf_backPoint, lcalf_medialPoint, lcalf_lateralPoint] = getCalf(self)

            %Left Leg 
            legVal = 0;
            zStart = self.l_ankle(3); % ankle
            zEnd   = self.crotch(3);
            [lCalfGirth,lZValue,lAngle] = calfGirth(self, zStart, zEnd, legVal);
            
            
            

            % Right Leg
            legVal = 1;
            zStart = self.r_ankle(3); % ankle  
            zEnd   = self.crotch(3);
            [rCalfGirth,rZValue,rAngle] = calfGirth(self, zStart, zEnd, legVal);

            % If one of the calfgirth failed, use the others zValue
            % If both failed, use zValue specified in calfGirthOther
            if lCalfGirth.value == 0 && rCalfGirth.value == 0
                lCalfGirth = calfGirthOther(self,0,lAngle,0);
                rCalfGirth = calfGirthOther(self,0,rAngle,1);
            elseif lCalfGirth.value == 0
                lCalfGirth = calfGirthOther(self,rZValue,lAngle,0);
            elseif rCalfGirth.value == 0
                rCalfGirth = calfGirthOther(self,lZValue,rAngle,1);
            end
            
            [~, Idx_lcalf_backPoint] = max(lCalfGirth.points(:,2));
            lcalf_backPoint = lCalfGirth.points(Idx_lcalf_backPoint,:);
            
            [~, Idx_lcalf_medialPoint] = min(lCalfGirth.points(:,1));
            lcalf_medialPoint =  lCalfGirth.points(Idx_lcalf_medialPoint,:);
            
            [~, Idx_lcalf_lateralPoint] = max(lCalfGirth.points(:,1));
            lcalf_lateralPoint = lCalfGirth.points(Idx_lcalf_lateralPoint,:);
            
            [~, Idx_rcalf_backPoint] = max(rCalfGirth.points(:,2));
            rcalf_backPoint = rCalfGirth.points(Idx_rcalf_backPoint,:);
            
            [~, Idx_rcalf_medialPoint] = max(rCalfGirth.points(:,1));
            rcalf_medialPoint =  rCalfGirth.points(Idx_rcalf_medialPoint,:);
            
            [~, Idx_rcalf_lateralPoint] = min(rCalfGirth.points(:,1));
            rcalf_lateralPoint = rCalfGirth.points(Idx_rcalf_lateralPoint,:);
        end
        
        % If the calfgirth failed, this one can be used
        function calfGirth = calfGirthOther(self,zValue,angle,legVal)
            [x,z] = rotate_person(self.v(:,1),self.v(:,3),angle);
            y = self.v(:,2);
%             [y,~] = rotate_person(self.v(:,2),self.v(:,3),angle);
            if legVal == 0
                idx = find(self.v(:,1) > 0);
                zStart = self.l_ankle(3);
                zEnd   = self.crotch(3);
            else 
                idx = find(self.v(:,1) < 0);
                zStart = self.r_ankle(3); 
                zEnd   = self.crotch(3);
            end
            if zValue == 0 % If no zValue was given, use this one
                zValue = 0.25*(zEnd - zStart) + zStart;
            end
            
            [vOnLine,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue, idx);
            [calfGirth.value,b] = getCircumference(x(vIdxOnLine),y(vIdxOnLine));
            calfGirth.points = [vOnLine(b,1),vOnLine(b,2),vOnLine(b,3)];
            if (self.circ_ellipse)
                [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(vOnLine(:,1), vOnLine(:,2));
                calfGirth.a_over_b = semiminor_axis/semimajor_axis;
                calfGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            if(self.circ_cpd)
                [calfGirth.templatePoints,calfGirth.templateValue] = template_circumference( vOnLine);
            end            
        end

        function [calfGirth,zValue,angle] = calfGirth(self, zStart, zEnd, LegVal)
            diff = zEnd - zStart;
            zStart = 0.15*diff + zStart;
            zEnd = zEnd - 0.5*diff;
            % rotate and divide the body in half along vertical axes
            if(LegVal == 0) %If left leg
                p1 = [self.l_ankle(1),self.l_ankle(3)];
                p2 = [(self.crotch(1)+self.l_hip(1))/2,(self.crotch(3)+self.l_hip(3))/2]; % Midpoint of crotch and hip
                hypotenuse = norm(p1 - p2);
                adjacent = p2(2) - p1(2);
                theta = acos(adjacent/hypotenuse);
                angle = -theta;
                [x,z] = rotate_person(self.v(:,1),self.v(:,3),angle);
                y = self.v(:,2);
%                 [y,~] = rotate_person(self.v(:,2),self.v(:,3),angle);
                idx = find(self.v(:,1) > 0);
            else %Else right leg
                p1 = [self.r_ankle(1),self.r_ankle(3)];
                p2 = [(self.crotch(1)+self.r_hip(1))/2,(self.crotch(3)+self.r_hip(3))/2]; % Midpoint of crotch and hip
                hypotenuse = norm(p1 - p2);
                adjacent = p2(2) - p1(2);
                theta = acos(adjacent/hypotenuse);
                angle = theta;
                [x,z] = rotate_person(self.v(:,1),self.v(:,3),angle);
                y = self.v(:,2);
%                 [y,~] = rotate_person(self.v(:,2),self.v(:,3),angle);
                idx = find(self.v(:,1) < 0);
            end
            % We compare the girth for n slices of the leg from zStart to
            % zEnd. We identify the interval in which the girth decreases
            % for the first time.
            n = 20;
            intervals = linspace(zStart, zEnd, n+1);
            zPoints = mean([intervals(1:n);intervals(2:n+1)]);
            [~,vIdxOnLine] = getVOnLine(self,[x,y,z],zPoints(1),idx);
            circumference = zeros(1,n);
            circumference(1) = getCircumference(x(vIdxOnLine),y(vIdxOnLine));
            for i = 1:n
                if i == n
                    break;
                end
                [~,vIdxOnLine] = getVOnLine(self,[x,y,z],zPoints(i+1),idx);
                circumference(i+1) = getCircumference(x(vIdxOnLine),y(vIdxOnLine));
                if circumference(i) > circumference(i+1)
                    break;
                end
            end
            
            if i < 2 || i > n - 1 % Too close to zStart or zEnd, so use min y value instead                
                maxY = zeros(1,n);
                vOnLine = getVOnLine(self,[x,y,z],zPoints(1),idx);
                maxY(1) = max(vOnLine(:,2));
                for i = 1:n
                    if i == n
                        break;
                    end
                    vOnLine = getVOnLine(self,[x,y,z],zPoints(i+1),idx);
                    maxY(i+1) = max(vOnLine(:,2));
                    if maxY(i) > max(i+1)
                        break;
                    end
                end
            end
            
            if i < 2 || i > n - 1
                % calfGirth failed, so calfGirthOther will be called
                calfGirth.value = 0;
                zValue = 0;
                return;
            end
            
            % We find the largest girth in the previous identified interval
            m = 20;
            zPoints = linspace(intervals(i),intervals(i+1),m);

            [vOnLine,vIdxOnLine] = getVOnLine(self, [x,y,z], zPoints, idx);
            circumference = zeros(1,m);
            for i = 1:m
                circumference(i) = getCircumference(x(vIdxOnLine{i}),y(vIdxOnLine{i}));
            end
            
            [calfGirth.value, girthIdx] = max(circumference);
            [~,b] = getCircumference(x(vIdxOnLine{girthIdx}),y(vIdxOnLine{girthIdx}));
            calfGirth.points = [vOnLine{girthIdx}(b,1),vOnLine{girthIdx}(b,2),vOnLine{girthIdx}(b,3)];
            zValue = zPoints(girthIdx);
            
            
            
            if(self.circ_ellipse)
            [semimajor_axis, semiminor_axis, ~, ~, ~] = ellipse_fit(vOnLine{girthIdx}(b,1),vOnLine{girthIdx}(b,2));
            calfGirth.a_over_b = semiminor_axis/semimajor_axis;
            calfGirth.ellipseValue= sqrt(2)*pi*sqrt(semiminor_axis.^2+semimajor_axis.^2);
            end
            if(self.circ_cpd)
                [calfGirth.templatePoints,calfGirth.templateValue] = template_circumference( vOnLine);
            end
        end
        
        function [x,y,z] = slice_n_dice(self, n1, n2, zStart,zEnd, keepIdx)
            for n = [n1 n2] 
                zPoints = linspace(zStart, zEnd, n+1);
                dist    = zeros(n, 1);

                for i = 1:n
                    points = getVOnLine(self,self.v,mean([zPoints(i),zPoints(i+1)]),keepIdx);
                    dist(i) = max(points(:,2));
                end

                if n == n2
                    dist= sosmooth3(dist, 7);
                end

                j=1;
                cnd = 1;
                while(cnd)
                     j = j+1;
                     cnd1 = j<=n-1;
                     cnd2 =  dist(j-1) < dist(j);
                     cnd = logical(cnd1*cnd2);
                end

                zStart = zPoints(j-1);
                zEnd = zPoints(j);
            end
            vOnLine = getVOnLine(self,self.v,mean([zPoints(j-1),zPoints(j)]),keepIdx);
            x = vOnLine(:,1);
            y = vOnLine(:,2);
            z = vOnLine(:,3);
        end
        
        function [p1,p3] = findMaxMin(self,left,right,num)
        % Finds the maximum of the mins
        % This function is similar to find_minmax but uses vOnLine
            [v1,v3] = rotate_person(self.v(:,1),self.v(:,3),pi/2);
            [right(1),right(3)] = rotate_person(right(1),right(3),pi/2);
            [left(1),left(3)] = rotate_person(left(1),left(3),pi/2);

            partition = linspace(left(3),right(3),num);
            [vOnLine,vIdxOnLine] = getVOnLine(self,[v1,self.v(:,2),v3],partition,1:length(self.v));
            minValues = zeros(num,2);
            for i = 1:num
                [minValues(i,1),minValues(i,2)] = max(v1(vIdxOnLine{i})); % Find max of each portion
            end
            [~,maxIndex] = min(minValues(:,1)); % Get the max of the mins
            p = vOnLine{maxIndex}(minValues(maxIndex,2),:);
            p1 = p(1);
            p3 = p(3);
        end
        
        % Extracts all the computed values for easier use
        function values = extractValues(self)
            values.chestGirth = self.chestCircumference.value;
            values.waistGirth = self.waistCircumference.value;
            values.hipGirth = self.hipCircumference.value;
            values.rThighGirth = self.rThighGirth.value;
            values.lThighGirth = self.lThighGirth.value;
            values.rCalfGirth = self.rCalfCircumference.value;
            values.lCalfGirth = self.lCalfCircumference.value;
            values.lWristGirth = self.l_wristgirth.value;
            values.rWristGirth = self.r_wristgirth.value;
            values.rForearmGirth = self.r_forearmgirth.value;
            values.lForearmGirth = self.l_forearmgirth.value;
            values.rBicepGirth = self.r_bicepgirth.value;
            values.lBicepGirth = self.l_bicepgirth.value;
            values.rAnkleGirth = self.r_ankle_girth.value;
            values.lAnkleGirth = self.l_ankle_girth.value;
            values.lArmLength = self.leftArmLength;
            values.rArmLength = self.rightArmLength;
            values.trunkLength = self.trunkLength;
            values.lLegLength = self.lLegLength;
            values.rLegLength = self.rLegLength;
            values.crotchHeight = self.crotchHeight;
            values.collarScalpLength = self.collarScalpLength;
            values.volume = self.volume;
            values.surfaceArea = self.surfaceArea.total;
            values.headSA = self.surfaceArea.head;
            values.trunkSA = self.surfaceArea.trunk;
            values.lArmSA = self.surfaceArea.lArm;
            values.rArmSA = self.surfaceArea.rArm;
            values.lLegSA = self.surfaceArea.lLeg;
            values.rLegSA = self.surfaceArea.rLeg;
            
            %%%%% temp
%             values.chestCircumference = self.chestCircumference.value;
%             values.waistCircumference = self.waistCircumference.value;
%             values.hipCircumference = self.hipCircumference.value;
%             values.rThighGirth = self.rThighGirth.value;
%             values.lThighGirth = self.lThighGirth.value;
%             values.rCalfCircumference = self.rCalfCircumference.value;
%             values.lCalfCircumference = self.lCalfCircumference.value;
%             values.l_wristgirth = self.l_wristgirth.value;
%             values.rWristGirth = self.r_wristgirth.value;
%             values.rForearmGirth = self.r_forearmgirth.value;
%             values.lForearmGirth = self.l_forearmgirth.value;
%             values.rBicepGirth = self.r_bicepgirth.value;
%             values.lBicepGirth = self.l_bicepgirth.value;
%             values.rAnkleGirth = self.r_ankle_girth.value;
%             values.lAnkleGirth = self.l_ankle_girth.value;
%             values.lArmLength = self.leftArmLength;
%             values.rArmLength = self.rightArmLength;
%             values.trunkLength = self.trunkLength;
%             values.lLegLength = self.lLegLength;
%             values.rLegLength = self.rLegLength;
%             values.crotchHeight = self.crotchHeight;
%             values.collarScalpLength = self.collarScalpLength;
%             values.volume = self.volume;
%             values.surfaceArea = self.surfaceArea.total;
%             values.headSA = self.surfaceArea.head;
%             values.trunkSA = self.surfaceArea.trunk;
%             values.lArmSA = self.surfaceArea.lArm;
%             values.rArmSA = self.surfaceArea.rArm;
%             values.lLegSA = self.surfaceArea.lLeg;
%             values.rLegSA = self.surfaceArea.rLeg;
            
        end
        
        function values = extractEllipseValues(self)
            values.chestGirth = self.chestCircumference.ellipseValue;
            values.waistGirth = self.waistCircumference.ellipseValue;
            values.hipGirth = self.hipCircumference.ellipseValue;
            values.rThighGirth = self.rThighGirth.ellipseValue;
            values.lThighGirth = self.lThighGirth.ellipseValue;
            values.rCalfGirth = self.rCalfCircumference.ellipseValue;
            values.lCalfGirth = self.lCalfCircumference.ellipseValue;
            values.lWristGirth = self.l_wristgirth.ellipseValue;
            values.rWristGirth = self.r_wristgirth.ellipseValue;
            values.rForearmGirth = self.r_forearmgirth.ellipseValue;
            values.lForearmGirth = self.l_forearmgirth.ellipseValue;
            values.rBicepGirth = self.r_bicepgirth.ellipseValue;
            values.lBicepGirth = self.l_bicepgirth.ellipseValue;
            values.rAnkleGirth = self.r_ankle_girth.ellipseValue;
            values.lAnkleGirth = self.l_ankle_girth.ellipseValue;
            values.lArmLength = self.leftArmLength;
            values.rArmLength = self.rightArmLength;
            values.trunkLength = self.trunkLength;
            values.lLegLength = self.lLegLength;
            values.rLegLength = self.rLegLength;
            values.crotchHeight = self.crotchHeight;
            values.collarScalpLength = self.collarScalpLength;
            values.volume = self.volume;
            values.surfaceArea = self.surfaceArea.total;
            values.headSA = self.surfaceArea.head;
            values.trunkSA = self.surfaceArea.trunk;
            values.lArmSA = self.surfaceArea.lArm;
            values.rArmSA = self.surfaceArea.rArm;
            values.lLegSA = self.surfaceArea.lLeg;
            values.rLegSA = self.surfaceArea.rLeg;
        end

        function values = extractTemplateValues(self)
            values.chestGirth = self.chestCircumference.templateValue;
            values.waistGirth = self.waistCircumference.templateValue;
            values.hipGirth = self.hipCircumference.templateValue;
            values.rThighGirth = self.rThighGirth.templateValue;
            values.lThighGirth = self.lThighGirth.templateValue;
            values.rCalfGirth = self.rCalfCircumference.templateValue;
            values.lCalfGirth = self.lCalfCircumference.templateValue;
            values.lWristGirth = self.l_wristgirth.templateValue;
            values.rWristGirth = self.r_wristgirth.templateValue;
            values.rForearmGirth = self.r_forearmgirth.templateValue;
            values.lForearmGirth = self.l_forearmgirth.templateValue;
            values.rBicepGirth = self.r_bicepgirth.templateValue;
            values.lBicepGirth = self.l_bicepgirth.templateValue;
            values.rAnkleGirth = self.r_ankle_girth.templateValue;
            values.lAnkleGirth = self.l_ankle_girth.templateValue;
            values.lArmLength = self.leftArmLength;
            values.rArmLength = self.rightArmLength;
            values.trunkLength = self.trunkLength;
            values.lLegLength = self.lLegLength;
            values.rLegLength = self.rLegLength;
            values.crotchHeight = self.crotchHeight;
            values.collarScalpLength = self.collarScalpLength;
            values.volume = self.volume;
            values.surfaceArea = self.surfaceArea.total;
            values.headSA = self.surfaceArea.head;
            values.trunkSA = self.surfaceArea.trunk;
            values.lArmSA = self.surfaceArea.lArm;
            values.rArmSA = self.surfaceArea.rArm;
            values.lLegSA = self.surfaceArea.lLeg;
            values.rLegSA = self.surfaceArea.rLeg;
        end
        
        function [markers] = getMarkers(self)
        % markers = getMarkers(self) returns a list of all the markers
            markers = [struct('point',self.nose_tip,            'label','nose_tip',         'front',1)
                       struct('point',self.rwrist_front,        'label','rwrist_front',     'front',1)
                       struct('point',self.rwrist_back,        'label','rwrist_back',      'front',0)
                       struct('point',self.rwrist_lateral,      'label','rwrist_lateral',   'front',1)
                       struct('point',self.rwrist_medial,       'label','rwrist_medial',    'front',1)
                       struct('point',self.lwrist_front,        'label','lwrist_front',     'front',1)
                       struct('point',self.lwrist_back,         'label','lwrist_back',      'front',0)
                       struct('point',self.lwrist_lateral,      'label','lwrist_lateral',   'front',1)
                       struct('point',self.lwrist_medial,       'label','lwrist_medial',    'front',1)
                       struct('point',self.rforearm_front,      'label','rforearm_front',   'front',1)
                       struct('point',self.rforearm_back,       'label','rforearm_back',    'front',0)
                       struct('point',self.rforearm_lateral,    'label','rforearm_lateral', 'front',1)
                       struct('point',self.rforearm_medial,     'label','rforearm_medial',  'front',1)
                       struct('point',self.lforearm_front,      'label','lforearm_front',   'front',1)
                       struct('point',self.lforearm_back,       'label','lforearm_back',    'front',0)
                       struct('point',self.lforearm_lateral,    'label','lforearm_lateral', 'front',1)
                       struct('point',self.lforearm_medial,     'label','lforearm_medial',  'front',1)
                       struct('point',self.rbicep_front,        'label','rbicep_front',     'front',1)
                       struct('point',self.rbicep_back,         'label','rbicep_back',      'front',0)
                       struct('point',self.rbicep_lateral,      'label','rbicep_lateral',   'front',1)
                       struct('point',self.rbicep_medial,       'label','rbicep_medial',    'front',1)
                       struct('point',self.lbicep_front,        'label','lbicep_front',     'front',1)
                       struct('point',self.lbicep_back,         'label','lbicep_back',      'front',0)
                       struct('point',self.lbicep_lateral,      'label','lbicep_lateral',   'front',1)
                       struct('point',self.lbicep_medial,       'label','lbicep_medial',    'front',1)
                       struct('point',self.lowerBack,           'label','lowerBack',        'front',0)
                       struct('point',self.r_hip,               'label','r_hip',            'front',1)
                       struct('point',self.l_hip,               'label','l_hip',            'front',1)
                       struct('point',self.rtoe_tip,            'label','rtoe_tip',         'front',1)
                       struct('point',self.rheel_tip,           'label','rheel_tip',        'front',0)
                       struct('point',self.ltoe_tip,            'label','ltoe_tip',         'front',1)
                       struct('point',self.lheel_tip,           'label','lheel_tip',        'front',0)
                       struct('point',self.r_ankle,             'label','r_ankle',          'front',1)
                       struct('point',self.rankle_medialPoint,  'label','rankle_medial',    'front',1)
                       struct('point',self.rankle_lateralPoint, 'label','rankle_lateral',   'front',1)
                       struct('point',self.l_ankle,             'label','l_ankle',          'front',1)
                       struct('point',self.lankle_medialPoint,  'label','lankle_medial',    'front',1)
                       struct('point',self.lankle_lateralPoint, 'label','lankle_latera',    'front',1)
                       struct('point',self.rcalf_backPoint,     'label','rcalf_back',       'front',0)
                       struct('point',self.rcalf_medialPoint,   'label','rcalf_medial',     'front',1)
                       struct('point',self.rcalf_lateralPoint,  'label','rcalf_latera',     'front',1)
                       struct('point',self.lcalf_backPoint,     'label','lcalf_back',       'front',0)
                       struct('point',self.lcalf_medialPoint,   'label','lcalf_medial',     'front',1)
                       struct('point',self.lcalf_lateralPoint,  'label','lcalf_lateralt',   'front',1)
                       struct('point',self.rthigh_front,        'label','rthigh_front',     'front',1)
                       struct('point',self.rthigh_back,         'label','rthigh_back',      'front',0)
                       struct('point',self.rthigh_lateral,      'label','rthigh_lateral',   'front',1)
                       struct('point',self.rthigh_medial,       'label','rthigh_medial',    'front',1)
                       struct('point',self.lthigh_front,        'label','lthigh_front',     'front',1)
                       struct('point',self.lthigh_back,         'label','lthigh_back',      'front',0)
                       struct('point',self.lthigh_lateral,      'label','lthigh_lateral',   'front',1)
                       struct('point',self.lthigh_medial,       'label','lthigh_medial',    'front',1)
                       struct('point',self.crotch,              'label','crotch',           'front',1)
                       struct('point',self.rShoulder,           'label','rShoulder',        'front',1)
                       struct('point',self.lShoulder,           'label','lShoulder',        'front',1)
                       struct('point',self.r_armpit,            'label','r_armpit',         'front',1)
                       struct('point',self.l_armpit,            'label','l_armpit',         'front',1)];
        end
        
        function  createMarkers_template(self, input)
            % Create template F file
            
            extPos = find(input == '.', 1, 'last');
            output = [input(1:extPos-1) '.mkr'];
            
            markers = self.getMarkers();
            
            fid = fopen(output,'w');
            fprintf(fid,'1\r\n');
            fprintf(fid,'33\r\n');
            for i = 1:length(markers)
                marker = markers(i);
                [~,idx] = min(normAll(bsxfun(@minus,self.v,marker.point)));
                fprintf(fid, '%d 2  %d %d %d 1 0 0\r\n', i, idx , idx, idx );
            end
            
            fclose(fid);
        end
        
        function  createMarkers_Txt_Jpg(self, input)
        % Create marker file and pdf for scan
            
            extPos = find(input == '.', 1, 'last');
            outputMKR = [input(1:extPos-1) '.mkr'];
            
            markers = self.getMarkers();
            
            % Write mkr file
            fid = fopen(outputMKR,'w');
            fprintf(fid,'1\r\n');
            fprintf(fid,'33\r\n');
            for i = 1:length(markers)
                marker = markers(i);
                fprintf(fid, '%d_%s 0 %f %f %f\r\n', i, marker.label, marker.point(1) , marker.point(2), marker.point(3));
            end
            fclose(fid);
            
            % Create 2D .pdf file
            outputPDF = [input(1:extPos-1) '.pdf'];
            
            scan = figure('visible','off');
            gca.Colormap = [1 1 1];
            gca.GridLineStyle = 'none';
            gca.LineStyle = 'none';
            gca.LineColor = [1 1 1];
            
            %set(gca,'GridLineStyle', 'none','Colormap',[1 1 1])
            
            subplot(1,2,1)
            title('Front view')
            hold on
            lArmVertices = self.v(self.lArmIdx,:);
            plot(lArmVertices(:,1),lArmVertices(:,3),'.','Color','b','MarkerSize', 30); hold on;
            rArmVertices = self.v(self.rArmIdx,:);
            plot(rArmVertices(:,1),rArmVertices(:,3),'.','Color','b','MarkerSize', 30); hold on;
            legVertices = self.v(self.legIdx,:);
            plot(legVertices(:,1),legVertices(:,3),'.','Color','b','MarkerSize', 30); hold on;
            headVertices = self.v(self.headIdx,:);
            plot(headVertices(:,1),headVertices(:,3),'.','Color','b','MarkerSize', 30); hold on;
            trunkVertices = self.v(self.trunkIdx,:);
            plot(trunkVertices(:,1),trunkVertices(:,3),'.','Color','b','MarkerSize', 30);

            for i = 1:length(markers)
                marker = markers(i);
                if marker.front
                    plot(marker.point(1), marker.point(3), '.', 'Color','m', 'MarkerSize', 15)
                    labelpoints(marker.point(1), marker.point(3), marker.label, 'color', 'm', 'FontSize',5)
                end
            end
            hold off
            
            subplot(1,2,2)
            title('Back view')
            hold on
            lArmVertices = self.v(self.lArmIdx,:);
            plot(-lArmVertices(:,1),lArmVertices(:,3),'.','Color','b','MarkerSize', 30); hold on;
            rArmVertices = self.v(self.rArmIdx,:);
            plot(-rArmVertices(:,1),rArmVertices(:,3),'.','Color','b','MarkerSize', 30); hold on;
            legVertices = self.v(self.legIdx,:);
            plot(-legVertices(:,1),legVertices(:,3),'.','Color','b','MarkerSize', 30); hold on;
            headVertices = self.v(self.headIdx,:);
            plot(-headVertices(:,1),headVertices(:,3),'.','Color','b','MarkerSize', 30); hold on;
            trunkVertices = self.v(self.trunkIdx,:);
            plot(-trunkVertices(:,1),trunkVertices(:,3),'.','Color','b','MarkerSize', 30);
            
            for i = 1:length(markers)
                marker = markers(i);
                if ~marker.front
                    plot(-marker.point(1), marker.point(3), '.', 'Color','m', 'MarkerSize', 15)
                    labelpoints(-marker.point(1), marker.point(3), marker.label, 'color', 'm', 'FontSize',5)
                end
            end
            hold off
            
            saveas(scan, outputPDF)
        end

        function plot2d(self,keyPoints)
        % Plots the avatar in 2D front view
            if ~exist('keyPoints','var')
                keyPoints = true;
            end

            %figure; plot(self.v(:, 1),self.v(:, 3),'.');hold on;
            if keyPoints == true
                plot(self.r_wrist(1), self.r_wrist(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.r_armpit(1), self.r_armpit(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.r_hip(1), self.r_hip(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.l_armpit(1), self.l_armpit(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.l_wrist(1), self.l_wrist(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.l_hip(1), self.l_hip(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.crotch(1), self.crotch(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
%                 plot(self.k10(1), self.k10(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
%                 plot(self.k11(1), self.k11(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.lShoulder(1), self.lShoulder(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.rShoulder(1), self.rShoulder(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.l_ankle(1), self.l_ankle(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.r_ankle(1), self.r_ankle(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
                plot(self.collar(1), self.collar(3),'r--o','LineWidth',4,'MarkerSize',2); hold on;
            end
            axis tight
            axis equal
        end
        
        function plot3d(self)
        % Plots the avatar in 3D
%             figure;
            lArmVertices = self.v(self.lArmIdx,:);
            plot3(lArmVertices(:,1),lArmVertices(:,2),lArmVertices(:,3),'.'); hold on;
            rArmVertices = self.v(self.rArmIdx,:);
            plot3(rArmVertices(:,1),rArmVertices(:,2),rArmVertices(:,3),'.'); hold on;
            legVertices = self.v(self.legIdx,:);
            plot3(legVertices(:,1),legVertices(:,2),legVertices(:,3),'.'); hold on;
            headVertices = self.v(self.headIdx,:);
            plot3(headVertices(:,1),headVertices(:,2),headVertices(:,3),'.'); hold on;
            trunkVertices = self.v(self.trunkIdx,:);
            plot3(trunkVertices(:,1),trunkVertices(:,2),trunkVertices(:,3),'.');
        end
            
        function plot3d_2(self)
            % left arm
            lArmFaces = getFaces(self.f,self.lArmIdx);
            
            % right arm
            rArmFaces = getFaces(self.f,self.rArmIdx);

            % head
            headFaces = getFaces(self.f, self.headIdx);

            % legs
            legFaces = getFaces(self.f, self.legIdx);
            
            % trunk
            trunkFaces = getFaces(self.f, self.trunkIdx);

            %figure;
            h_lArm = patch('vertices', self.v, 'faces', lArmFaces, 'FaceColor', 'y');
            h_rArm = patch('vertices', self.v, 'faces', rArmFaces, 'FaceColor', 'm');
            h_head = patch('vertices', self.v, 'faces', headFaces, 'FaceColor', 'c');
            h_legs = patch('vertices', self.v, 'faces', legFaces, 'FaceColor', 'g');
            h_trunk = patch('vertices', self.v, 'faces', trunkFaces, 'FaceColor', 'b');

            set(h_lArm,'LineStyle','none')
            set(h_rArm,'LineStyle','none')
            set(h_head,'LineStyle','none')
            set(h_legs,'LineStyle','none')
            set(h_trunk,'LineStyle','none')
            
            patch(self.r_wristgirth.points(:,1),self.r_wristgirth.points(:,2),self.r_wristgirth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.l_wristgirth.points(:,1),self.l_wristgirth.points(:,2),self.l_wristgirth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.hipCircumference.points(:,1),self.hipCircumference.points(:,2),self.hipCircumference.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.waistCircumference.points(:,1),self.waistCircumference.points(:,2),self.waistCircumference.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.r_ankle_girth.points(:,1),self.r_ankle_girth.points(:,2),self.r_ankle_girth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.l_ankle_girth.points(:,1),self.l_ankle_girth.points(:,2),self.l_ankle_girth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.rCalfCircumference.points(:,1),self.rCalfCircumference.points(:,2),self.rCalfCircumference.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.lCalfCircumference.points(:,1),self.lCalfCircumference.points(:,2),self.lCalfCircumference.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.rThighGirth.points(:,1),self.rThighGirth.points(:,2),self.rThighGirth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.lThighGirth.points(:,1),self.lThighGirth.points(:,2),self.lThighGirth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.r_forearmgirth.points(:,1),self.r_forearmgirth.points(:,2),self.r_forearmgirth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.l_forearmgirth.points(:,1),self.l_forearmgirth.points(:,2),self.l_forearmgirth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.r_bicepgirth.points(:,1),self.r_bicepgirth.points(:,2),self.r_bicepgirth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.l_bicepgirth.points(:,1),self.l_bicepgirth.points(:,2),self.l_bicepgirth.points(:,3),'r','EdgeColor','r','LineWidth',2)
            patch(self.chestCircumference.points(:,1),self.chestCircumference.points(:,2),self.chestCircumference.points(:,3),'r','EdgeColor','r','LineWidth',2)
            
            light('Position',[-50,-50,50],'Style','infinite');
            light('Position',[50,50,50],'Style','infinite');
            lighting phong;
            axis tight
            axis equal
            view([0 0]);
        end
        function plot3d_3(self)
            % left arm
            lArmFaces = getFaces(self.f,self.lArmIdx);
            
            % right arm
            rArmFaces = getFaces(self.f,self.rArmIdx);

            % head
            headFaces = getFaces(self.f, self.headIdx);

            % legs
            legFaces = getFaces(self.f, self.legIdx);
            
            % trunk
            trunkFaces = getFaces(self.f, self.trunkIdx);

            %figure;
            h_lArm = patch('vertices', self.v, 'faces', lArmFaces, 'FaceColor', 'y');
            h_rArm = patch('vertices', self.v, 'faces', rArmFaces, 'FaceColor', 'm');
            h_head = patch('vertices', self.v, 'faces', headFaces, 'FaceColor', 'c');
            h_legs = patch('vertices', self.v, 'faces', legFaces, 'FaceColor', 'g');
            h_trunk = patch('vertices', self.v, 'faces', trunkFaces, 'FaceColor', 'b');

            set(h_lArm,'LineStyle','none')
            set(h_rArm,'LineStyle','none')
            set(h_head,'LineStyle','none')
            set(h_legs,'LineStyle','none')
            set(h_trunk,'LineStyle','none')
            
            
            light('Position',[-50,-50,50],'Style','infinite');
            light('Position',[50,50,50],'Style','infinite');
            lighting phong;
            axis tight
            axis equal
            view([0 0]);
        end
        
        function plotCurve(self,d)
            if d == 1
                distneg = self.rcurve;
                distpos = self.lcurve;
                dist = distpos - distneg;
            elseif d == 2
                distneg = self.fcurve;
                distpos = self.bcurve;
                dist = distpos - distneg;
            end
            n = size(dist);
            
            %figure; 
            subplot(1,5,[2,3]); 
            plot(self.v(:, d),self.v(:, 3),'.'); 
            title('2D Image of Subject');
            hold on
            
            subplot(1,5,1)
            plot(distneg, 1:n,'-d')
            title('Length Negative - Center')
            ylabel('Line Number')
            
            subplot(1,5,4)
            plot(distpos, 1:n,'-d')
            title('Length Center - Positive')
            
            subplot(1,5,5)
            plot(dist, 1:n,'-d')
            title('Length of Line')
        end
        
        function plotAll(self)
            plot2d(self);
            plot3d(self);
            plotCurve(self,1);
            plotCurve(self,2);
        end
        
        function plot2d_gui(self,axes)
        % Plots the avatar in 2D front view

                plot(axes,self.v(:, 1),self.v(:, 3),'.');
                hold(axes,'on')
                    plot(axes,self.r_wrist(1), self.r_wrist(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.r_armpit(1), self.r_armpit(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.r_hip(1), self.r_hip(3),'g--o','LineWidth',4,'MarkerSize',2);
                    plot(axes,self.l_armpit(1), self.l_armpit(3),'g--o','LineWidth',4,'MarkerSize',2);
                    plot(axes,self.l_wrist(1), self.l_wrist(3),'g--o','LineWidth',4,'MarkerSize',2);
                    plot(axes,self.l_hip(1), self.l_hip(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.crotch(1), self.crotch(3),'g--o','LineWidth',4,'MarkerSize',2);
%                     plot(axes,self.k10(1), self.k10(3),'g--o','LineWidth',4,'MarkerSize',2); 
%                     plot(axes,self.k11(1), self.k11(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.lShoulder(1), self.lShoulder(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.rShoulder(1), self.rShoulder(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.collar(1), self.collar(3),'g--o','LineWidth',4,'MarkerSize',2);
                hold off
            end
        
        function plot3d_gui(self,axes)
            % left arm
            lArmFaces = getFaces(self,self.lArmIdx);
            
            % right arm
            rArmFaces = getFaces(self,self.rArmIdx);

            % head
            headFaces = getFaces(self, self.headIdx);

            % legs
            legFaces = getFaces(self, self.legIdx);
            
            % trunk
            trunkFaces = getFaces(self, self.trunkIdx);
            
            h_lArm = patch(axes,'vertices', self.v, 'faces', lArmFaces, 'FaceColor', 'y');
            h_rArm = patch(axes,'vertices', self.v, 'faces', rArmFaces, 'FaceColor', 'm');
            h_head = patch(axes,'vertices', self.v, 'faces', headFaces, 'FaceColor', 'c');
            h_legs = patch(axes,'vertices', self.v, 'faces', legFaces, 'FaceColor', 'g');
            h_trunk = patch(axes,'vertices', self.v, 'faces', trunkFaces, 'FaceColor', 'b');

            set(h_lArm,'LineStyle','none')
            set(h_rArm,'LineStyle','none')
            set(h_head,'LineStyle','none')
            set(h_legs,'LineStyle','none')
            set(h_trunk,'LineStyle','none')

            light('Position',[-50,-50,50],'Style','infinite');
            light('Position',[50,50,50],'Style','infinite');
            lighting phong;
        end
         
        function plot3d_points_gui(self,axes)
        % Plots the avatar in 3D
            
            lArmVertices = self.v(self.lArmIdx,:);
            hold(axes,'on');
            plot3(axes,lArmVertices(:,1),lArmVertices(:,2),lArmVertices(:,3),'.'); 
            rArmVertices = self.v(self.rArmIdx,:);
            plot3(axes,rArmVertices(:,1),rArmVertices(:,2),rArmVertices(:,3),'.'); 
            legVertices = self.v(self.legIdx,:);
            plot3(axes,legVertices(:,1),legVertices(:,2),legVertices(:,3),'.'); 
            headVertices = self.v(self.headIdx,:);
            plot3(axes,headVertices(:,1),headVertices(:,2),headVertices(:,3),'.'); 
            trunkVertices = self.v(self.trunkIdx,:);
            plot3(axes,trunkVertices(:,1),trunkVertices(:,2),trunkVertices(:,3),'.');
            hold(axes,'off');
        end
        
        function plotCurve_gui(self,d,laxes,caxes,raxes,daxes)
            if d == 1
                distneg = self.rcurve;
                distpos = self.lcurve;
                dist = distpos - distneg;
            elseif d == 2
                distneg = self.fcurve;
                distpos = self.bcurve;
                dist = distpos - distneg;
            end
            n = size(dist);
            plot(caxes,self.v(:, d),self.v(:, 3),'.'); 
            
           
            plot(laxes,distneg, 1:n,'-d')
            
            
            
            plot(raxes,distpos, 1:n,'-d') 
            
            plot(daxes,dist, 1:n,'-d')
        end
        
        function plot_simplePartition(self)
        
            % left arm
            lArmFaces = getFaces(self.f,self.lArmIdx);
            
            % right arm
            rArmFaces = getFaces(self.f,self.rArmIdx);

            % head
            topIdx = [];
            
                maxArmpitZ = max(self.r_armpit(3), self.l_armpit(3));
                topIdx = uint32(find(self.v(:,3) > maxArmpitZ));
            
                % we don't want parts of the arms
                armIdx = [self.lArmIdx; self.rArmIdx];
                [~, ia, ~] = intersect(topIdx, armIdx);
                topIdx(ia) = [];
            
            topFaces = getFaces(self.f, topIdx);

            % legs
            v3 = self.v(:,3);
            both_legs = find(v3 <= self.crotch(3));
			armIdx = [self.lArmIdx; self.rArmIdx];
            [~, ia, ~] = intersect(both_legs, armIdx);
            both_legs(ia) = [];			            
            lLegIdx = both_legs(self.v(both_legs,1)>self.crotch(1));
            rLegIdx = both_legs(self.v(both_legs,1)<=self.crotch(1));            
            rLegFaces = getFaces(self.f, rLegIdx);
            lLegFaces = getFaces(self.f, lLegIdx);
            
            % trunk
            centerIdx = (uint32(1:size(self.v,1)))';
            nonCenterIdx = [self.lArmIdx; self.rArmIdx; ...
                           both_legs; topIdx];
            centerIdx(nonCenterIdx) = [];            
            centerFaces = getFaces(self.f, centerIdx);
            
%             figure; hold on;
%             patch('vertices', self.v, 'faces', lArmFaces, 'FaceColor', 'y','LineStyle','none');
%             patch('vertices', self.v, 'faces', rArmFaces, 'FaceColor', 'm','LineStyle','none');
%             patch('vertices', self.v, 'faces', topFaces, 'FaceColor', 'c','LineStyle','none');
%             patch('vertices', self.v, 'faces', lLegFaces, 'FaceColor', 'g','LineStyle','none');
%             patch('vertices', self.v, 'faces', rLegFaces, 'FaceColor', 'r','LineStyle','none');
%             patch('vertices', self.v, 'faces', centerFaces, 'FaceColor', 'b','LineStyle','none');
% 
%             patch(self.r_wristgirth.points(:,1),self.r_wristgirth.points(:,2),self.r_wristgirth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.l_wristgirth.points(:,1),self.l_wristgirth.points(:,2),self.l_wristgirth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.hipCircumference.points(:,1),self.hipCircumference.points(:,2),self.hipCircumference.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.waistCircumference.points(:,1),self.waistCircumference.points(:,2),self.waistCircumference.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.r_ankle_girth.points(:,1),self.r_ankle_girth.points(:,2),self.r_ankle_girth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.l_ankle_girth.points(:,1),self.l_ankle_girth.points(:,2),self.l_ankle_girth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.rCalfCircumference.points(:,1),self.rCalfCircumference.points(:,2),self.rCalfCircumference.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.lCalfCircumference.points(:,1),self.lCalfCircumference.points(:,2),self.lCalfCircumference.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.rThighGirth.points(:,1),self.rThighGirth.points(:,2),self.rThighGirth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.lThighGirth.points(:,1),self.lThighGirth.points(:,2),self.lThighGirth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.r_forearmgirth.points(:,1),self.r_forearmgirth.points(:,2),self.r_forearmgirth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.l_forearmgirth.points(:,1),self.l_forearmgirth.points(:,2),self.l_forearmgirth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.r_bicepgirth.points(:,1),self.r_bicepgirth.points(:,2),self.r_bicepgirth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.l_bicepgirth.points(:,1),self.l_bicepgirth.points(:,2),self.l_bicepgirth.points(:,3),'k','EdgeColor','k','LineWidth',.2)
%             patch(self.chestCircumference.points(:,1),self.chestCircumference.points(:,2),self.chestCircumference.points(:,3),'k','EdgeColor','k','LineWidth',.2)       
%             
%             light('Position',[-50,-50,50],'Style','infinite');
%             light('Position',[50,50,50],'Style','infinite');
%             lighting phong;
        end
    end
end

%% other functions
function [newV,newF] = meshPoly(v,e)
% [newV,newF] = meshPoly(v,e) creates a mesh for a given polynomial. It
% uses the centroid of all edges to create a new face. newV will be the
% centoid. This is a very simple mesh and faces could be overlapping.
    polyV = unique(e(:));
    centroid = mean(v(polyV,:));
    newF = [e(:,2), e(:,1), repmat(length(v)+1,length(e),1)];
    newV = centroid;
end

function [p1, p3] = find_minmax(v1, v3, left, right,size)
    x_pts = linspace(left,right,size);      %Partiton x-interval from m to M with 101 points (100 regions)
    mn_v3 = zeros(length(x_pts)-1,1);   %holds min z-coordinates for each partition
    corr_v1 = zeros(length(x_pts)-1,1); %holds corresponding x-coordinate for min z-coordinate 

    for i=1:length(x_pts)-1               %For each region 
        portion_v3 = v3(x_pts(i)<v1 & v1<x_pts(i+1)); %The z-coordinates that are available in each subinterval 
        portion_v1 = v1(x_pts(i)<v1 & v1<x_pts(i+1));   %x-coordinates that correspond to z-coordinates on portion_v3
        if isempty(portion_v3)
            mn_v3(i) = -Inf;
        else
            [mn_v3(i), mn_v3_Idx]= min(portion_v3); %Find min z-coordinate in subinterval and index
            corr_v1(i) = portion_v1(mn_v3_Idx);     %gives x_coordinate that corresponds to min z-coordinate
        end
    end
    [p3, mxIdx_mn_v3] = max(mn_v3);    %Finds z-coordinate and index of crotch
    p1 = corr_v1(mxIdx_mn_v3);     %Gives x-coordinate of crotch
end
function output = sosmooth3(x,N) %%N is odd
    h = ones(N,1);
    output = conv(h,x);
    a = [(1:N)';N.*ones(length(output)-(2*N),1);(N:-1:1)'];
    output = output./a;
    n = (N-1)/2;
    cut = output(2*n+1:end-(2*n));
    for i = 1:n
        cut = [output(2*(n-i+1)-1);cut;output(2*(n-1))];
    end
    output = cut;
end
function v = crossAll(v1,v2)
    % Finds the cross product for all pairs
    % v(1,:) = cross(v1(1,:),v2(1,:))
    % ...
    v = NaN(length(v1),3);
    v(:,1) = v1(:,2).*v2(:,3)-v2(:,2).*v1(:,3);
    v(:,2) = v1(:,3).*v2(:,1)-v2(:,3).*v1(:,1);
    v(:,3) = v1(:,1).*v2(:,2)-v2(:,1).*v1(:,2);
end
function norms = normAll(M)
% Finds the norm for all rows
    norms = sqrt(sum(M.^2,2));
end
function [c,b] = getCircumference(x,y)
    b = convhull(x,y);
    t_boundary = [x(b),y(b)];
    c = sum(normAll(t_boundary(1:end-1,:)-t_boundary(2:end,:)));
end
function [volume] = SignedVolumeOfTriangle(v1,v2,v3)
%     v321 = v3(:,1) .* v2(:,2) .* v1(:,3);
%     v231 = v2(:,1) .* v3(:,2) .* v1(:,3);
%     v312 = v3(:,1) .* v1(:,2) .* v2(:,3);
%     v132 = v1(:,1) .* v3(:,2) .* v2(:,3);
%     v213 = v2(:,1) .* v1(:,2) .* v3(:,3);
%     v123 = v1(:,1) .* v2(:,2) .* v3(:,3);
%     volume = (1/6)*(-v321+v231+v312-v132-v213+v123);
    volume=(1/6)*dot(cross(v1,v2),v3);
end
function bdyEdges = getBoundaryEdges(faces)
% bdyEdges = getBoundaryEdges(faces) returns all edges that only occur once

    % All edges
    E = [faces(:,[1,2]);faces(:,[2,3]);faces(:,[3,1])];
    % Sort each row
    Esort = sort(E,2);
    % Count number of occurences of each edge
    [~,IE,IU] = unique(Esort,'rows');
    counts = accumarray(IU(:), 1);
    % Unique edges not sorted
    U = E(IE,:);
    % extract edges that only occurred once
    bdyEdges = U(counts==1,:);
end
function [f,v] = removeBoundaryProblems(f,v)
% [f,v] = removeBoundaryProblems(f,v) removes all faces that have a vertex
% on the boundary that does not occur exactly twice.
    numRemoved = 0;
    initialLength = length(f);
    warningSent = false;
    bdyEdges = getBoundaryEdges(f);

    % Find all vertices that occur more than twice or just once
    [a,b]=hist(bdyEdges(:),double(unique(bdyEdges(:))));
    nottwo = b(a > 2 | a == 1);
    
    % Remove faces that have one of those vertices and repeat
    while nottwo
        [~,facesToRemove] = getFaces(f,nottwo);
        [f,v] = removeFace(f,v,facesToRemove);
        
        numRemoved = numRemoved + sum(facesToRemove);
        if (numRemoved > initialLength*0.1) && ~warningSent
            warning('removeBoundaryProblems has removed more than 10% of all faces')
        end
        bdyEdges = getBoundaryEdges(f);
        [a,b]=hist(bdyEdges(:),double(unique(bdyEdges(:))));
        nottwo = b(a > 2 | a == 1);
    end
end
function list = findHoles(bdyEdges)
%     pCases = [];
% 
%     vertices = unique([bdyEdges(:,1);bdyEdges(:,2)]);
%     for vertex = vertices'
%         if sum(sum(bdyEdges == vertex)) == d
%             pCases = [pCases;vertex];
%         end
%     end
    list = zeros(1,length(bdyEdges));
    holeNum = 1;
    row = 1;
    col = 1;
    list(row) = holeNum;
    start = row;
    while true
        col = mod(col,2)+1; % get other col
        [a,b] = find(bdyEdges == bdyEdges(row,col));
        if sum([a(1),b(1)] == [row,col]) == 2
            row = a(2);
            col = b(2);
        else
            row = a(1);
            col = b(1);
        end
        list(row) = holeNum;
        
        if sum(find(list == 0)) == 0, break; end
        
        if row == start;
            holeNum = holeNum + 1;
            row = find(list == 0, 1);
            start = row;
            col = 1;
        end
        
    end
end
function [semimajor_axis, semiminor_axis, x0, y0, phi] = ellipse_fit(x, y)
    % Programmed by: Tal Hendel <thendel@tx.technion.ac.il>
    % Faculty of Biomedical Engineering, Technion- Israel Institute of Technology     
    x = x(:);
    y = y(:);
    %Construct M
    M = [2*x.*y y.^2 2*x 2*y ones(size(x))];
    % Multiply (-X.^2) by pseudoinverse(M)
    e = M\(-x.^2);
    %Extract parameters from vector e
    a = 1;
    b = e(1);
    c = e(2);
    d = e(3);
    f = e(4);
    g = e(5);
    %Use Formulas from Mathworld to find semimajor_axis, semiminor_axis, x0, y0, and phi
    delta = b^2-a*c;
    x0 = (c*d - b*f)/delta;
    y0 = (a*f - b*d)/delta;
    phi = 0.5 * acot((c-a)/(2*b));
    nom = 2 * (a*f^2 + c*d^2 + g*b^2 - 2*b*d*f - a*c*g);
    s = sqrt(1 + (4*b^2)/(a-c)^2);
    a_prime = sqrt(nom/(delta* ( (c-a)*s -(c+a))));
    b_prime = sqrt(nom/(delta* ( (a-c)*s -(c+a))));
    semimajor_axis = max(a_prime, b_prime);
    semiminor_axis = min(a_prime, b_prime);
    if (a_prime < b_prime)
        phi = pi/2 - phi;
    end
   
    %%% Ellipse Circumference 
    
    
end
function [vertex,face] = read_ply(filename)

% read_ply - read data from PLY file.
%
%   [vertex,face] = read_ply(filename);
%
%   'vertex' is a 'nb.vert x 3' array specifying the position of the vertices.
%   'face' is a 'nb.face x 3' array specifying the connectivity of the mesh.
%
%   IMPORTANT: works only for triangular meshes.
%
%   Copyright (c) 2003 Gabriel Peyr?

[d,c] = plyread(filename);
try
vi = d.face.vertex_indices;
catch
    vi = d.face.vertex_index;
end 
nf = length(vi);
face = zeros(nf,3);
for i=1:nf
    face(i,:) = vi{i}+1;
end

vertex = [d.vertex.x, d.vertex.y, d.vertex.z];
end
 
function obj = readObj(fname)
    %
    % obj = readObj(fname)
    % This function can read styku or fit3d_proscanner file
    %
    % INPUT: fname - wavefront object file full path
    %
    % OUTPUT: obj.v - mesh vertices
    %       : obj.f - face definition assuming faces are made of of 3 vertices
    
    % Find number of headerlines
    fileID = fopen(fname);
    headerLines = 0;
    while 1
        tline = fgetl(fileID);
        ln = sscanf(tline,'%s',1); % line type
        if strcmp(ln,'v') || strcmp(ln,'f')
            break 
        end
        headerLines = headerLines + 1;
    end
    fclose(fileID);
    
    % Get v and f
    fileID = fopen(fname);
    v = zeros(0,3);
    f = zeros(0,3);
    
    try
        while(1)
            C = textscan(fileID,'%s %f %f %f','HeaderLines', headerLines,'CommentStyle','#');

            id = char(C{1});
            if isempty(id), break; end
            [~,idSize] = size(id);
            vIdx = id(:,1) == 'v';
            fIdx = id(:,1) == 'f';
            if idSize > 1
                vIdx = logical(vIdx .* (id(:,2) == ' '));
            end
            vn = sum(vIdx); % number of vertices
            fn = sum(fIdx); % number of faces

            vNew = zeros(vn,3);
            vNew(:,1) = C{2}(vIdx);
            vNew(:,2) = C{3}(vIdx);
            vNew(:,3) = C{4}(vIdx);
            v = [v; vNew];

            fNew = zeros(fn,3);
            fNew(:,1) = C{2}(fIdx);
            fNew(:,2) = C{3}(fIdx);
            fNew(:,3) = C{4}(fIdx);
            f = [f; fNew];
            if feof(fileID), break; end
            headerLines = 1;
            if idSize > 1, break; end
        end
        if ~feof(fileID)
            C = textscan(fileID,'%s %s %s %s', 'HeaderLines', 1);
            id = char(C{1});
            [~,idSize] = size(id);
            vIdx = id(:,1) == 'v';
            fIdx = id(:,1) == 'f';
            if idSize > 1
                vIdx = logical(vIdx .* (id(:,2) == ' '));
            end
            vn = sum(vIdx); % number of vertices
            fn = sum(fIdx); % number of faces

            vNew = zeros(vn,3);
            vNew(:,1) = str2double(C{2}(vIdx));
            vNew(:,2) = str2double(C{3}(vIdx));
            vNew(:,3) = str2double(C{4}(vIdx));
            v = [v; vNew];

            fNew = zeros(fn,3);
            a = C{2}(fIdx);
            b = C{3}(fIdx);
            c = C{4}(fIdx);
            fNew(:,1) = str2double(strtok(a,'/'));
            fNew(:,2) = str2double(strtok(b,'/'));
            fNew(:,3) = str2double(strtok(c,'/'));
            f = [f;fNew];
        end
        fclose(fileID);
    catch ME
        fclose(fileID);
        warning('An error occured with the message: %s\nAttempting to read obj file with regexp.\nYou may want to preprocess the obj file.', ME.message)
        % Reading obj file with regexp
        file = fileread(fname);
        match = regexp(file,'f\s(\d+)\S*\s(\d+)\S*\s(\d+)\S*(?:\n|\r|$)','tokens');
        f = str2double([match{:}]);
        f = reshape(f,3,[])';
        match = regexp(file,'v\s([-+]?\d+\.?\d*)\s([-+]?\d+\.?\d*)\s([-+]?\d+\.?\d*)','tokens');
        v = str2double([match{:}]);
        v = reshape(v,3,[])';
    end
    % set up matlab object 
    obj.v = v; obj.f = f;
end
function [Elements,varargout] = plyread(Path,Str)
%PLYREAD   Read a PLY 3D data file.
%   [DATA,COMMENTS] = PLYREAD(FILENAME) reads a version 1.0 PLY file
%   FILENAME and returns a structure DATA.  The fields in this structure
%   are defined by the PLY header; each element type is a field and each
%   element property is a subfield.  If the file contains any comments,
%   they are returned in a cell string array COMMENTS.
%
%   [TRI,PTS] = PLYREAD(FILENAME,'tri') or
%   [TRI,PTS,DATA,COMMENTS] = PLYREAD(FILENAME,'tri') converts vertex
%   and face data into triangular connectivity and vertex arrays.  The
%   mesh can then be displayed using the TRISURF command.
%
%   Note: This function is slow for large mesh files (+50K faces),
%   especially when reading data with list type properties.
%
%   Example:
%   [Tri,Pts] = PLYREAD('cow.ply','tri');
%   trisurf(Tri,Pts(:,1),Pts(:,2),Pts(:,3)); 
%   colormap(gray); axis equal;
%
%   See also: PLYWRITE

% Pascal Getreuer 2004

[fid,Msg] = fopen(Path,'rt');	% open file in read text mode

if fid == -1, error(Msg); end

Buf = fscanf(fid,'%s',1);
if ~strcmp(Buf,'ply')
   fclose(fid);
   error('Not a PLY file.'); 
end


%%% read header %%%

Position = ftell(fid);
Format = '';
NumComments = 0;
Comments = {};				% for storing any file comments
NumElements = 0;
NumProperties = 0;
Elements = [];				% structure for holding the element data
ElementCount = [];		% number of each type of element in file
PropertyTypes = [];		% corresponding structure recording property types
ElementNames = {};		% list of element names in the order they are stored in the file
PropertyNames = [];		% structure of lists of property names

while 1
   Buf = fgetl(fid);   								% read one line from file
   BufRem = Buf;
   Token = {};
   Count = 0;
   
   while ~isempty(BufRem)								% split line into tokens
      [tmp,BufRem] = strtok(BufRem);
      
      if ~isempty(tmp)
         Count = Count + 1;							% count tokens
         Token{Count} = tmp;
      end
   end
   
   if Count 		% parse line
      switch lower(Token{1})
      case 'format'		% read data format
         if Count >= 2
            Format = lower(Token{2});
            
            if Count == 3 & ~strcmp(Token{3},'1.0')
               fclose(fid);
               error('Only PLY format version 1.0 supported.');
            end
         end
      case 'comment'		% read file comment
         NumComments = NumComments + 1;
         Comments{NumComments} = '';
         for i = 2:Count
            Comments{NumComments} = [Comments{NumComments},Token{i},' '];
         end
      case 'element'		% element name
         if Count >= 3
            if isfield(Elements,Token{2})
               fclose(fid);
               error(['Duplicate element name, ''',Token{2},'''.']);
            end
            
            NumElements = NumElements + 1;
            NumProperties = 0;
   	      Elements = setfield(Elements,Token{2},[]);
            PropertyTypes = setfield(PropertyTypes,Token{2},[]);
            ElementNames{NumElements} = Token{2};
            PropertyNames = setfield(PropertyNames,Token{2},{});
            CurElement = Token{2};
            ElementCount(NumElements) = str2double(Token{3});
            
            if isnan(ElementCount(NumElements))
               fclose(fid);
               error(['Bad element definition: ',Buf]); 
            end            
         else
            error(['Bad element definition: ',Buf]);
         end         
      case 'property'	% element property
         if ~isempty(CurElement) & Count >= 3            
            NumProperties = NumProperties + 1;
            eval(['tmp=isfield(Elements.',CurElement,',Token{Count});'],...
               'fclose(fid);error([''Error reading property: '',Buf])');
            
            if tmp
               error(['Duplicate property name, ''',CurElement,'.',Token{2},'''.']);
            end            
            
            % add property subfield to Elements
            eval(['Elements.',CurElement,'.',Token{Count},'=[];'], ...
               'fclose(fid);error([''Error reading property: '',Buf])');            
            % add property subfield to PropertyTypes and save type
            eval(['PropertyTypes.',CurElement,'.',Token{Count},'={Token{2:Count-1}};'], ...
               'fclose(fid);error([''Error reading property: '',Buf])');            
            % record property name order 
            eval(['PropertyNames.',CurElement,'{NumProperties}=Token{Count};'], ...
               'fclose(fid);error([''Error reading property: '',Buf])');
         else
            fclose(fid);
            
            if isempty(CurElement)            
               error(['Property definition without element definition: ',Buf]);
            else               
               error(['Bad property definition: ',Buf]);
            end            
         end         
      case 'end_header'	% end of header, break from while loop
         break;		
      end
   end
end

%%% set reading for specified data format %%%

if isempty(Format)
	warning('Data format unspecified, assuming ASCII.');
   Format = 'ascii';
end

switch Format
case 'ascii'
   Format = 0;
case 'binary_little_endian'
   Format = 1;
case 'binary_big_endian'
   Format = 2;
otherwise
   fclose(fid);
   error(['Data format ''',Format,''' not supported.']);
end

if ~Format   
   Buf = fscanf(fid,'%f');		% read the rest of the file as ASCII data
   BufOff = 1;
else
   % reopen the file in read binary mode
   fclose(fid);
   
   if Format == 1
      fid = fopen(Path,'r','ieee-le.l64');		% little endian
   else
      fid = fopen(Path,'r','ieee-be.l64');		% big endian
   end
   
   % find the end of the header again (using ftell on the old handle doesn't give the correct position)   
   BufSize = 8192;
   Buf = [blanks(10),char(fread(fid,BufSize,'uchar')')];
   i = [];
   tmp = -11;
   
   while isempty(i)
   	i = findstr(Buf,['end_header',13,10]);			% look for end_header + CR/LF
   	i = [i,findstr(Buf,['end_header',10])];		% look for end_header + LF
      
      if isempty(i)
         tmp = tmp + BufSize;
         Buf = [Buf(BufSize+1:BufSize+10),char(fread(fid,BufSize,'uchar')')];
      end
   end
   
   % seek to just after the line feed
   fseek(fid,i + tmp + 11 + (Buf(i + 10) == 13),-1);
end


%%% read element data %%%

% PLY and MATLAB data types (for fread)
PlyTypeNames = {'char','uchar','short','ushort','int','uint','float','double', ...
   'char8','uchar8','short16','ushort16','int32','uint32','float32','double64'};
MatlabTypeNames = {'schar','uchar','int16','uint16','int32','uint32','single','double'};
SizeOf = [1,1,2,2,4,4,4,8];	% size in bytes of each type

for i = 1:NumElements
   % get current element property information
   eval(['CurPropertyNames=PropertyNames.',ElementNames{i},';']);
   eval(['CurPropertyTypes=PropertyTypes.',ElementNames{i},';']);
   NumProperties = size(CurPropertyNames,2);
   
%   fprintf('Reading %s...\n',ElementNames{i});
      
   if ~Format	%%% read ASCII data %%%
      for j = 1:NumProperties
         Token = getfield(CurPropertyTypes,CurPropertyNames{j});
         
         if strcmpi(Token{1},'list')
            Type(j) = 1;
         else
            Type(j) = 0;
			end
      end
      
      % parse buffer
      if ~any(Type)
         % no list types
         Data = reshape(Buf(BufOff:BufOff+ElementCount(i)*NumProperties-1),NumProperties,ElementCount(i))';
         BufOff = BufOff + ElementCount(i)*NumProperties;
      else
         ListData = cell(NumProperties,1);
         
         for k = 1:NumProperties
            ListData{k} = cell(ElementCount(i),1);
         end
         
         % list type
		   for j = 1:ElementCount(i)
   	      for k = 1:NumProperties
      	      if ~Type(k)
         	      Data(j,k) = Buf(BufOff);
            	   BufOff = BufOff + 1;
	            else
   	            tmp = Buf(BufOff);
      	         ListData{k}{j} = Buf(BufOff+(1:tmp))';
         	      BufOff = BufOff + tmp + 1;
            	end
            end
         end
      end
   else		%%% read binary data %%%
      % translate PLY data type names to MATLAB data type names
      ListFlag = 0;		% = 1 if there is a list type 
      SameFlag = 1;     % = 1 if all types are the same
      
      for j = 1:NumProperties
         Token = getfield(CurPropertyTypes,CurPropertyNames{j});
         
         if ~strcmp(Token{1},'list')			% non-list type
	         tmp = rem(strmatch(Token{1},PlyTypeNames,'exact')-1,8)+1;
         
            if ~isempty(tmp)
               TypeSize(j) = SizeOf(tmp);
               Type{j} = MatlabTypeNames{tmp};
               TypeSize2(j) = 0;
               Type2{j} = '';
               
               SameFlag = SameFlag & strcmp(Type{1},Type{j});
	         else
   	         fclose(fid);
               error(['Unknown property data type, ''',Token{1},''', in ', ...
                     ElementNames{i},'.',CurPropertyNames{j},'.']);
         	end
         else											% list type
            if length(Token) == 3
               ListFlag = 1;
               SameFlag = 0;
               tmp = rem(strmatch(Token{2},PlyTypeNames,'exact')-1,8)+1;
               tmp2 = rem(strmatch(Token{3},PlyTypeNames,'exact')-1,8)+1;
         
               if ~isempty(tmp) & ~isempty(tmp2)
                  TypeSize(j) = SizeOf(tmp);
                  Type{j} = MatlabTypeNames{tmp};
                  TypeSize2(j) = SizeOf(tmp2);
                  Type2{j} = MatlabTypeNames{tmp2};
	   	      else
   	   	      fclose(fid);
               	error(['Unknown property data type, ''list ',Token{2},' ',Token{3},''', in ', ...
                        ElementNames{i},'.',CurPropertyNames{j},'.']);
               end
            else
               fclose(fid);
               error(['Invalid list syntax in ',ElementNames{i},'.',CurPropertyNames{j},'.']);
            end
         end
      end
      
      % read file
      if ~ListFlag
         if SameFlag
            % no list types, all the same type (fast)
            Data = fread(fid,[NumProperties,ElementCount(i)],Type{1})';
         else
            % no list types, mixed type
            Data = zeros(ElementCount(i),NumProperties);
            
         	for j = 1:ElementCount(i)
        			for k = 1:NumProperties
               	Data(j,k) = fread(fid,1,Type{k});
              	end
         	end
         end
      else
         ListData = cell(NumProperties,1);
         
         for k = 1:NumProperties
            ListData{k} = cell(ElementCount(i),1);
         end
         
         if NumProperties == 1
            BufSize = 512;
            SkipNum = 4;
            j = 0;
            
            % list type, one property (fast if lists are usually the same length)
            while j < ElementCount(i)
               Position = ftell(fid);
               % read in BufSize count values, assuming all counts = SkipNum
               [Buf,BufSize] = fread(fid,BufSize,Type{1},SkipNum*TypeSize2(1));
               Miss = find(Buf ~= SkipNum);					% find first count that is not SkipNum
               fseek(fid,Position + TypeSize(1),-1); 		% seek back to after first count                              
               
               if isempty(Miss)									% all counts are SkipNum
                  Buf = fread(fid,[SkipNum,BufSize],[int2str(SkipNum),'*',Type2{1}],TypeSize(1))';
                  fseek(fid,-TypeSize(1),0); 				% undo last skip
                  
                  for k = 1:BufSize
                     ListData{1}{j+k} = Buf(k,:);
                  end
                  
                  j = j + BufSize;
                  BufSize = floor(1.5*BufSize);
               else
                  if Miss(1) > 1									% some counts are SkipNum
                     Buf2 = fread(fid,[SkipNum,Miss(1)-1],[int2str(SkipNum),'*',Type2{1}],TypeSize(1))';                     
                     
                     for k = 1:Miss(1)-1
                        ListData{1}{j+k} = Buf2(k,:);
                     end
                     
                     j = j + k;
                  end
                  
                  % read in the list with the missed count
                  SkipNum = Buf(Miss(1));
                  j = j + 1;
                  ListData{1}{j} = fread(fid,[1,SkipNum],Type2{1});
                  BufSize = ceil(0.6*BufSize);
               end
            end
         else
            % list type(s), multiple properties (slow)
            Data = zeros(ElementCount(i),NumProperties);
            
            for j = 1:ElementCount(i)
         		for k = 1:NumProperties
            		if isempty(Type2{k})
               		Data(j,k) = fread(fid,1,Type{k});
            		else
               		tmp = fread(fid,1,Type{k});
               		ListData{k}{j} = fread(fid,[1,tmp],Type2{k});
		            end
      		   end
      		end
         end
      end
   end
   
   % put data into Elements structure
   for k = 1:NumProperties
   	if (~Format & ~Type(k)) | (Format & isempty(Type2{k}))
      	eval(['Elements.',ElementNames{i},'.',CurPropertyNames{k},'=Data(:,k);']);
      else
      	eval(['Elements.',ElementNames{i},'.',CurPropertyNames{k},'=ListData{k};']);
		end
   end
end

clear Data ListData;
fclose(fid);

if (nargin > 1 & strcmpi(Str,'Tri')) | nargout > 2   
   % find vertex element field
   Name = {'vertex','Vertex','point','Point','pts','Pts'};
   Names = [];
   
   for i = 1:length(Name)
      if any(strcmp(ElementNames,Name{i}))
         Names = getfield(PropertyNames,Name{i});
         Name = Name{i};         
         break;
      end
   end
   
   if any(strcmp(Names,'x')) & any(strcmp(Names,'y')) & any(strcmp(Names,'z'))
      eval(['varargout{1}=[Elements.',Name,'.x,Elements.',Name,'.y,Elements.',Name,'.z];']);
   else
      varargout{1} = zeros(1,3);
	end
           
   varargout{2} = Elements;
   varargout{3} = Comments;
   Elements = [];
   
   % find face element field
   Name = {'face','Face','poly','Poly','tri','Tri'};
   Names = [];
   
   for i = 1:length(Name)
      if any(strcmp(ElementNames,Name{i}))
         Names = getfield(PropertyNames,Name{i});
         Name = Name{i};
         break;
      end
   end
   
   if ~isempty(Names)
      % find vertex indices property subfield
	   PropertyName = {'vertex_indices','vertex_indexes','vertex_index','indices','indexes'};           
      
   	for i = 1:length(PropertyName)
      	if any(strcmp(Names,PropertyName{i}))
         	PropertyName = PropertyName{i};
	         break;
   	   end
      end
      
      if ~iscell(PropertyName)
         % convert face index lists to triangular connectivity
         eval(['FaceIndices=varargout{2}.',Name,'.',PropertyName,';']);
  			N = length(FaceIndices);
   		Elements = zeros(N*2,3);
   		Extra = 0;   

			for k = 1:N
   			Elements(k,:) = FaceIndices{k}(1:3);
   
   			for j = 4:length(FaceIndices{k})
      			Extra = Extra + 1;      
	      		Elements(N + Extra,:) = [Elements(k,[1,j-1]),FaceIndices{k}(j)];
   			end
         end
         Elements = Elements(1:N+Extra,:) + 1;
      end
   end
else
   varargout{1} = Comments;
end
end


function [addf,f,v] = meshRepair(f,v)

 [list,bdyEdges,f,v] = getListOfHoles(f,v);
 cnd_hole = ~isempty(list);
 addf=[]; 
 while(cnd_hole==1)
    vA_hatL = cell(1,max(list));
    for i = 1:max(list)
        holeEdges = bdyEdges(list==i,:);
        d = 2; % Polynomial degree
        
        srtd_i = holeEdges;
        indx_r = 1;
        for j = 1:size(holeEdges,1)-1
            temp = [holeEdges(1:indx_r-1,:);[NaN NaN];holeEdges(indx_r+1:end,:)];
            [indx_r,indx_c] = find(temp==srtd_i(j,2));
            srtd_i(j+1,:) = holeEdges(indx_r,:);
            if (indx_c == 2)
                srtd_i(j+1,:) = flip(srtd_i(j+1,:));
            end
        end
        if (srtd_i(end,1)==srtd_i(1,1))
            srtd_i(end,:) = flip(srtd_i(end,:));
        end
        vA_hatL{1,i} = zeros(size(srtd_i,1),3);
        vA_hatL{1,i}(:,1:2) = srtd_i ;
        for j = 1:size(srtd_i,1)
            f_i = f(sum(ismember(f,srtd_i(j,:)),2)==2,:);
            vA_hatL{1,i}(j,3) = f_i(1,~ismember(f_i,srtd_i(j,:)));
        end
        
        %[fitPoly,p] = fitPolyToHole(f,v,holeEdges,d);
    end

    % Filling the holes:
%     for z=1:size(vA_hatL,2)
%         A(z,1)=size(vA_hatL{1,z},1);
%     end
    [~,addf,v,f]=holeFilling_Fit3D(f,vA_hatL,v,holeEdges,d);
   
   [list,bdyEdges,f,v] = getListOfHoles(f,v);
   cnd_hole = ~isempty(list);
 end
 

end
function [l_e_created,vA_hat,AvV,f]=holeFilling_Fit3D(f,vA_hat,AvV,holeEdges,d)

%figure;

all_v_e=[];

Size_AvV=[size(AvV,1)];
num_of_newf = zeros(size(vA_hat,2)+1,1);
fitPoly=[];
for q = 1:size(vA_hat,2)
    
   %%% add check if hole bigger than 4 edges 
    clear fitPoly;
    vA_hat_q = vA_hat{1,q};
    [fitPoly,p] = fitPolyToHole(f,AvV,vA_hat_q(:,1:2),d);
Bdry_info=zeros(size(vA_hat_q,1),6,4);
Bdry_info(:,1:3,1) = vA_hat_q;
for k=1:3
    Bdry_info(:,k,2:end)=AvV(vA_hat_q(:,k),:);
end 
v1 = permute(Bdry_info(:,1,2:4),[1,3,2]);
v2 = permute(Bdry_info(:,2,2:4),[1,3,2]);
epsilon = mean(sqrt(sum((v2'-v1').^ 2))/2);
%epsilon = prctile(sqrt(sum((v2-v1).^ 2,2)),10)/10;

first_edges = sort(Bdry_info(:,[1,2],1),2);

 %%
num_e=1;
while(size(Bdry_info,1)>=num_e)
    e1=permute(Bdry_info(num_e,2,2:end)-Bdry_info(num_e,1,2:end),[1,3,2]);
    ver1=permute(Bdry_info(num_e,1,2:end),[1,3,2]);
    ver2=permute(Bdry_info(num_e,2,2:end),[1,3,2]);
   
    if (num_e == size(Bdry_info,1))
        e2 = permute(Bdry_info(1,2,2:end)-Bdry_info(1,1,2:end),[1,3,2]);
        ver3=permute(Bdry_info(1,2,2:end),[1,3,2]);
    else
        e2 = permute(Bdry_info(num_e+1,2,2:end)-Bdry_info(num_e+1,1,2:end),[1,3,2]);
        ver3=permute(Bdry_info(num_e+1,2,2:end),[1,3,2]);
    end
%     CosBeta = dot(e1,e2)/(norm(e1)*norm(e2));
%     Beta = acosd(CosBeta);
    Beta=rad2deg(anglePoints3d(ver1,ver2,ver3));
    sz_BI = size(Bdry_info,1);
    if(Beta<20)
        % Merge
        if (num_e == sz_BI)
            Bdry_info([1,num_e],:,:) = [];
        else
            Bdry_info(num_e:num_e+1,:,:) = [];
        end
        if (size(Bdry_info,1)==3)
            f = [f; [Bdry_info(1,1,1), Bdry_info(1,2,1), Bdry_info(2,2,1)]];   
            Bdry_info = [];
            break;
        elseif(size(Bdry_info,1)==4)
            f = [f; [Bdry_info(1,1,1), Bdry_info(1,2,1), Bdry_info(2,2,1)];[Bdry_info(1,1,1), Bdry_info(2,2,1), Bdry_info(3,2,1)]];   
            Bdry_info = [];
            break;
        elseif(size(Bdry_info,1)==1) %%%%
            Bdry_info=[];
            break
        end
        if ~isempty(Bdry_info) %%%
            if ((num_e == 1) || (num_e == sz_BI) || (num_e == sz_BI-1))                        
                f(f== Bdry_info(end,2,1)) = Bdry_info(1,1,1);
                AvV(Bdry_info(end,2,1),:) = NaN;
    %             if(Bdry_info(end,2,1)==327769)
    %                 sima=0;
    %             end
                for qq = q+1 : size(vA_hat,2)
                    vA_hat{1,qq}(vA_hat{1,qq}== Bdry_info(end,2,1)) = Bdry_info(1,1,1);
                end
                Bdry_info(end,2,:) = Bdry_info(1,1,:);
                num_e = num_e - 1;
            else
    %            try 
                f(f==Bdry_info(num_e-1,2,1)) = Bdry_info(num_e,1,1);
    %             catch
    %                 md=0;
    %             end 
                AvV(Bdry_info(num_e-1,2,1),:) = NaN;
    %             if(Bdry_info(num_e-1,2,1)==327769)
    %                 sima=0;vc
    %             end
                for qq = q+1 : size(vA_hat,2)
                    vA_hat{1,qq}(vA_hat{1,qq}== Bdry_info(num_e-1,2,1)) = Bdry_info(num_e,1,1);
                end
                Bdry_info(num_e-1,2,:) = Bdry_info(num_e,1,:);
                num_e = num_e - 1;
            end 
        end
    elseif(Beta<90)
        if (num_e == size(Bdry_info,1))
            %dist = sqrt(sum((Bdry_info(num_e,1,:).^2 - Bdry_info(1,2,:)).^2))/3;
            dist= distancePoints(Bdry_info(num_e,1,:),Bdry_info(1,2,:));
        else
            %dist = sqrt(sum((Bdry_info(num_e,1,:).^2 - Bdry_info(num_e+1,2,:)).^2))/3;
            dist= distancePoints(Bdry_info(num_e,1,:),Bdry_info(num_e+1,2,:));
        end
        if(abs(dist)<7*epsilon)
            % Connect to create a new face
            Bdry_info(num_e,3,:) = Bdry_info(num_e,2,:); %%% num_e+1 changed 
            if (num_e == size(Bdry_info,1))
                f = [f; [Bdry_info(num_e,1,1), Bdry_info(num_e,2,1), Bdry_info(1,2,1)]];   
                Bdry_info(num_e,2,:) = Bdry_info(1,2,:);            
                Bdry_info(1,:,:) = [];
            else
                f = [f; [Bdry_info(num_e,1,1), Bdry_info(num_e,2,1), Bdry_info(num_e+1,2,1)]];   
                Bdry_info(num_e,2,:) = Bdry_info(num_e+1,2,:);
                Bdry_info(num_e+1,:,:) = [];
            end
            if (size(Bdry_info,1)==3)
                f = [f; [Bdry_info(1,1,1), Bdry_info(1,2,1), Bdry_info(2,2,1)]];   
                Bdry_info = [];
                break;
            elseif(size(Bdry_info,1)==4)
                f = [f; [Bdry_info(1,1,1), Bdry_info(1,2,1), Bdry_info(2,2,1)];[Bdry_info(1,1,1), Bdry_info(2,2,1), Bdry_info(3,2,1)]];   
                Bdry_info = [];
                break;
            end
            num_e = num_e - 1;
        end
    end
    num_e = num_e + 1;

end
    
if(~isempty(Bdry_info))
    vA_hat_q = Bdry_info(:,1:3,1);

    v1v2=permute(Bdry_info(:,2,2:end)-Bdry_info(:,1,2:end),[1,3,2]);
    vA_hat_q = sort(vA_hat_q,2);

    Normv1v2=sqrt((v1v2(:,1)).^2+(v1v2(:,2)).^2+(v1v2(:,3)).^2);
     epsilon=mean(Normv1v2(:,1))/2;
%     epsilon = epsilon ./ 2;
   %epsilon = prctile(Normv1v2,10)/2;

    Bdry_info_tmp=Bdry_info;


%% Finding the distances between the second triangle and the rest of the initial vertices


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% REMBEMER TO CHANGE THIS:
    mx_vIdx = size(AvV,1);
%tmp_size1 = 1;
%tmp_size2 = size(vA_hat_q,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     hold on 
end

while(size(Bdry_info,1)>3)
    i = 2; %Bdry_info_tmp
    j = 2; %Bdry_info
    cnd = 1;
    
    
    while(cnd==1) 

        
           [Bdry_info,Bdry_info_tmp,j,i]=FillHoles_Fit3D(AvV,f,Bdry_info_tmp,Bdry_info,j,i,fitPoly,p,epsilon); % updating Bdry_info indexes

           i = i + 1;
           j = j + 1;

% Last free edge of B_I : Connect the last edge with the first "new face"
           if j==size(Bdry_info,1)
                Bdry_info(end,5,:)=zeros(1,1,4);    
                    tmp = Bdry_info(j-1:j,:,:);

                    tmp(1,1,:)=Bdry_info(j-1,1,:);
                    tmp(1,2,:)=Bdry_info_tmp(i-1,1,:); 
                    tmp(1,3,:)=Bdry_info(j,1,:);
                    tmp(1,5,:)=Inf; 

                    tmp(2,1,:)=Bdry_info_tmp(i-1,1,:); 
                    tmp(2,2,:)=Bdry_info(1,5,:);
                    tmp(2,3,:)=Bdry_info(j,1,:);
                    tmp(2,5,:)=Inf;

                    tmp(3,1,:)=Bdry_info(1,5,:);  
                    tmp(3,2,:)=Bdry_info(1,2,:);
                    tmp(3,3,:)=Bdry_info(1,1,:);
                    tmp(3,5,:)=Inf;     

                    tmp(4,1,:)=Bdry_info(j,1,:);  
                    tmp(4,2,:)=Bdry_info(j,2,:);
                    tmp(4,3,:)=Bdry_info(1,5,:);
                    tmp(4,3,2:4)=Bdry_info(1,5,2:4);
%                     tmp(4,3,1)=size(AvV,1)+1;
                    tmp(4,5,:)=Inf; 
                                        
                    if(tmp(2,1,:) == tmp(2,2,:)) 
                        Bdry_info_tmp = [Bdry_info_tmp(2:end-2,:,:);tmp([1,3],:,:)];                         
                    else 
                        Bdry_info_tmp = [Bdry_info_tmp(2:end-2,:,:);tmp(1:3,:,:)]; 
                    end
                cnd=0;
           else
               cnd=1;
           end
           
    end
    
    

    Bdry_info=Bdry_info_tmp;
    Bdry_info(:,4:5,:)=0;
    Bdry_info(:,6,1)=0;
  
            for j1 = 1:size(Bdry_info,1) 
                for j2 = 1:3
                    if (Bdry_info(j1,j2,1)==0)
                        sum_j3 = zeros(size(AvV,1),1);
                        for j3 = 1:3
                            sum_j3 = sum_j3 + ...
                                (permute(Bdry_info(j1,j2,j3+1),[2,3,1])==AvV(:,j3)); 
                        end
                        rw_num1=find(sum_j3==3);
                        if (isempty(rw_num1))
                            mx_vIdx = mx_vIdx + 1;
                            Bdry_info(j1,j2,1) = mx_vIdx;
                            AvV = [AvV ; permute(Bdry_info(j1,j2,2:4),[1,3,2])];
                        else
                            Bdry_info(j1,j2,1) = rw_num1;
                        end
                    end
                end
            end
        
            [~,idx_first]=ismember(permute(tmp(4,3,2:4),[2,3,1]),AvV,'rows');
            tmp(4,3,1)=idx_first;
            vA_hat_q=[vA_hat_q;sort(tmp(4,1:3,1),2)];

            Bdry_info_tmp2 = sort(Bdry_info(:,1:3,1),2); %indx sorted
            %tmp_size1 = size(vA_hat_q,1);
            vA_hat_q = [vA_hat_q;Bdry_info_tmp2];
%             % TRY
%             vA_hat_q=sort(vA_hat_q,2);
            %%
            vA_hat_q = unique(vA_hat_q,'rows');
            
%             if vA_hat_q(1,1)==0
%                 vA_hat_q(1,1)=max(max(vA_hat_q));
%             end
            
            %tmp_size2 = size(vA_hat_q,1);

            j5=2;
            cntr6=0;
            while j5<size(Bdry_info,1)
                cntr6=cntr6+1;
                if  Bdry_info(j5,1,1)~=Bdry_info(j5-1,2,1)
                    Bdry_info=[Bdry_info(1:j5-1,:,:);Bdry_info(j5+1:end,:,:)];
                    j5=j5-1;
                end
                j5=j5+1;
            end 
            
            
            j3 = 3;
            cntr2 = 0;
            while( j3<=size(Bdry_info,1))
                cntr2=cntr2+1;
                if Bdry_info(j3,2,:)==Bdry_info(j3-2,1,:)
                   vA_hat_q=[vA_hat_q;sort([Bdry_info(j3-2,1:2,1),Bdry_info(j3-1,2,1)],2)];
                    if j3==size(Bdry_info,1)
                        Bdry_info = Bdry_info(1:j3-3,:,:); 
                        j3=size(Bdry_info,1);
                    else 
                        if j3==size(Bdry_info,1)-1
                           Bdry_info = Bdry_info(1:j3-3,:,:); 
                           j3=size(Bdry_info,1);
                        else
                            Bdry_info = [Bdry_info(1:j3-3,:,:);Bdry_info(j3+1:end,:,:)];
                            j3=j3-1;
                        end
                    end 
                end 
                j3=j3+1;
            end  
            
            if (size(Bdry_info,1)>2)
                [~,Indx_end_Bdry_info]=ismember(Bdry_info(2,1,1),Bdry_info(2:end,2,1));
                if (Indx_end_Bdry_info==0)
                    [~,Indx_end_Bdry_info]=ismember(Bdry_info(1,1,1),Bdry_info(:,2,1));
                    Bdry_info=Bdry_info(1:Indx_end_Bdry_info,:,:);
                else
                    Bdry_info=Bdry_info(2:Indx_end_Bdry_info+1,:,:);
                end
                
            else 
                Bdry_info = [];
            end                        

    if(~isempty(vA_hat_q))
        f = [f;vA_hat_q];
        [~,idx,~] = unique(sort(f,2),'rows'); % preserving original face orientation
        f = f(idx,:);
    end
    
end

if size(Bdry_info,1)==3
    vA_hat_q=[vA_hat_q;Bdry_info(1,1,1),Bdry_info(2,1,1),Bdry_info(3,1,1)];
    f = [f;[Bdry_info(1,1,1),Bdry_info(2,1,1),Bdry_info(3,1,1)]];
    [~,idx,~] = unique(sort(f,2),'rows'); % preserving original face orientation
    f = f(idx,:);    
end

vA_hat{1,q} = vA_hat_q;

num_of_newf(q+1) = size(vA_hat_q,1);

st_f = sort(vA_hat_q,2); 
v_e = unique([st_f(:,[1,2]);st_f(:,[2,3]);st_f(:,[1,3])],'rows');
select_edges = ~ismember(v_e,first_edges,'rows');
all_v_e = [all_v_e ; v_e(select_edges,:)];

end

tmp = zeros(sum(num_of_newf),3);

for i = 1:size(vA_hat,2)
    tmp(sum(num_of_newf(1:i))+1:sum(num_of_newf(1:i+1)),:)=vA_hat{1,i};
end
vA_hat = tmp;

idx_nan_V=[1;find(isnan(AvV(:,1)));length(AvV)];
red_f=0:length(idx_nan_V);
for kk=1:length(idx_nan_V)-1
    f((idx_nan_V(kk))<=f & f<=idx_nan_V(kk+1))=f(idx_nan_V(kk)<=f & f<=idx_nan_V(kk+1))-red_f(kk);    vA_hat((idx_nan_V(kk))<=vA_hat & vA_hat<=idx_nan_V(kk+1))=vA_hat(idx_nan_V(kk)<=vA_hat & vA_hat<=idx_nan_V(kk+1))-red_f(kk);
    first_edges((idx_nan_V(kk))<=first_edges & first_edges<=idx_nan_V(kk+1))=first_edges(idx_nan_V(kk)<=first_edges & first_edges<=idx_nan_V(kk+1))-red_f(kk);
    all_v_e((idx_nan_V(kk))<=all_v_e & all_v_e<=idx_nan_V(kk+1))=all_v_e(idx_nan_V(kk)<=all_v_e & all_v_e<=idx_nan_V(kk+1))-red_f(kk);
    vA_hat((idx_nan_V(kk))<=vA_hat & vA_hat<=idx_nan_V(kk+1))=vA_hat(idx_nan_V(kk)<=vA_hat & vA_hat<=idx_nan_V(kk+1))-red_f(kk);
end 
AvV(isnan(AvV(:,1)),:) = [];

l_e_created = sqrt((AvV(all_v_e(:,1),1)- AvV(all_v_e(:,2),1)).^2 + (AvV(all_v_e(:,1),2)- AvV(all_v_e(:,2),2)).^2 + (AvV(all_v_e(:,1),3)- AvV(all_v_e(:,2),3)).^2);


%% tmp to delete later
f_e=sort([f(:,[1,2]);f(:,[2,3]);f(:,[1,3])],2);
AAA=ismember(f_e,all_v_e);
B=sum(AAA,2);
XX=find(B,2);
for lll=1:size(XX)
    [u,v]=ismember(all_v_e,f_e(XX(lll),:));
    [w,y]=ismember(3,sum(v,2));
    if y~=0
        l_e_created(y,:)=[];
    end
end

end
function [list,bdyEdges,f,v] = getListOfHoles(f,v)
% [list,bdyEdges,f,v] = getListOfHoles(f,v) groups the boundary edges into
% a list. Note that some vertices and faces are removed in this function.
% So f and v need to be updated.
%
% Example:
%   [list,bdyEdges,f,v] = getListOfHoles(f,v);
%   bdyEdges(list==1) % boundary edges of first hole
%   bdyEdges(list==2) % boundary edges of second hole

    % Get the boundary edges
    bdyEdges = getBoundaryEdges(f);

    % Every vertex in bdyEdges should occur twice. Remove the faces where
    % it does not.
    [u,~,n] = unique(bdyEdges(:));
    counts = accumarray(n,1);
    
    badVertices = u(counts~=2); % vertices that did not occur twice
    
    if ~isempty(u(counts>2))
%         [~,rangeF] = getFaces(f,find(bdyEdges(:)));
%         figure;
%         plotAva(f(rangeF,:),v,'r')
%         stop = 1;
    end
    
    
    while ~isempty(badVertices)
        faceToRemove = [];
        [~,badRow] = ismember(badVertices,bdyEdges);
        badRow = mod(badRow-1,length(bdyEdges))+1; % row in bdyEdges that contains badVertices
        for i = 1:length(badRow)
            try
            e = bdyEdges(badRow(i),:);
            catch
                marcelline = 1;
            end
            badFaceIdx = find((sum(f == e(1),2) + sum(f == e(2),2)) == 2);
%             fprintf('Edge %d %d\n',e(1),e(2))
%             fprintf('Face %d %d %d\n',f(badFaceIdx,1),f(badFaceIdx,2),f(badFaceIdx,3))
                         
            faceToRemove = [faceToRemove badFaceIdx]; %%% unique added

        end
        
        [f,v] = removeFace(f,v,faceToRemove); % Remove the face
        
        bdyEdges = getBoundaryEdges(f); % get new boundary edges
        [u,~,n] = unique(bdyEdges(:));
        counts = accumarray(n,1);
        badVertices = u(counts~=2); % vertices that did not occur twice
    end
    
    try
        % I am not sure what this section does but an error occurs when
        % mod_loc_rep contains a zero. 
        sz_f = size(f,1);
        E = sort([f(:,[1,2]); f(:,[2,3]); f(:,[1,3])],2);
        loc_rep = find(ismember(E,bdyEdges,'rows'));
        mod_loc_rep = mod(loc_rep,sz_f);
        counts = accumarray(mod_loc_rep, 1);
        isolated_f = find(counts==3);
        if (~isempty(isolated_f))
            1/0;
            f(isolated_f,:) = [];
            E = sort([f(:,[1,2]); f(:,[2,3]); f(:,[1,3])],2);
            [u,~,n] = unique(E,'rows');
            counts = accumarray(n(:), 1);
            bdyEdges = u(counts==1,:);
        end
    catch ME
        warning('An error occurred in getListOfHoles with message: %s', ME.message)
    end
    % attachedHole_f = find(counts==2);
    % if (~isempty(attachedHole_f))
    %     f(attachedHole_f,:) = [];
    % end
    % if (~isempty(isolated_f) || ~isempty(attachedHole_f))
    %     f(remove_f,:) = [];
    %     E = sort([f(:,[1,2]); f(:,[2,3]); f(:,[1,3])],2);
    %     [u,~,n] = unique(E,'rows');
    %     counts = accumarray(n(:), 1);
    %     bdyEdges = u(counts==1,:);
    % end

    % Group boundary edges together
    list = zeros(1,length(bdyEdges));
    holeNum = 1;
    row = 1;
    col = 1;
    if (~isempty(bdyEdges))
        list(row) = holeNum;
        start = row;
        while true
            col = mod(col,2)+1; % get other col
            % Find next edge
            [a,b] = find(bdyEdges == bdyEdges(row,col));
            if sum([a(1),b(1)] == [row,col]) == 2
                row = a(2);
                col = b(2);
            else
                row = a(1);
                col = b(1);
            end
            % Update list
            if row ~= start
                list(row) = holeNum;
            else
                % if row == start, continue with next hole
                holeNum = holeNum + 1;
                col = 1;
                row = find(list==0, 1); % next row not assigned to a hole
                if ~isempty(row)
                    list(row) = holeNum;
                    start = row;
                else
                    break; % Break if no edge is left
                end
            end
        end
    end
end
function fitPoly = fitPolynomial(v,d,p)
% [a,E] = fitPolynomial(v,order) fit a polynomial to the given points.
% Input:
%   v - the points
%   d - the degree of the polynomial
%   p - the permutation

    v = v(:,p);
    normalizeData(1) = mean(v(:,1));
    normalizeData(2) = mean(v(:,2));
    normalizeData(3) = std(v(:,1));
    normalizeData(4) = std(v(:,2));
    v(:,1) = (v(:,1)-normalizeData(1))./normalizeData(3);
    v(:,2) = (v(:,2)-normalizeData(2))./normalizeData(4);
    
    [s,~] = size(v);
    E = ones(s,1);
    for i = 1:d
        for j = 0:i
            E = [E (v(:,2).^j).*(v(:,1).^(i-j))];
        end
    end
    a = E\v(:,3); % coefficiants of polynomial
    fitPoly.a = a;
    fitPoly.E = E;
    fitPoly.normalizeData = normalizeData;
    fitPoly.d = d;
end
function [fitPoly,p] = fitPolyToHole(f,v,bdyEdges,d)
% [fitPoly,p] = fitPolyToHole(f,v,bdyEdges,d) fits a polynomial to the hole
%
% Input:
%   f - faces
%   v - vertices
%   bdyEdges - boundary edges defining the hole
%   d - degree
%
% Output:
%   fitPoly - fitted polynomial
%   p - permutation

    % Get vertices around the hole
    u = unique(bdyEdges(:));
    range = max([max(v(:,1)),max(v(:,2)),max(v(:,3))])*0.005;
    rangeV = zeros(length(v),1);
    for i = 1:length(u)
        newRangeV = v(:,1)+range > v(u(i),1) & v(u(i),1) > v(:,1)-range &...
                    v(:,2)+range > v(u(i),2) & v(u(i),2) > v(:,2)-range &...
                    v(:,3)+range > v(u(i),3) & v(u(i),3) > v(:,3)-range;
        rangeV = rangeV | newRangeV;
    end
    
    rangeV = unique(find(rangeV));
    
    % Find the permutation to use
    dist = zeros(3,1);
    dist(1) = max(v(rangeV,1))-min(v(rangeV,1));
    dist(2) = max(v(rangeV,2))-min(v(rangeV,2));
    dist(3) = max(v(rangeV,3))-min(v(rangeV,3));
    if min(dist) == dist(1)
        p = [2,3,1];
    elseif min(dist) == dist(2)
        p = [3,1,2];
    else
        p = [1,2,3];
    end
           
    fitPoly = fitPolynomial(v(rangeV,:),d,p);
    
    % --------------------
    % Plotting faces around hole
    % --------------------
%     rangeF = getFaces(f,rangeV);
%     figure;
%     plotAva(rangeF,v,'r');
%     hold on
%     
%     % --------------------
%     % Plotting polynomial
%     % --------------------
%     X = linspace(min(v(rangeV,p(1))),max(v(rangeV,p(1))),50);
%     Y = linspace(min(v(rangeV,p(2))),max(v(rangeV,p(2))),50);
%     [X,Y] = meshgrid(X,Y);
%     Z = evalPolynomial(X,Y,fitPoly);
%     if isequal(p,[1,2,3])
%         surf(X,Y,Z,'facealpha',0.5);
%     elseif isequal(p,[2,3,1])
%         surf(Z,X,Y,'facealpha',0.5);
%     else
%         surf(Y,Z,X,'facealpha',0.5);
%     end
end
function alpha = anglePoints3d(varargin)
%ANGLEPOINTS3D Compute angle between three 3D points
%
%   ALPHA = anglePoints3d(P1, P2)
%   Computes angle (P1, O, P2), in radians, between 0 and PI.
%
%   ALPHA = anglePoints3d(P1, P2, P3)
%   Computes angle (P1, P2, P3), in radians, between 0 and PI.
%
%   ALPHA = anglePoints3d(PTS)
%   PTS is a 3x3 or 2x3 array containing coordinate of points.
%
%   See also
%   points3d, angles3d
%
%   ---------
%   author : David Legland 
%   INRA - TPV URPOI - BIA IMASTE
%   created the 21/02/2005.
%

%   HISTORY
%   20/09/2005: add case of single argument for all points
%   04/01/2007: check typo
%   27/05/2014: adjust known vector sizes n1, n0, n2 once corrected for


p2 = [0 0 0];
if length(varargin) == 1
    pts = varargin{1};
    if size(pts, 1)==2
        p1 = pts(1,:);
        p0 = [0 0 0];
        p2 = pts(2,:);
    else
        p1 = pts(1,:);
        p0 = pts(2,:);
        p2 = pts(3,:);
    end
    
elseif length(varargin) == 2
    p1 = varargin{1};
    p0 = [0 0 0];
    p2 = varargin{2};
    
elseif length(varargin) == 3
    p1 = varargin{1};
    p0 = varargin{2};
    p2 = varargin{3};
end

% ensure all data have same size
n1 = size(p1, 1);
n2 = size(p2, 1);
n0 = size(p0, 1);

if n1 ~= n0
    if n1 == 1
        p1 = repmat(p1, [n0 1]);
        n1 = n0;
    elseif n0==1
        p0 = repmat(p0, [n1 1]);
    else
        error('Arguments P1 and P0 must have the same size');
    end
end

if n1 ~= n2
    if n1 == 1
        p1 = repmat(p1, [n2 1]);
    elseif n2 == 1
        p2 = repmat(p2, [n1 1]);
    else
        error('Arguments P1 and P2 must have the same size');
    end
end

% normalized vectors
p1 = normalizeVector3d(p1 - p0);
p2 = normalizeVector3d(p2 - p0);

% compute angle
alpha = acos(dot(p1, p2, 2));
end
function vn = normalizeVector3d(v)
%NORMALIZEVECTOR3D Normalize a 3D vector to have norm equal to 1
%
%   V2 = normalizeVector3d(V);
%   Returns the normalization of vector V, such that ||V|| = 1. Vector V is
%   given as a row vector.
%
%   If V is a N-by-3 array, normalization is performed for each row of the
%   input array.
%
%   See also:
%   vectors3d, vectorNorm3d
%
%   ---------
%   author : David Legland 
%   INRA - TPV URPOI - BIA IMASTE
%   created the 29/11/2004.
%

% HISTORY
% 2005-11-30 correct a bug
% 2009-06-19 rename as normalizeVector3d
% 2010-11-16 use bsxfun (Thanks to Sven Holcombe)

vn   = bsxfun(@rdivide, v, sqrt(sum(v.^2, 2)));
end
function c = vectorCross3d(a,b)
%VECTORCROSS3D Vector cross product faster than inbuilt MATLAB cross.
%
%   C = vectorCross3d(A, B) 
%   returns the cross product of the 3D vectors A and B, that is: 
%       C = A x B
%   A and B must be N-by-3 element vectors. If either A or B is a 1-by-3
%   row vector, the result C will have the size of the other input and will
%   be the  concatenation of each row's cross product. 
%
%   Example
%     v1 = [2 0 0];
%     v2 = [0 3 0];
%     vectorCross3d(v1, v2)
%     ans =
%         0   0   6
%
%
%   Class support for inputs A,B:
%      float: double, single
%
%   See also DOT.

%   Sven Holcombe

% needed_colons = max([3, length(size(a)), length(size(b))]) - 3;
% tmp_colon = {':'};
% clnSet = tmp_colon(ones(1, needed_colons));
% 
% c = bsxfun(@times, a(:,[2 3 1],clnSet{:}), b(:,[3 1 2],clnSet{:})) - ...
%     bsxfun(@times, b(:,[2 3 1],clnSet{:}), a(:,[3 1 2],clnSet{:}));

sza = size(a);
szb = size(b);

% Initialise c to the size of a or b, whichever has more dimensions. If
% they have the same dimensions, initialise to the larger of the two
switch sign(numel(sza) - numel(szb))
    case 1
        c = zeros(sza);
    case -1
        c = zeros(szb);
    otherwise
        c = zeros(max(sza, szb));
end

c(:) =  bsxfun(@times, a(:,[2 3 1],:), b(:,[3 1 2],:)) - ...
        bsxfun(@times, b(:,[2 3 1],:), a(:,[3 1 2],:));
end
function plane2 = normalizePlane(plane1)
%NORMALIZEPLANE Normalize parametric representation of a plane
%
%   PLANE2 = normalizePlane(PLANE1);
%   Transforms the plane PLANE1 in the following format:
%   [X0 Y0 Z0  DX1 DY1 DZ1  DX2 DY2 DZ2], where:
%   - (X0, Y0, Z0) is a point belonging to the plane
%   - (DX1, DY1, DZ1) is a first direction vector
%   - (DX2, DY2, DZ2) is a second direction vector
%   into another plane, with the same format, but with:
%   - (x0 y0 z0) is the closest point of plane to the origin
%   - (DX1 DY1 DZ1) has norm equal to 1
%   - (DX2 DY2 DZ2) has norm equal to 1 and is orthogonal to (DX1 DY1 DZ1)
%   
%   See also:
%   planes3d, createPlane
%
%   ---------
%   author: David Legland 
%   INRA - TPV URPOI - BIA IMASTE
%   created the 21/02/2005.
%

%   HISTORY
%   21/08/2009 compute origin after computation of vectors (more precise)
%       and add support for several planes.

% compute first direction vector
d1  = normalizeVector3d(plane1(:,4:6));

% compute second direction vector
n   = normalizeVector3d(planeNormal(plane1));
d2  = -normalizeVector3d(vectorCross3d(d1, n));

% compute origin point of the plane
origins = repmat([0 0 0], [size(plane1, 1) 1]);
p0 = projPointOnPlane(origins, [plane1(:,1:3) d1 d2]);

% create the resulting plane
plane2 = [p0 d1 d2];
end
function point = projPointOnPlane(point, plane)
%PROJPOINTONPLANE Return the orthogonal projection of a point on a plane
%
%   PT2 = projPointOnPlane(PT1, PLANE);
%   Compute the (orthogonal) projection of point PT1 onto the plane PLANE,
%   given as [X0 Y0 Z0  VX1 VY1 VZ1  VX2 VY2 VZ2] (origin point, first
%   direction vector, second directionvector).
%   
%   The function is fully vectorized, in that multiple points may be
%   projected onto multiple planes in a single call, returning multiple
%   points. With the exception of the second dimension (where
%   SIZE(PT1,2)==3, and SIZE(PLANE,2)==9), each dimension of PT1 and PLANE
%   must either be equal or one, similar to the requirements of BSXFUN. In
%   basic usage, point PT1 is a [N*3] array, and PLANE is a [N*9] array
%   (see createPlane for details). Result PT2 is a [N*3] array, containing
%   coordinates of orthogonal projections of PT1 onto planes PLANE. In
%   vectorised usage, PT1 is an [N*3*M*P...] matrix, and PLANE is an
%   [X*9*Y...] matrix, where (N,X), (M,Y), etc, are either equal pairs, or
%   one of the two is one.
%
%   See also:
%   planes3d, points3d, planePosition, intersectLinePlane

%   ---------
%   author : David Legland 
%   INRA - TPV URPOI - BIA IMASTE
%   created the 18/02/2005.
%

%   HISTORY
%   21/08/2006: debug support for multiple points or planes
%   22/04/2013: uses bsxfun for mult. pts/planes in all dimensions (Sven H)

% Unpack the planes into origins and normals, keeping original shape
plSize = size(plane);
plSize(2) = 3;
[origins, normals] = deal(zeros(plSize));
origins(:) = plane(:,1:3,:);
normals(:) = vectorCross3d(plane(:,4:6,:), plane(:, 7:9,:));

% difference between origins of plane and point
dp = bsxfun(@minus, origins, point);

% relative position of point on normal's line
t = bsxfun(@rdivide, sum(bsxfun(@times,normals,dp),2), sum(normals.^2,2));

% add relative difference to project point back to plane
point = bsxfun(@plus, point, bsxfun(@times, t, normals));
end
function dist = distancePoints(p1, p2, varargin)
%DISTANCEPOINTS Compute distance between two points
%
%   D = distancePoints(P1, P2)
%   Return the Euclidean distance between points P1 and P2.
%
%   If P1 and P2 are two arrays of points, result is a N1-by-N2 array
%   containing distance between each point of P1 and each point of P2. 
%
%   D = distancePoints(P1, P2, NORM)
%   Compute distance using the specified norm. NORM=2 corresponds to usual
%   euclidean distance, NORM=1 corresponds to Manhattan distance, NORM=inf
%   is assumed to correspond to maximum difference in coordinate. Other
%   values (>0) can be specified.
%
%   D = distancePoints(..., 'diag')
%   compute only distances between P1(i,:) and P2(i,:).
%
%   See also:
%   points2d, minDistancePoints, nndist, hausdorffDistance
%

% ------
% Author: David Legland
% e-mail: david.legland@nantes.inra.fr
% Copyright 2009 INRA - Cepia Software Platform.
% created the 24/02/2004.
%

%   HISTORY :
%   25/05/2004: manage 2 array of points
%   07/04/2004: add option for computing only diagonal.
%   30/10/2006: generalize to any dimension, and manage different norms
%   03/01/2007: bug for arbitrary norm, and update doc
%   28/08/2007: fix bug for norms 2 and infinite, in diagonal case


%% Setup options

% default values
diag = false;
norm = 2;

% check first argument: norm or diag
if ~isempty(varargin)
    var = varargin{1};
    if isnumeric(var)
        norm = var;
    elseif strncmp('diag', var, 4)
        diag = true;
    end
    varargin(1) = [];
end

% check last argument: diag
if ~isempty(varargin)
    var = varargin{1};
    if strncmp('diag', var, 4)
        diag = true;
    end
end


% number of points in each array and their dimension
n1  = size(p1, 1);
n2  = size(p2, 1);
d   = size(p1, 2);

if diag
    % compute distance only for apparied couples of pixels
    dist = zeros(n1, 1);
    
    if norm == 2
        % Compute euclidian distance. this is the default case
        % Compute difference of coordinate for each pair of point
        % and for each dimension. -> dist is a [n1*n2] array.
        for i = 1:d
            dist = dist + (p2(:,i)-p1(:,i)).^2;
        end
        dist = sqrt(dist);
        
    elseif norm == inf
        % infinite norm corresponds to maximal difference of coordinate
        for i = 1:d
            dist = max(dist, abs(p2(:,i)-p1(:,i)));
        end
        
    else
        % compute distance using the specified norm.
        for i = 1:d
            dist = dist + power((abs(p2(:,i)-p1(:,i))), norm);
        end
        dist = power(dist, 1/norm);
    end
else
    % compute distance for all couples of pixels
    dist = zeros(n1, n2);
    
    if norm == 2
        % Compute euclidian distance. This is the default case.
        % Compute difference of coordinate for each pair of point
        % and for each dimension. -> dist is a [n1*n2] array.
        for i = 1:d
            % equivalent to:
            % dist = dist + ...
            %   (repmat(p1(:,i), [1 n2])-repmat(p2(:,i)', [n1 1])).^2;
            dist = dist + bsxfun (@minus, p1(:,i), p2(:, i)').^2;
        end
        dist = sqrt(dist);
        
    elseif norm == inf
        % infinite norm corresponds to maximal difference of coordinate
        for i = 1:d
            dist = max(dist, abs(bsxfun (@minus, p1(:,i), p2(:, i)')));
        end
        
    else
        % compute distance using the specified norm.
        for i = 1:d
            % equivalent to:
            % dist = dist + power((abs(repmat(p1(:,i), [1 n2]) - ...
            %     repmat(p2(:,i)', [n1 1]))), norm);
            dist = dist + power(abs(bsxfun(@minus, p1(:,i), p2(:, i)')), norm);
        end
        dist = power(dist, 1/norm);
    end
end

end
function [Bdry_info,Bdry_info_tmp,j,i]=FillHoles_Fit3D(v,f,Bdry_info_tmp,Bdry_info,j,i,fitPoly,p,epsilon)



    if Bdry_info(1,6,1)==0
        Bdry_info(:,6,1)=Inf;
        holeEdges=Bdry_info(:,1:2,1);
        Bdry_info(:,5,2:4)=getThirdV(f,v,holeEdges,fitPoly,p);
        Bdry_info_tmp=Bdry_info;
    end

%                  Variables for the distance
                    VA1=permute(Bdry_info(j,5,2:end),[1,3,2]);
                    VA2=permute(Bdry_info(j-1,5,2:end),[1,3,2]);
                    V11=permute(Bdry_info(j,5,2:end),[1,3,2]);
                    NormvAvA=abs(distancePoints(VA1,VA2));
                    NormV1VA=abs(distancePoints(V11,VA1));

    %                 Variables for alpha
                    P3=permute(Bdry_info(j-1,5,2:end),[1,3,2]);
                    P2=permute(Bdry_info(j,1,2:end),[1,3,2]);
                    P5=permute(Bdry_info(j,5,2:end),[1,3,2]);
                    alpha=anglePoints3d (P3,P2,P5); % angle in radians

    %                 Variables for the Overlap
                    P1=permute(Bdry_info(j-1,1,2:end),[1,3,2]);
                    P4=permute(Bdry_info(j,2,2:end),[1,3,2]);
                    Plane1=createPlane(P1, P2, P3);
                    Plane2=createPlane(P3, P2, P5); 
                    Plane3=createPlane(P5, P2, P4);
                    N_Plane1=planeNormal(Plane1);
                    N_Plane2=planeNormal(Plane2);
                    N_Plane3=planeNormal(Plane3);

                    CosBeta1 = dot(N_Plane1,N_Plane2)/(norm(N_Plane1)*norm(N_Plane2));
                    Beta1 = acosd(CosBeta1);

                    CosBeta2 = dot(N_Plane2,N_Plane3)/(norm(N_Plane2)*norm(N_Plane3));
                    Beta2 = acosd(CosBeta2);

    %%                       
                        if (NormvAvA < 1.5*epsilon && NormV1VA < epsilon)
%                         if mn_val < 1.2*epsilon
%                             if (mn_indx==3) %C1
                                tmp = Bdry_info(j-1:j,:,:);
                                tmp(1,2,:)=Bdry_info(j-1,5,:); 
                                tmp(1,3,:)=Bdry_info(j-1,2,:);
                                tmp(1,5,:)=Inf;                             
                                tmp(2,1,:)=Bdry_info(j-1,5,:);   
                                tmp(2,3,:)=Bdry_info(j-1,2,:);
                                tmp(2,5,:)=Inf;
%                                 if i~=size(Bdry_info,1)
                                Bdry_info_tmp = [Bdry_info_tmp(1:i-2,:,:);tmp;Bdry_info_tmp(i+1:end,:,:)]; 
                                Bdry_info(j,5,:) = tmp(2,1,:);
                        else
                            if alpha<(0.7*pi) 
                                cnd_Beta1 = ((Beta1==0 || Beta1==180) &&  (Beta2==0 || Beta2==180)) && (alpha<0);
                                cnd_Beta2 = ((Beta1>90 && Beta1<180) && (Beta2>90 && Beta2<180));
                                if (cnd_Beta1 || cnd_Beta2) % C3
                                    tmp = Bdry_info(j-1:j,:,:);
                                    tmp(2,1,:)=Bdry_info(j-1,5,:);   
                                    tmp(2,3,:)=Bdry_info(j-1,2,:);
                                    tmp(2,5,:)=Inf;
                                    tmp(1,2,:)=Bdry_info(j-1,5,:); 
                                    tmp(1,3,:)=Bdry_info(j-1,2,:);
                                    tmp(1,5,:)=Inf;
                                        Bdry_info_tmp = [Bdry_info_tmp(1:i-2,:,:);tmp;Bdry_info_tmp(i+1:end,:,:)];
                                        Bdry_info(j,5,:) = tmp(2,1,:);

                                else %C4
                                    tmp=repmat(Bdry_info(j,:,:),[3,1,1]);
                                    tmp(3,1,:)=Bdry_info(j,5,:);  
                                    tmp(3,3,:)=Bdry_info(j,1,:);
                                    tmp(3,5,:)=Inf;
                                    tmp(2,1,:)=Bdry_info(j-1,5,:);
                                    tmp(2,2,:)=Bdry_info(j,5,:); 
                                    tmp(2,3,:)=Bdry_info(j,1,:);
                                    tmp(2,5,:)=Inf;
                                    tmp(1,1,:)=Bdry_info(j-1,1,:);
                                    tmp(1,2,:)=Bdry_info(j-1,5,:); 
                                    tmp(1,3,:)=Bdry_info(j,1,:);
                                    tmp(1,5,:)=Inf;
                                     Bdry_info_tmp =[Bdry_info_tmp(1:i-2,:,:);tmp;Bdry_info_tmp(i+1:end,:,:)]; 
                                    i=i+1;
                                end
                            else  % C5
                                    tmp=repmat(Bdry_info(j,:,:),[4,1,1]);
                                    tmp(4,1,:)=Bdry_info(j,5,:);  
                                    tmp(4,3,:)=Bdry_info(j,1,:);
                                    tmp(4,5,:)=Inf;
                                    tmp(3,1,:)=Bdry_info(j,1,:);
                                    tmp(3,2,:)=Bdry_info(j,5,:);  
                                    tmp(3,3,:)=Bdry_info(j,2,:);
                                    tmp(3,5,:)=Inf;
%                                     
                                    Bdry_info_tmp = [Bdry_info_tmp(1:i-1,:,:);tmp(3:4,:,:);Bdry_info_tmp(i+1:end,:,:)]; 
                                    i=i+1;
                            end
                        end  

end
function thirdV = getThirdV(f,v,holeEdges,fitPoly,p)
% thirdV = getThirdV(f,v,holeEdges,a) returns the third vertex to make a
% face with the holeEdges. These vertices are on the polynomial defined by
% the coefficients in a.
% 
% Input:
%   f - all faces
%   v - all vertices
%   holeEdges - n by 2 array containing the edges of the hole, where n is
%               the number of edges.
%   fitPoly - the polynomial computed with fitPolynomial
%   p - the permutation of the axes

    % Permutate
    v = v(:,p);
    
    % Epsilon determines how big the new faces will be. It is computing
    % the average length of the boundary edges and the multiplying by
    % 0.75. Not sure how it is done before. Feel free to change.
    v1 = v(holeEdges(:,1),:);
    v2 = v(holeEdges(:,2),:);
 % epsilon = mean(sqrt(sum((v2'-v1').^ 2)))/3.5; %% Fit3D
    epsilon = mean(sqrt(sum((v2'-v1').^ 2)))/6; 

    %epsilon = prctile(sqrt(sum((v2-v1).^ 2,2)),5);
    %epsilon = median(sqrt(sum((v2'-v1').^ 2)))/80;
    
    len = length(holeEdges);
    thirdV = zeros(len,3); % This will store the third vertex
    eMid = zeros(len,3); % Midpoint of boudnary edges
    n = zeros(len,3);
    k = zeros(len,3); % Normal to edge on triangle plane
    
    for j = 1:len
        e = holeEdges(j,:);
        [~,face1] = getFaces(f,e(1));
        [~,face2] = getFaces(f,e(2));
        face = f(face1 & face2,:); % This is the face that has the edge
        [~,idx,~] = unique(sort(face,2),'rows'); % preserving original face orientation
        face=face(idx,:);
        face=face(1,:);
        v1 = v(e(1),:);
        v2 = v(e(2),:);
        v3 = v(face(~(face == e(1) | face == e(2))),:);
        
         try
        n0 = cross(v2-v1,v3-v1); % Normal to triangle
         catch 
             sima=0;
         end
        n(j,:) = n0/norm(n0);
        k0 = cross(v2-v1,n(j,:)); % Normal to edge on triangle plane
        k(j,:) = k0/norm(k0); % Unit length
        eMid(j,:) = (v1+v2)/2; % Mid point of boundary edge
    end
    
    
    thirdV = eMid + epsilon*k;
    % Now plug in thirdV into closestPointOnPoly the get the closest point
    % on the polynomial
    for j = 1:len
        try
            thirdV(j,:) = closestPointOnPoly(fitPoly,thirdV(j,:));
        catch
            stop =1;
        end
    end
    
%     % Compare normals and merge thrid vertices if needed
%     gamma = 0.8;
%     crossOfNormal = normAll(cross(n(:,:),n([2:len,1],:)));
%     for j = 1:len
%         if crossOfNormal(j) > gamma
%             j1 = j;
%             j2 = mod(j,len)+1;
%             v1 = thirdV(j1,:);
%             v2 = thirdV(j2,:);
%             ave = (v1+v2)/2;
%             thirdV(j1,:) = ave;
%             thirdV(j2,:) = ave;
%         end
%     end
    
    % Reverse the orientation
    oReverse(p) = 1:length(p);
    thirdV = thirdV(:,oReverse);
    
    % --------------------
    % Plotting new vertices
    % --------------------
%    figure;
%     for j = 1:len        
%         plot3(thirdV(j,1),thirdV(j,2),thirdV(j,3),'b*')
%     end
end
function x = closestPointOnPoly(fitPoly,p)
% x = closestPointOnPoly(fitPoly,p)
% Gives the point closest to p on the polynomial fitPoly
    
    fun = @(x) gradOfDistanceOfPointToPoly(fitPoly,p,x);
    x0 = [p(1),p(2)];
    options = optimset('Display','off');
    try
    x = fsolve(fun,x0,options);
    catch
        sima=0;
    end 
    x(3) = evalPolynomial(x(1),x(2),fitPoly);
end
function D = gradOfDistanceOfPointToPoly(fitPoly,p,x)
% D = gradOfDistanceOfPointToPoly(fitPoly,p,x)
% Gives the gradient of the distance funtion of a point to a polynomial.
%
% Input:
%   fitPoly     - polynomial from the function fitPolyToHole
%   p           - the point
%   x=(x_1,x_2) - variable of distance function

    % Normalize data
    x(1) = (x(1)-fitPoly.normalizeData(1))./fitPoly.normalizeData(3);
    x(2) = (x(2)-fitPoly.normalizeData(2))./fitPoly.normalizeData(4);
    p(1) = (p(1)-fitPoly.normalizeData(1))./fitPoly.normalizeData(3);
    p(2) = (p(2)-fitPoly.normalizeData(2))./fitPoly.normalizeData(4);
    
    % Polynomial
    k = 1;
    P = fitPoly.a(k);
    for i = 1:fitPoly.d
        for j = 0:i
            k = k+1;
            P = P + fitPoly.a(k)*x(2).^j*x(1).^(i-j);
        end
    end
    
    % Derivative of P with respect to x
    k = 1;
    Px = 0;
    for i = 1:fitPoly.d
        for j = 0:i
            k = k+1;
            if i-j-1>=0
                Px = Px + fitPoly.a(k)*x(2).^j * x(1).^(i-j-1)*(i-j);
            end
        end
    end
    
    % Derivative of P with respect to y
    k = 1;
    Py = 0;
    for i = 1:fitPoly.d
        for j = 0:i
            k = k+1;
            if j-1>=0
                Py = Py + fitPoly.a(k)*x(2).^(j-1)*j * x(1).^(i-j);
            end
        end
    end
    
    % Gradient of distance function
    D = zeros(1,2);
    D(1) = 2*(x(1)-p(1))*fitPoly.normalizeData(3)^2 + 2*(P-p(3))*Px;
    D(2) = 2*(x(2)-p(2))*fitPoly.normalizeData(4)^2 + 2*(P-p(3))*Py;
end
function [faces,idx] = getFaces(f,vIndex)
% Gets all the faces that have all 3 vertices in vIndex
%   faces = getFaces(self,self.lArmIdx) gets all the faces of the
%   left arm
    if islogical(vIndex)
        vIndex = find(vIndex);
    end
    idx = ismember(f, vIndex);
    idx = idx(:,1) | idx(:,2) | idx(:,3);
    faces = f(idx,:);
end
function z = evalPolynomial(x,y,fitPoly)
% z = evalPolynomial(x,y,fitPoly) evaluates the polynomial at (x,y) with the
% given coefficients in a.
% fitpoly can be computed with fitPolynomial

    x = (x-fitPoly.normalizeData(1))./fitPoly.normalizeData(3);
    y = (y-fitPoly.normalizeData(2))./fitPoly.normalizeData(4);
   
    % Set up matrix
    [s1,s2] = size(x);
    E = ones(s1*s2,1);
    for i = 1:fitPoly.d
        for j = 0:i
            E = [E (y(:).^j).*(x(:).^(i-j))];
        end
    end
    % Evaluate
    z = E*fitPoly.a;
    z = reshape(z,size(x));
end
function n = planeNormal(plane)
%PLANENORMAL Compute the normal to a plane
%
%   N = planeNormal(PLANE) 
%   compute the normal of the given plane
%   PLANE : [x0 y0 z0 dx1 dy1 dz1 dx2 dy2 dz2]
%   N : [dx dy dz]
%   
%   See also
%   geom3d, planes3d, createPlane
%
%   ---------
%   author : David Legland 
%   INRA - TPV URPOI - BIA IMASTE
%   created the 17/02/2005.
%

%   HISTORY
%   15/04/2013 Extended to N-dim planes by Sven Holcombe

% plane normal
outSz = size(plane);
outSz(2) = 3;
n = zeros(outSz);
n(:) = vectorCross3d(plane(:,4:6,:), plane(:, 7:9,:));
end
function plane = createPlane(varargin)
%CREATEPLANE Create a plane in parametrized form
%
%   PLANE = createPlane(P1, P2, P3) 
%   creates a plane containing the 3 points
%
%   PLANE = createPlane(PTS) 
%   The 3 points are packed into a single 3x3 array.
%
%   PLANE = createPlane(P0, N);
%   Creates a plane from a point and from a normal to the plane. The
%   parameter N is given either as a 3D vector (1-by-3 row vector), or as
%   [THETA PHI], where THETA is the colatitute (angle with the vertical
%   axis) and PHI is angle with Ox axis, counted counter-clockwise (both
%   given in radians).
%   
%   The created plane data has the following format:
%   PLANE = [X0 Y0 Z0  DX1 DY1 DZ1  DX2 DY2 DZ2], with
%   - (X0, Y0, Z0) is a point belonging to the plane
%   - (DX1, DY1, DZ1) is a first direction vector
%   - (DX2, DY2, DZ2) is a second direction vector
%   The 2 direction vectors are normalized and orthogonal.
%
%   See also:
%   planes3d, medianPlane
%   
%   ---------
%   author: David Legland 
%   INRA - TPV URPOI - BIA IMASTE
%   created the 18/02/2005.
%

%   HISTORY
%   24/11/2005 add possibility to pack points for plane creation
%   21/08/2006 return normalized planes
%   06/11/2006 update doc for planes created from normal

if length(varargin) == 1
    var = varargin{1};
    
    if iscell(var)
        plane = zeros([length(var) 9]);
        for i=1:length(var)
            plane(i,:) = createPlane(var{i});
        end
    elseif size(var, 1) >= 3
        % 3 points in a single array
        p1 = var(1,:);
        p2 = var(2,:);
        p3 = var(3,:);
        
        % create direction vectors
        v1 = p2 - p1;
        v2 = p3 - p1;

        % create plane
        plane = normalizePlane([p1 v1 v2]);
        return;
    end
    
elseif length(varargin) == 2
    % plane origin
    p0 = varargin{1};
    
    % second parameter is either a 3D vector or a 3D angle (2 params)
    var = varargin{2};
    if size(var, 2) == 2
        % normal is given in spherical coordinates
        n = sph2cart2([var ones(size(var, 1))]);
    elseif size(var, 2)==3
        % normal is given by a 3D vector
        n = normalizeVector3d(var);
    else
        error ('wrong number of parameters in createPlane');
    end
    
    % ensure same dimension for parameters
    if size(p0, 1)==1
        p0 = repmat(p0, [size(n, 1) 1]);
    end
    if size(n, 1)==1
        n = repmat(n, [size(p0, 1) 1]);
    end

    % find a vector not colinear to the normal
    v0 = repmat([1 0 0], [size(p0, 1) 1]);
    inds = vectorNorm3d(cross(n, v0, 2))<1e-14;
    v0(inds, :) = repmat([0 1 0], [sum(inds) 1]);
%     if abs(cross(n, v0, 2))<1e-14
%         v0 = repmat([0 1 0], [size(p0, 1) 1]);
%     end
    
    % create direction vectors
    v1 = normalizeVector3d(cross(n, v0, 2));
    v2 = -normalizeVector3d(cross(v1, n, 2));

    % concatenate result in the array representing the plane
    plane = [p0 v1 v2];
    return;
    
elseif length(varargin)==3
    p1 = varargin{1};    
    p2 = varargin{2};
    p3 = varargin{3};
    
    % create direction vectors
    v1 = p2 - p1;
    v2 = p3 - p1;
   
    plane = normalizePlane([p1 v1 v2]);
    return;
  
else
    error('Wrong number of arguments in "createPlane".');
end

end 
function [v,f] = CleaningMesh(v,f)

    f = double(f);
    vNum = length(v);
    fNum = length(f);
    %% getting rid of repeated faces:
    [~,idx,~] = unique(sort(f,2),'rows'); % preserving original face orientation
    f = f(idx,:);
    vNumRepeated = vNum - length(v); vNum = length(v);
    fNumRepeated = fNum - length(f); fNum = length(f);
    fprintf('Omitting reapeated faces: f-%d, v-%d\n',fNumRepeated,vNumRepeated);
    
    %% getting rid of bad-shaped faces:
    [f,v] = omitBadShapedFaces(f,v);
    vNumBadShape = vNum - length(v); vNum = length(v);
    fNumBadShape = fNum - length(f); fNum = length(f);
    fprintf('Omitting bad shaped faces: f-%d, v-%d\n',fNumBadShape,vNumBadShape);
    %% getting rid of faces sticking out:
    [f,v] = uselessFaceOmission(f,v);
    vNumUseless = vNum - length(v); vNum = length(v);
    fNumUseless = fNum - length(f); fNum = length(f);
    fprintf('Omitting bad shaped faces: f-%d, v-%d\n',fNumUseless,vNumUseless);

    %% Make Large faces smaller
    [f,v] = divideLargeFaces(f,v);
     %% getting rid of repeated faces:
    [~,idx,~] = unique(sort(f,2),'rows'); % preserving original face orientation
    f = f(idx,:);
    vNumRepeated = vNum - length(v); %vNum = length(v);
    fNumRepeated = fNum - length(f); %fNum = length(f);
    fprintf('Omitting reapeated faces: f-%d, v-%d\n',fNumRepeated,vNumRepeated);    
end
function [newf,newv] = omitBadShapedFaces(f,v)
ext_f = zeros(size(f,1),3,3);
for i = 1:3
    ext_f(:,:,i) = v(f(:,i),:);
end
len_edge = f;
len_edge(:,1) = sqrt(sum((ext_f(:,:,2)-ext_f(:,:,1)).^2,2));
len_edge(:,2) = sqrt(sum((ext_f(:,:,3)-ext_f(:,:,2)).^2,2));
len_edge(:,3) = sqrt(sum((ext_f(:,:,3)-ext_f(:,:,1)).^2,2));
clear ext_f;
ave_len_edge = sum(sum(len_edge)) ./ numel(len_edge);
if (length(v)>30000)
    thr_div = round(length(v) ./ 100);
else
    thr_div = round(length(v) ./ 5000);
end
thr = ave_len_edge ./ thr_div;
indx = len_edge > thr;
sum_indx = sum(indx,2);
tiny_f = f((sum_indx==0),:);
newv = v;
newf = f;
% getting rid of tiny faces (faces with three small edges):
while(~isempty(tiny_f))
    tiny_f = sort(tiny_f,2);
    [~,idx_1] = sort(tiny_f(:,1));
    tiny_f = tiny_f(idx_1,:);   
    for i = 1:size(tiny_f,1)
        newf(newf==tiny_f(i,2))=tiny_f(i,1);
        newf(newf==tiny_f(i,3))=tiny_f(i,1);
        newv(tiny_f(i,2),:) = [nan nan nan];
        newv(tiny_f(i,3),:) = [nan nan nan];
    end
    nlen_edge = newf;
    nlen_edge(:,1) = newf(:,2)-newf(:,1);
    nlen_edge(:,2) = newf(:,3)-newf(:,2);
    nlen_edge(:,3) = newf(:,3)-newf(:,1);
    newf = newf((prod(nlen_edge,2)~=0),:);
    ext_f = zeros(size(newf,1),3,3);
    for i = 1:3
        ext_f(:,:,i) = newv(newf(:,i),:);
    end
    len_edge = newf;
    len_edge(:,1) = sqrt(sum((ext_f(:,:,2)-ext_f(:,:,1)).^2,2));
    len_edge(:,2) = sqrt(sum((ext_f(:,:,3)-ext_f(:,:,2)).^2,2));
    len_edge(:,3) = sqrt(sum((ext_f(:,:,3)-ext_f(:,:,1)).^2,2));
    clear ext_f;
    indx = len_edge > thr;
    sum_indx = sum(indx,2);
    tiny_f = newf((sum_indx==0),:);
end
% getting rid of tiny wide-angled faces (faces with two small edges):
TWA_f = newf((sum_indx==1),:);
while(~isempty(TWA_f))
    id_TWAf = indx((sum_indx==1),:);
    for i = 1:size(TWA_f,1)
        if (prod(id_TWAf(i,:) == [0,1,0])~=0)
            newf(newf==TWA_f(i,1))=TWA_f(i,2);
            newv(TWA_f(i,1),:) = [nan nan nan];
        elseif (prod(id_TWAf(i,:) == [0,0,1])~=0)
            newf(newf==TWA_f(i,2))=TWA_f(i,3);
            newv(TWA_f(i,2),:) = [nan nan nan];
        elseif (prod(id_TWAf(i,:) == [1,0,0])~=0)
            newf(newf==TWA_f(i,3))=TWA_f(i,1);
            newv(TWA_f(i,3),:) = [nan nan nan];
        end
    end
    nlen_edge = newf;
    nlen_edge(:,1) = newf(:,2)-newf(:,1);
    nlen_edge(:,2) = newf(:,3)-newf(:,2);
    nlen_edge(:,3) = newf(:,3)-newf(:,1);
    newf = newf((prod(nlen_edge,2)~=0),:);
    ext_f = zeros(size(newf,1),3,3);
    for i = 1:3
        ext_f(:,:,i) = newv(newf(:,i),:);
    end
    len_edge = newf;
    len_edge(:,1) = sqrt(sum((ext_f(:,:,2)-ext_f(:,:,1)).^2,2));
    len_edge(:,2) = sqrt(sum((ext_f(:,:,3)-ext_f(:,:,2)).^2,2));
    len_edge(:,3) = sqrt(sum((ext_f(:,:,3)-ext_f(:,:,1)).^2,2));
    clear ext_f;
    indx = len_edge > thr;
    sum_indx = sum(indx,2);
    TWA_f = newf((sum_indx==1),:);
end
% getting rid of narrow faces (faces with one small edge):
narrow_f = newf((sum_indx==2),:);
while(~isempty(narrow_f))
    id_narrowf = indx((sum_indx==2),:);
    bad_v = zeros(size(narrow_f,1),2);
    for i = 1:size(narrow_f,1)
        if (id_narrowf(i,1) == 0)
            bad_v(i,:) = [narrow_f(i,1),narrow_f(i,2)];
        elseif (id_narrowf(i,2) == 0)
            bad_v(i,:) = [narrow_f(i,2),narrow_f(i,3)];
        elseif (id_narrowf(i,3) == 0)
            bad_v(i,:) = [narrow_f(i,1),narrow_f(i,3)];
        end
    end
    bad_v = unique(sort(bad_v,2),'rows');
    for i = 1:size(bad_v,1)
        newf(newf==bad_v(i,1)) = bad_v(i,2);
        newv(bad_v(i,1),:) = [nan nan nan];
    end
    nlen_edge = newf;
    nlen_edge(:,1) = newf(:,2)-newf(:,1);
    nlen_edge(:,2) = newf(:,3)-newf(:,2);
    nlen_edge(:,3) = newf(:,3)-newf(:,1);
    newf = newf((prod(nlen_edge,2)~=0),:);
    ext_f = zeros(size(newf,1),3,3);
    for i = 1:3
        ext_f(:,:,i) = newv(newf(:,i),:);
    end
    len_edge = newf;
    len_edge(:,1) = sqrt(sum((ext_f(:,:,2)-ext_f(:,:,1)).^2,2));
    len_edge(:,2) = sqrt(sum((ext_f(:,:,3)-ext_f(:,:,2)).^2,2));
    len_edge(:,3) = sqrt(sum((ext_f(:,:,3)-ext_f(:,:,1)).^2,2));
    clear ext_f;
    indx = len_edge > thr;
    sum_indx = sum(indx,2);
    narrow_f = newf((sum_indx==2),:);
end
omitted_v = isnan(newv(:,1));
newv = newv(~omitted_v,:);
omitted_v = find(omitted_v);
for i = 1:size(omitted_v,1)
    newf(newf>omitted_v(i)) = newf(newf>omitted_v(i)) - 1;
    omitted_v = omitted_v - 1;
end
end
function [f,v] = uselessFaceOmission(f,v)
edges = [f(:,[1,2]); f(:,[2,3]); f(:,[1,3])];
edges = sort(edges,2);
% [unique_edges,~,rep_indx_e] = unique(edges,'rows');
[~,~,rep_indx_e] = unique(edges,'rows');
rep_indx = accumarray(rep_indx_e,1);
rep_indx = find(rep_indx>2);
%rep_indx = find(rep_indx==1);

while (~isempty(rep_indx))
    sz_f = size(f,1);
    for i = 1:size(rep_indx,1)
%         loc_rep = find(ismember(edges,unique_edges(rep_indx(i),:),'rows'));
        loc_rep = find(rep_indx_e == rep_indx(i));
%         tmp = edges(loc_rep,:)
        mod_loc_rep = mod(loc_rep,sz_f);
        rep_faces = f(mod_loc_rep,:);
        idx = ~isnan(rep_faces(:,1));
        rep_faces = rep_faces(idx,:);
        loc_rep = loc_rep(idx,:);
        mod_loc_rep = mod(loc_rep,sz_f);
        rep_v = floor(loc_rep/sz_f);
        indic = zeros(size(rep_v));
        cntr = 0;
        for j = 1:length(rep_v)
            switch rep_v(j)
                case 0
                    indic(j) = sum(sum(f==rep_faces(j,3)));
                    if(indic(j)==1)
                        f(mod_loc_rep(j),:) = [nan nan nan];
                        v(rep_faces(j,3),:) = [nan nan nan];
                        cntr = cntr + 1;
                    end
                case 1
                    indic(j) = sum(sum(f==rep_faces(j,1)));
                    if(indic(j)==1)
                        f(mod_loc_rep(j),:)=[nan nan nan];
                        v(rep_faces(j,1),:) = [nan nan nan];
                        cntr = cntr + 1;
                    end
                case 2
                    indic(j) = sum(sum(f==rep_faces(j,2)));
                    if (indic(j)==1)
                        f(mod_loc_rep(j),:) = [nan nan nan];
                        v(rep_faces(j,2),:) = [nan nan nan];
                        cntr = cntr + 1;
                    end
            end
        end
        j = length(rep_v);
        if (cntr == 0)
            while(j>2)
                [~,id_mn] = min(indic);
                f(mod_loc_rep(id_mn),:) = [nan nan nan];
                if (rep_v(id_mn)==0)
                    try
                    v(rep_faces(id_mn,3),:) = [nan nan nan];
                    catch
                        sima = 0;
                    end
                    f(mod(find(f==rep_faces(id_mn,3)),sz_f),:) = nan;
                else
                    v(rep_faces(id_mn,rep_v(id_mn)),:) = [nan nan nan];
                    f(mod(find(f==rep_faces(id_mn,rep_v(id_mn))),sz_f),:) = nan;
                end
                indic(id_mn) = [];
                rep_v(id_mn) = [];
                j=j-1;
            end
        end
    end
    f = f(~isnan(f(:,1)),:);
    omitted_v = isnan(v(:,1));
    v = v(~omitted_v,:);
    omitted_v = find(omitted_v);
    for i = 1:size(omitted_v,1)
        f(f>omitted_v(i)) = f(f>omitted_v(i)) - 1;
        omitted_v = omitted_v - 1;
    end
    edges = [f(:,[1,2]); f(:,[2,3]); f(:,[1,3])];
    edges = sort(edges,2);
    [~,~,rep_indx_e] = unique(edges,'rows');
%     [unique_edges,~,rep_indx_e] = unique(edges,'rows');
    rep_indx = accumarray(rep_indx_e,1);
    rep_indx = find(rep_indx>2);
end
end
function [f,v] = divideLargeFaces(f,v)
    % All edges not repeated
    E = unique(sort([f(:,[1,2]); f(:,[2,3]); f(:,[1,3])],2),'rows');
    v1 = v(E(:,1),:);
    v2 = v(E(:,2),:);
    % Compute the lengths
    edgeLength = normAll(v1-v2);
    aveLength = mean(edgeLength);
    largeEdges = find(edgeLength>aveLength*10)';
    while ~isempty(largeEdges)
        fprintf('# of large edges: %d\n',length(largeEdges))
        for i = largeEdges
            e1 = E(i,1);
            e2 = E(i,2);
            [~,face1] = getFaces(f,e1);
            [~,face2] = getFaces(f,e2);
            faces = find(face1 & face2); % The faces with that edge
            midV = (v(e1,:)+v(e2,:))/2;
            v = [v;midV];
            newVIdx = length(v);
            for faceIdx = faces'
                face = f(faceIdx,:);
                newFace = face;
                newFace(newFace == e1) = newVIdx;
                f = [f;newFace];
                newFace = face;
                newFace(newFace == e2) = newVIdx;
                f = [f;newFace];
            end
            f(faces,:) = [];
        end
        % Recalculate edge lengths
        E = unique(sort([f(:,[1,2]); f(:,[2,3]); f(:,[1,3])],2),'rows');
        v1 = v(E(:,1),:);
        v2 = v(E(:,2),:);
        edgeLength = normAll(v1-v2);
        largeEdges = find(edgeLength>aveLength*10)';
    end
end
function [f,v] = deleteFaceIntersections(f,v)
% [f,v] = deleteFaceIntersections(f,v) deletes faces that intersect with
% one another.

    epsilon = 10^-10;
    remove = zeros(1,length(f));
    v0 = v(f(:,1),:);
    v1 = v(f(:,2),:);
    v2 = v(f(:,3),:);
    N = cross(v1-v0,v2-v0,2); % normal of each triangle/plane
    d = -dot(N,v0,2);

    range = max([max(v(:,1)),max(v(:,2)),max(v(:,3))])*0.025;
    
    % Center of all faces
    c1 = mean([v0(:,1),v1(:,1),v2(:,1)],2);
    c2 = mean([v0(:,2),v1(:,2),v2(:,2)],2);
    c3 = mean([v0(:,3),v1(:,3),v2(:,3)],2);
    c = [c1 c2 c3];
    for i2 = 1:length(f)
        rangeV = v(:,1)+range > c(i2,1) & c(i2,1) > v(:,1)-range &...
                 v(:,2)+range > c(i2,2) & c(i2,2) > v(:,2)-range &...
                 v(:,3)+range > c(i2,3) & c(i2,3) > v(:,3)-range;
        [~,rangeF] = getFaces(f,find(rangeV));
        rangeF(i2) = 0; % Don't check with itself
        
        rangeF = find(rangeF)';
        len = length(rangeF);
        % Here we check the sideness of the vertices of the triangle
        % with respect to the plane of the other triangle.
        % For example, if dv10>0, dv11<0, and dv12<0, T1 intersects
        % with Plane2.
        % If the value is equal to zero, then the vertex is on the
        % plane.
        
        % Sideness of vertices of T1 with respect to Plane2
        dv10 = dot(repmat(N(i2,:),len,1),v0(rangeF,:),2) + repmat(d(i2),len,1);
        dv11 = dot(repmat(N(i2,:),len,1),v1(rangeF,:),2) + repmat(d(i2),len,1);
        dv12 = dot(repmat(N(i2,:),len,1),v2(rangeF,:),2) + repmat(d(i2),len,1);
        % Sideness of vertices of T2 with respect to Plane1
        dv20 = dot(N(rangeF,:),repmat(v0(i2,:),len,1),2) + d(rangeF);
        dv21 = dot(N(rangeF,:),repmat(v1(i2,:),len,1),2) + d(rangeF);
        dv22 = dot(N(rangeF,:),repmat(v2(i2,:),len,1),2) + d(rangeF);
        
        planeintersect = (dv10.*dv11<0 | dv11.*dv12<0 | dv10.*dv12< 0) & ...
                         (dv20.*dv21<0 | dv21.*dv22<0 | dv20.*dv22< 0);
        planesidentical = abs(dv10) < epsilon & abs(dv11) < epsilon & abs(dv12) < epsilon;
        edgeonplane = (dv10 == 0 & dv11 == 0) | ...
                   (dv11 == 0 & dv12 == 0) | ...
                   (dv12 == 0 & dv10 == 0) | ...
                   (dv20 == 0 & dv21 == 0) | ...
                   (dv21 == 0 & dv22 == 0) | ...
                   (dv22 == 0 & dv20 == 0);

        % Planes intersect
        % For each triangle there are 3 cases:
        %   case=0 - v0 is on opposite side of v1 and v2
        %   case=1 - v1 is on opposite side of v0 and v2
        %   case=2 - v2 is on opposite side of v0 and v1

        % If dv == 0, it means that that vertex is on the
        % plane. We will say that a vertex on the plane is on
        % the opposite side of the others.
        case1 = ones(len,1);
        case1(dv11 .* dv12 > 0) = 0;
        case1(dv10 .* dv11 > 0) = 2; % v2 of face 1 is on opposite side of v0 and v1
        case1(dv12 == 0) = 2;
        case1(dv11 == 0) = 1;
        case1(dv10 == 0) = 0; % v0 of face 1 is on plane of face 2
        
        case2 = ones(len,1);
        case2(dv21 .* dv22 > 0) = 0;
        case2(dv20 .* dv21 > 0) = 2;
        case2(dv22 == 0) = 2;
        case2(dv21 == 0) = 1;
        case2(dv20 == 0) = 0;
        
        % Now that we know that the T1 intersects with Plane2 and T2
        % intersects with P1, we can project the vertices of the
        % triangles onto an appropiate axis.
        D = cross(repmat(N(i2,:),len,1),N(rangeF,:),2);
        [~,Didx] = max(abs(D),[],2);
        v0r = v0(rangeF,:);
        v1r = v1(rangeF,:);
        v2r = v2(rangeF,:);
        pv10 = v0r((Didx-1)*len+(0:len-1)'+1);
        pv11 = v1r((Didx-1)*len+(0:len-1)'+1);
        pv12 = v2r((Didx-1)*len+(0:len-1)'+1);
        pv20 = v0(i2,:); pv20 = pv20(Didx)';
        pv21 = v1(i2,:); pv21 = pv21(Didx)';
        pv22 = v2(i2,:); pv22 = pv22(Didx)';

        % Compute intervals of triangles intersecting with planes
        t11 = zeros(len,1); t12 = zeros(len,1);
        c10 = (case1 == 0); c11 = (case1 == 1); c12 = (case1 == 2);
        t11(c10) = pv12(c10)+(pv10(c10)-pv12(c10)).*(dv12(c10)./(dv12(c10)-dv10(c10)));
        t12(c10) = pv10(c10)+(pv11(c10)-pv10(c10)).*(dv10(c10)./(dv10(c10)-dv11(c10)));
        t11(c11) = pv10(c11)+(pv11(c11)-pv10(c11)).*(dv10(c11)./(dv10(c11)-dv11(c11)));
        t12(c11) = pv11(c11)+(pv12(c11)-pv11(c11)).*(dv11(c11)./(dv11(c11)-dv12(c11)));
        t11(c12) = pv11(c12)+(pv12(c12)-pv11(c12)).*(dv11(c12)./(dv11(c12)-dv12(c12)));
        t12(c12) = pv12(c12)+(pv10(c12)-pv12(c12)).*(dv12(c12)./(dv12(c12)-dv10(c12)));
        
        t21 = zeros(len,1); t22 = zeros(len,1);
        c20 = (case2 == 0); c21 = (case2 == 1); c22 = (case2 == 2);
        t21(c20) = pv22(c20)+(pv20(c20)-pv22(c20)).*(dv22(c20)./(dv22(c20)-dv20(c20)));
        t22(c20) = pv20(c20)+(pv21(c20)-pv20(c20)).*(dv20(c20)./(dv20(c20)-dv21(c20)));
        t21(c21) = pv20(c21)+(pv21(c21)-pv20(c21)).*(dv20(c21)./(dv20(c21)-dv21(c21)));
        t22(c21) = pv21(c21)+(pv22(c21)-pv21(c21)).*(dv21(c21)./(dv21(c21)-dv22(c21)));
        t21(c22) = pv21(c22)+(pv22(c22)-pv21(c22)).*(dv21(c22)./(dv21(c22)-dv22(c22)));
        t22(c22) = pv22(c22)+(pv20(c22)-pv22(c22)).*(dv22(c22)./(dv22(c22)-dv20(c22)));
        
        % Create intervals
        T1 = sort([t11 t12],2);
        T2 = sort([t21 t22],2);
        
        % Check for interval overlap
        % T2(1) < T1(2) < T2(2))
        cnd1 = (T1(:,2) - T2(:,1)) > epsilon;
        cnd2 = (T2(:,2) - T1(:,2)) > epsilon;
        % T1(1) < T2(2) < T1(2)
        cnd3 = (T2(:,2) - T1(:,1)) > epsilon;
        cnd4 = (T1(:,2) - T2(:,2)) > epsilon;
        
        cnd = (cnd1 & cnd2) | (cnd3 & cnd4);
        cnd = cnd & planeintersect;
        
        remove(rangeF(cnd)) = 1;
        if sum(cnd) > 0
            remove(i2) = 1;
        end
        
        % Planes are identical
        % Here we project the faces onto an appropriate plane and check if
        % the vertices of one face are within the other face
        
        % Project vertices onto plane
        [~,idx] = max(abs(N),[],2);
        idx = idx(rangeF);
        proj = zeros(len,2);
        proj(idx==1,:) = repmat([2,3],sum(idx==1),1);
        proj(idx==2,:) = repmat([1,3],sum(idx==2),1);
        proj(idx==3,:) = repmat([1,2],sum(idx==3),1);
        ppv10 = zeros(len,2);
        ppv11 = zeros(len,2);
        ppv12 = zeros(len,2);
        ppv10(:,1) = v0r((proj(:,1)-1)*len+(0:len-1)'+1);
        ppv10(:,2) = v0r((proj(:,2)-1)*len+(0:len-1)'+1);
        ppv11(:,1) = v1r((proj(:,1)-1)*len+(0:len-1)'+1);
        ppv11(:,2) = v1r((proj(:,2)-1)*len+(0:len-1)'+1);
        ppv12(:,1) = v2r((proj(:,1)-1)*len+(0:len-1)'+1);
        ppv12(:,2) = v2r((proj(:,2)-1)*len+(0:len-1)'+1);
        ppv20 = zeros(len,2);
        ppv21 = zeros(len,2);
        ppv22 = zeros(len,2);
        v0i2 = v0(i2,:); v1i2 = v1(i2,:); v2i2 = v2(i2,:);
        ppv20(:,1) = v0i2(proj(:,1));
        ppv20(:,2) = v0i2(proj(:,2));
        ppv21(:,1) = v1i2(proj(:,1));
        ppv21(:,2) = v1i2(proj(:,2));
        ppv22(:,1) = v2i2(proj(:,1));
        ppv22(:,2) = v2i2(proj(:,2));
        
        area = 0.5 *(-ppv11(:,2).*ppv12(:,1) + ppv10(:,2).*(-ppv11(:,1) + ppv12(:,1)) + ppv10(:,1).*(ppv11(:,2) - ppv12(:,2)) + ppv11(:,1).*ppv12(:,2));
        s0 = (1./(2.*area)).*(ppv10(:,2).*ppv12(:,1) - ppv10(:,1).*ppv12(:,2) + (ppv12(:,2) - ppv10(:,2)).*ppv20(:,1) + (ppv10(:,1) - ppv12(:,1)).*ppv20(:,2));
        s1 = (1./(2.*area)).*(ppv10(:,2).*ppv12(:,1) - ppv10(:,1).*ppv12(:,2) + (ppv12(:,2) - ppv10(:,2)).*ppv21(:,1) + (ppv10(:,1) - ppv12(:,1)).*ppv21(:,2));
        s2 = (1./(2.*area)).*(ppv10(:,2).*ppv12(:,1) - ppv10(:,1).*ppv12(:,2) + (ppv12(:,2) - ppv10(:,2)).*ppv22(:,1) + (ppv10(:,1) - ppv12(:,1)).*ppv22(:,2));
        
        t0 = (1./(2.*area)).*(ppv10(:,1).*ppv11(:,2) - ppv10(:,2).*ppv11(:,1) + (ppv10(:,2) - ppv11(:,2)).*ppv20(:,1) + (ppv11(:,1) - ppv10(:,1)).*ppv20(:,2));
        t1 = (1./(2.*area)).*(ppv10(:,1).*ppv11(:,2) - ppv10(:,2).*ppv11(:,1) + (ppv10(:,2) - ppv11(:,2)).*ppv21(:,1) + (ppv11(:,1) - ppv10(:,1)).*ppv21(:,2));
        t2 = (1./(2.*area)).*(ppv10(:,1).*ppv11(:,2) - ppv10(:,2).*ppv11(:,1) + (ppv10(:,2) - ppv11(:,2)).*ppv22(:,1) + (ppv11(:,1) - ppv10(:,1)).*ppv22(:,2));
        
        cnd = s0 >= 0 & t0 >= 0 & s0+t0<=1;
        cnd = cnd & planesidentical;
        shared = sum(f(rangeF,:)==f(i2,1),2);
        cnd = cnd & ~shared;
        remove(rangeF(cnd)) = 1;
        if sum(cnd) > 0
            remove(i2) = 1;
        end
        cnd = s1 >= 0 & t1 >= 0 & s1+t1<=1;
        cnd = cnd & planesidentical;
        shared = sum(f(rangeF,:)==f(i2,2),2);
        cnd = cnd & ~shared;
        remove(rangeF(cnd)) = 1;
        if sum(cnd) > 0
            remove(i2) = 1;
        end
        cnd = s2 >= 0 & t2 >= 0 & s2+t2<=1;
        cnd = cnd & planesidentical;
        shared = sum(f(rangeF,:)==f(i2,3),2);
        cnd = cnd & ~shared;
        remove(rangeF(cnd)) = 1;
        if sum(cnd) > 0
            remove(i2) = 1;
        end
    end
    remove = find(remove);
    [f,v] = removeFace(f,v,remove); % Remove the face
end
function [templatePoints,CircumferenceValue  ] = template_circumference(vOnLine)
            config = self.config_cpd;
            config.scene = [vOnLine(:,1), vOnLine(:,2)]; % same input as the input to ellipse_fits
            mnX = min(min(config.model(:,1)),min(config.scene(:,1)));
            mnY = min(min(config.model(:,2)),min(config.scene(:,2)));
            mxX = max(max(config.model(:,1)),max(config.scene(:,1)));
            mxY = max(max(config.model(:,2)),max(config.scene(:,2)));     
            gdX1D = linspace(mnX,mxX,10);
            gdY1D = linspace(mnY,mxY,10);  
            gdX2D = repmat(gdX1D,length(gdX1D),1); gdX2D = gdX2D(:);
            gdY2D = repmat(gdY1D',1,length(gdX1D));gdY2D = gdY2D(:);            
            config.ctrl_pts = [gdX2D gdY2D];
            config.init_param = zeros(size(config.ctrl_pts));
            [~,templatePoints] = gmmreg_cpd(config);
            %figure; plot(vOnLine(:,1), vOnLine(:,2),'.'); hold on; DisplayPoints(circumference.templatePoints,config.scene,2); hold off;
            [CircumferenceValue,~] = getCircumference(templatePoints(:,1),templatePoints(:,2));
            
end
function [bodyPart_template_v,bodyPart_template_f] = fitting_templateToBodyParts(circ_template,n,zValue_vec,vIdxOnLine,x,y,start_vId,self)
    l_circ_template = length(circ_template);
    bodyPart_template_v = zeros((n-1)*l_circ_template,3);
    bodyPart_template_f = zeros((n-2)*2*(l_circ_template-2),3);
    IdStart = 1;
    IdEnd = l_circ_template;
%     for zValueId = 2:length(zValue_vec)-1
    for zValueId = 2:length(zValue_vec)        
        x_slice = x(vIdxOnLine{zValueId}); y_slice = y(vIdxOnLine{zValueId});
        slice = [x_slice y_slice];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        upper_slice = slice(slice(:,2)>=0,:);
        [~,us_idx]=sort(upper_slice(:,1));
        upper_slice = upper_slice(us_idx,:);      
        lower_slice = slice(slice(:,2)<0,:);
        [~,ls_idx]=sort(lower_slice(:,1),'descend');
        lower_slice = lower_slice(ls_idx,:);
        slice = [upper_slice; lower_slice];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        config = self.config_cpd;
        config.model = circ_template;
        config.scene = slice;           
        [~,matched_template] = gmmreg_cpd(config);
    %                 DisplayPoints(matched_template,config.scene,2);    
        slice_template_v = [matched_template(:,1), matched_template(:,2), repmat(zValue_vec(zValueId),size(matched_template(:,1)))];
        bodyPart_template_v(IdStart:IdEnd,:) = slice_template_v;
        if(zValueId==2)
            template_IdV_old = (IdStart:IdEnd)';
            st = 1;
            ed = l_circ_template;
        elseif(zValueId > 2)
            template_IdV = (IdStart:IdEnd)';
            %%%
            possible_startId = template_IdV([1:3,end-2:end]);
            dist_sq = (bodyPart_template_v(template_IdV_old(1),1) - bodyPart_template_v(possible_startId,1)).^2+...
                (bodyPart_template_v(template_IdV_old(1),2) - bodyPart_template_v(possible_startId,2)).^2+...
                (bodyPart_template_v(template_IdV_old(1),3) - bodyPart_template_v(possible_startId,3)).^2;
            [~,idx]=min(dist_sq);
            new_start = find(template_IdV == possible_startId(idx));
            if (new_start>1)
                template_IdV_new = template_IdV([new_start:end,1:new_start-1]);
                bodyPart_template_v(IdStart:IdEnd,:) = bodyPart_template_v(template_IdV_new,:);  
            end
            %%%
            template_IdV_old_shifted = [template_IdV_old(2:end);template_IdV_old(1)];
            template_IdV_shifted = [template_IdV(2:end);template_IdV(1)];
            bodyPart_template_f(st:ed,:) = [template_IdV_old,template_IdV,template_IdV_old_shifted];
%             bodyPart_template_f(st - 1 + ((ed-st+1)/2),:) = []; ed = ed - 1;
%             bodyPart_template_f(ed,:) = []; ed = ed - 1;            
            st = ed + 1; ed = st + l_circ_template - 1;
            bodyPart_template_f(st:ed,:) = [template_IdV,template_IdV_old_shifted,template_IdV_shifted];
%             bodyPart_template_f(st - 1 + ((ed-st+1)/2),:) = []; ed = ed - 1;
%             bodyPart_template_f(ed,:) = []; ed = ed - 1;           
            st = ed + 1; ed = st + l_circ_template - 1;
            template_IdV_old = template_IdV;
        end
        IdStart = IdEnd + 1;
        IdEnd = IdStart + l_circ_template - 1;                
    end        
    bodyPart_template_f = bodyPart_template_f + start_vId;
end
function [r_arm_template_v,r_arm_template_f,l_arm_template_v,l_arm_template_f] = templateFitting_arm(self)
    circ_template = self.circ_template_s;
    l_circ_template = length(circ_template);
    l_circ_template_s = l_circ_template;
%             l_circ_template_l = length(self.circ_template_l);

    % right arm
    x = self.v(:,1); y = self.v(:,2); z = self.v(:,3);
    z_rarmpit = self.r_armpit(3);
    zStart = min(z(self.rArmIdx));
    zEnd = z_rarmpit;

    %n = (30*5)+1;
    n=(30/3)+1;
    zValue_vec = linspace(zStart,zEnd,n);
    [~,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue_vec, self.rArmIdx);
%             start_vId = (40+40)/2*l_circ_template + (60+40)/2*l_circ_template_l;
   % start_vId = 5*(2*1*42+1*2*42+1*4*30)*l_circ_template_s;
    start_vId = (1/3)*(2*1*42+1*2*42+1*4*30)*l_circ_template_s;

%             tic;
    [r_arm_template_v,r_arm_template_f] = fitting_templateToBodyParts(circ_template,n,zValue_vec,vIdxOnLine,x,y,start_vId,self);
%             toc

    % left arm
    z_larmpit = self.l_armpit(3);
    zStart = min(z(self.lArmIdx));
    zEnd = z_larmpit; 

%             n = 32;
    zValue_vec = linspace(zStart,zEnd,n);
    [~,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue_vec, self.lArmIdx);
%             start_vId = (40+40+20)/2*l_circ_template + (60+40)/2*l_circ_template_l;
    %start_vId = 5*(2*1*42+1*2*42+1*4*30+1*1*30)*l_circ_template_s;
    start_vId = (1/3)*(2*1*42+1*2*42+1*4*30+1*1*30)*l_circ_template_s;

%             tic;
    [l_arm_template_v,l_arm_template_f] = fitting_templateToBodyParts(circ_template,n,zValue_vec,vIdxOnLine,x,y,start_vId,self);
%             toc
end
function [r_leg_template_v,r_leg_template_f,l_leg_template_v,l_leg_template_f] = templateFitting_leg(self)
circ_template = self.circ_template_s;
    l_circ_template = length(circ_template);
    l_circ_template_s = l_circ_template;

    x = self.v(:,1); y = self.v(:,2); z = self.v(:,3);
    z_crotch = self.crotch(3);
    zStart = min(z);
    zEnd = z_crotch;
    x_crotch = self.crotch(1);
    x_r = x <= x_crotch; x_l = x > x_crotch;
    %n = (42*5)+1;
    n = (42/3)+1;
    zValue_vec = linspace(zStart,zEnd,n);

    % right leg
    keepIdx = 1:length(self.v);
    keepIdx(x_l) = [];
    keepIdx = keepIdx(~ismember(keepIdx,self.rArmIdx));
%             keepIdx = keepIdx(~ismember(keepIdx,self.lArmIdx));
    [~,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue_vec, keepIdx);     
    start_vId = 0;
%             tic;
    [r_leg_template_v,r_leg_template_f] = fitting_templateToBodyParts(circ_template,n,zValue_vec,vIdxOnLine,x,y,start_vId,self);        
%             toc

    % left leg
    keepIdx = 1:length(self.v);
    keepIdx(x_r) = [];
    keepIdx = keepIdx(~ismember(keepIdx,self.lArmIdx));
    [~,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue_vec, keepIdx);     
    %start_vId = 5*(1*1*42) * l_circ_template_s;
    start_vId = (1/3)*(1*1*42) * l_circ_template_s;
    
%             tic;
    [l_leg_template_v,l_leg_template_f] = fitting_templateToBodyParts(circ_template,n,zValue_vec,vIdxOnLine,x,y,start_vId,self);             
%             toc
end
function [trunk_template_v,trunk_template_f] = templateFitting_trunk(self)
    circ_template = self.circ_template_l;
%             l_circ_template = length(circ_template);
    l_circ_template_s = length(self.circ_template_s);

    x = self.v(:,1); y = self.v(:,2); z = self.v(:,3);
    z_crotch = self.crotch(3);
    z_armpit = [self.r_armpit(3) self.l_armpit(3)];
    zStart = z_crotch;
    zEnd = min(z_armpit);
    %n = (42*5)+1;
    n=(42/7)+1;
    zValue_vec = linspace(zStart,zEnd,n);

    keepIdx = 1:length(self.v);
    keepIdx = keepIdx(~ismember(keepIdx,self.rArmIdx));
    keepIdx = keepIdx(~ismember(keepIdx,self.lArmIdx));
    [~,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue_vec, keepIdx);    
    %start_vId = 5*(2*1*42)*l_circ_template_s;
    start_vId =(1/7)*(2*1*42)*l_circ_template_s;
%             tic;
    [trunk_belowArmpit_template_v,trunk_belowArmpit_template_f] = fitting_templateToBodyParts(circ_template,n,zValue_vec,vIdxOnLine,x,y,start_vId,self);
%             toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%% may need to add the between armpits / see stitching code with Alex? 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    circ_template = self.circ_template_xl;
    z_shoulder = [self.rShoulder(3) self.lShoulder(3)];
            zStart = max(z_armpit);
   % zStart = min(z_armpit);
   
    zEnd = max(z_shoulder);
    %n = (30*5)+1;
    n=(30/3)+1;
    zValue_vec = linspace(zStart,zEnd,n);            
    keepIdx = 1:length(self.v);
    [~,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue_vec, keepIdx);    
%             start_vId = (40+40)/2*l_circ_template_s + 60/2*l_circ_template;
    %start_vId = 5*(2*1*42+1*2*42)*l_circ_template_s;
    start_vId = (1/3)*(2*1*42+1*2*42)*l_circ_template_s;
%             tic;
    [trunk_aboveArmpit_template_v,trunk_aboveArmpit_template_f] = fitting_templateToBodyParts(circ_template,n,zValue_vec,vIdxOnLine,x,y,start_vId,self);        
%             toc    

    trunk_template_v = [trunk_belowArmpit_template_v;trunk_aboveArmpit_template_v];
    trunk_template_f = [trunk_belowArmpit_template_f;trunk_aboveArmpit_template_f];
end
function [head_neck_template_v,head_neck_template_f, head_neck_bottomSliceReducted_v, head_neck_bottomSliceReducted_f] = templateFitting_headNeck(self)

%%%%%%%%%%% using actual head but reducing the number of vertices on the boundary
%%%%%%%%%%% of the bottom hole of the head to connect to the trunk template %%%%%
            circ_template = self.circ_template_s;
            l_circ_template = length(circ_template);
            l_circ_template_s = l_circ_template;
            
            headFaces = getFaces(self.f, self.headIdx);
            holeEdges = getBoundaryEdges(headFaces);
            
            srtd_i = holeEdges;
            indx_r = 1;
            for j = 1:size(holeEdges,1)-1
                temp = [holeEdges(1:indx_r-1,:);[NaN NaN];holeEdges(indx_r+1:end,:)];
                [indx_r,indx_c] = find(temp==srtd_i(j,2));
                srtd_i(j+1,:) = holeEdges(indx_r,:);
                if (indx_c == 2)
                    srtd_i(j+1,:) = flip(srtd_i(j+1,:));
                end
            end
            if (srtd_i(end,1)==srtd_i(1,1))
                srtd_i(end,:) = flip(srtd_i(end,:));
            end
            srtd_i = srtd_i(:,1);
            srtd_i = [srtd_i;srtd_i(1)];
            v_bdry_edge = self.v(srtd_i,:);
            cnd = length(srtd_i)>l_circ_template_s*2;
            head_neck_bottomSliceReducted_f = headFaces;
            while(cnd)
               [~,idx] = min(sum(diff(v_bdry_edge,1,1).^2,2));
               [IdxFaces,~]=ismember(head_neck_bottomSliceReducted_f,srtd_i(idx));
               head_neck_bottomSliceReducted_f(sum(IdxFaces,2),:)=[];
               srtd_i(idx) = [];
               v_bdry_edge(idx,:) = [];
               cnd = length(srtd_i)>l_circ_template_s*2;
            end
            
            verticesHeadWithoutBottomSlice = unique(headFaces);
            lastSliceVertices = unique(holeEdges);
            for ff = 1:size(lastSliceVertices)
                verticesHeadWithoutBottomSlice(verticesHeadWithoutBottomSlice==lastSliceVertices(ff))=[];
            end 
            
            head_neck_bottomSliceReducted_v = sort([verticesHeadWithoutBottomSlice;srtd_i]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
%%%%%%%%%%% using 2D CPD for fitting template to head %%%%%%%%%%%%%%%%%%%%%
%             l_circ_template_l = length(self.circ_template_l);
            circ_template = self.circ_template_s;
            l_circ_template = length(circ_template);
            l_circ_template_s = l_circ_template;            
            x = self.v(:,1); y = self.v(:,2); z = self.v(:,3);
            z_shoulder = [self.rShoulder(3) self.lShoulder(3)];
            zStart = max(z_shoulder);
            zEnd = max(z);
            %n = (26*5)+1;
            n = (26*0.5)+1;
            zValue_vec = linspace(zStart,zEnd,n);
            
            keepIdx = 1:length(self.v);
            [~,vIdxOnLine] = getVOnLine(self, [x,y,z], zValue_vec, keepIdx);       
%             start_vId = (40+40+20+20)/2*l_circ_template + (40+60)/2*l_circ_template_l;
            start_vId = 0.5*(2*1*42+1*2*42+1*4*30+2*1*30)*l_circ_template_s;
%             tic;
            [head_neck_template_v,head_neck_template_f] = fitting_templateToBodyParts(circ_template,n,zValue_vec,vIdxOnLine,x,y,start_vId,self);        
%             toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function [new_v1, new_v3] = rotate_person(v1, v3, alpha)
% rotate_person Rotates the v1 and v3 by angle alpha.
% Angles are in radians.
% Positive alpha values rotates counter clockwise.
    person2d = [v1 v3];
    R = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)];
    person_rotated = (R*person2d')';
    new_v1 = person_rotated(:,1);
    new_v3 = person_rotated(:,2);
end
%%% CPD-related functions:
function [circ_template_s,circ_template_l,circ_template_xl] = init_circ()

            %x = linspace(-0.1,0.1,17);
            x = linspace(-0.1,0.1,9);
            y_Sq=(0.1).^2-(x).^2;
            y = real([sqrt(y_Sq) -sqrt(y_Sq(end-1:-1:2))]);
%             x = [x x(end-1:-1:2)];
%             bin_x = linspace(-0.1,0.1,9);
%             x = unique([linspace(bin_x(1),bin_x(2),5) linspace(bin_x(2),bin_x(8),9) linspace(bin_x(8),bin_x(9),5)]);
            x = [x x(end-1:-1:2)];
            circ_template_s = [x' y'];
            
            %x = linspace(-0.2,0.2,33);
            x = linspace(-0.2,0.2,17);
            y_Sq=(0.2).^2-(x).^2;
            y = real([sqrt(y_Sq) -sqrt(y_Sq(end-1:-1:2))]);
            x = [x x(end-1:-1:2)];
            circ_template_l = [x' y'];  
            
%             x = linspace(-0.2,0.2,65);
%             y_Sq=(0.2).^2-(x).^2;
%             y = real([sqrt(y_Sq) -sqrt(y_Sq(end-1:-1:2))]);
%             x = [x x(end-1:-1:2)];
%             circ_template_xl = [x' y']; 
            
            %x_center = linspace(-0.1,0.1,34);
            x_center = linspace(-0.1,0.1,18);
            y_Sq=(0.1).^2-(x_center).^2;
            y_center = real([sqrt(y_Sq(2:end-1)) -sqrt(y_Sq(end-1:-1:2))]);
            x_center = [x_center(2:end-1) x_center(end-1:-1:2)];

            %x_right = linspace(0.1,0.2,18);
            x_right = linspace(0.1,0.2,10);
            y_Sq=(0.05).^2-(x_right-0.15).^2;
            y_right = real([sqrt(y_Sq(3:end)) -sqrt(y_Sq(end-1:-1:2))]);     
            x_right = [x_right(3:end) x_right(end-1:-1:2)];

            %x_left = linspace(-0.2,-0.1,18);
            x_left = linspace(-0.2,-0.1,10);
            y_Sq=(0.05).^2-(x_left+0.15).^2;
            y_left = real([sqrt(y_Sq(1:end-2)) -sqrt(y_Sq(end-1:-1:2))]); 
            x_left = [x_left(1:end-2) x_left(end-1:-1:2)];

            x = [x_left x_center x_right];
            y = [y_left y_center y_right];

            circ_template_xl = [x' y'];
            
%             self.mnX_c_s = min(self.circ_template_s(:,1));
%             self.mnY_c_s = min(self.circ_template_s(:,2));
%             self.mxX_c_s = max(self.circ_template_s(:,1));
%             self.mxY_c_s = max(self.circ_template_s(:,2));
end
function config = initialize_config_cpd
config.ctrl_pts = [-0.147136620000000,0.251724140000000;0.123199920000000,0.251724140000000;0.393536470000000,0.251724140000000;0.663873020000000,0.251724140000000;0.934209570000000,0.251724140000000;-0.147136620000000,0.447701150000000;0.123199920000000,0.447701150000000;0.393536470000000,0.447701150000000;0.663873020000000,0.447701150000000;0.934209570000000,0.447701150000000;-0.147136620000000,0.643678160000000;0.123199920000000,0.643678160000000;0.393536470000000,0.643678160000000;0.663873020000000,0.643678160000000;0.934209570000000,0.643678160000000;-0.147136620000000,0.839655170000000;0.123199920000000,0.839655170000000;0.393536470000000,0.839655170000000;0.663873020000000,0.839655170000000;0.934209570000000,0.839655170000000;-0.147136620000000,1.03563220000000;0.123199920000000,1.03563220000000;0.393536470000000,1.03563220000000;0.663873020000000,1.03563220000000;0.934209570000000,1.03563220000000];
%config.init_sigma = .5; config.outliers = 1; config.lambda = 1;
config.init_sigma = .1; config.outliers = 1; config.lambda = 1;
%config.beta = 1; config.anneal_rate = .97; config.tol = 1e-18;
config.beta = 0.5; config.anneal_rate = .97; config.tol = 1e-18;
config.emtol = 1e-15; config.max_iter = 100; config.max_em_iter = 10;
config.motion = 'grbf';
config.init_param = zeros(size(config.ctrl_pts));
end 
function [param, model] = gmmreg_cpd(config)
%%=====================================================================
%% $Author: bjian $
%% $Date: 2008/12/07 00:43:34 $
%% $Revision: 1.2 $
%%=====================================================================

% todo: use the statgetargs() in statistics toolbox to process parameter name/value pairs
% Set up shared variables with OUTFUN
if nargin<1
    error('Usage: gmmreg_cpd(config)');
end
[n,d] = size(config.model); % number of points in model set
if (d~=2)&&(d~=3)
    error('The current program only deals with 2D or 3D point sets.');
end

% tic
model = config.model;
scene = config.scene;
ctrl_pts = config.ctrl_pts;
sigma = config.init_sigma;
anneal_rate = config.anneal_rate;
outliers = config.outliers;
lambda = config.lambda;
max_iter = config.max_iter;
max_em_iter = config.max_em_iter;

tol = config.tol;
EMtol = config.emtol;

[n,d] = size(ctrl_pts);
[m,d] = size(model);


% Rescaling and shifting to the origin
[model, centroid, scale] = cpd_normalize(model);
[ctrl_pts, centroid, scale] = cpd_normalize(ctrl_pts);
[scene, centroid, scale] = cpd_normalize(scene);
model0 = model;


switch lower(config.motion)
    case 'tps'
        K = tps_compute_kernel(ctrl_pts, ctrl_pts);
        Pn = [ones(n,1) ctrl_pts];
        PP = null(Pn');  % or use qr(Pn)
        kernel = PP'*K*PP;
        U = tps_compute_kernel(model, ctrl_pts);
        Pm = [ones(m,1) model];
        [q,r]   = qr(Pm);
        Q1      = q(:, 1:d+1);
        Q2      = q(:, d+2:m);
        R       = r(1:d+1,1:d+1);
        TB = U*PP;
        QQ = Q2*Q2';
        A = inv(TB'*QQ*TB + lambda*kernel)*TB'*QQ;
        basis = [Pm U*PP];
    case 'grbf'
        beta = config.beta;
        basis = cpd_G(model,ctrl_pts,beta);
        kernel = cpd_G(ctrl_pts,ctrl_pts,beta);
        %A = inv(basis'*basis+lambda*sigma*sigma*kernel)*basis';
        A = basis'*basis+lambda*sigma*sigma*kernel;
    otherwise
        error('Unknown motion model');
end % end of switch

param = config.init_param;  % it should always be of size n*d
model = model + basis*param;

%it_total = 1;
%flag_stop = 0;

iter=0; E=1; ntol=tol+10;

while (iter < max_iter) && (ntol > tol)
    EMiter=0; EMtol=tol+10;  % repeat at each termperature.
    model_old = model;
    while (EMiter < max_em_iter) && (EMtol > tol)
        %disp(sprintf('EMiter=%d',EMiter));
        %disp(sprintf('E=%f',E));
        E_old = E;
        % E-step: Given model&scene, update posterior probability matrix P.
        [P,Eu] = cpd_P(model, scene, sigma, outliers);
        % M-step: Given correspondence, solve warp parameter.
        %
        switch lower(config.motion)
            case 'tps'
                tps = param(d+2:end,:);
                E = Eu + lambda/2*trace(tps'*kernel*tps); % CPD energy function.
                s=sum(P,2);
                P=P./repmat(s,1,m); % normalization such that each row sums to 1
                motion = P * scene - model0;
                tps = A*motion;
                affine = inv(R)*Q1'*(motion-TB*tps);
                param = [affine; tps];
            case 'grbf'
                E = Eu + lambda/2*trace(param'*kernel*param); % CPD energy function.
                %param = A\(basis'*motion);
                dP=spdiags(sum(P,2),0,m,m); % precompute diag(P)

                % with ctrl_pts
                param =(basis'*dP*basis+lambda*sigma^2*kernel)\(basis'*(P*scene-dP*model0));
                % when ctrl_pts is same as model0
                % param =(dP*basis+lambda*sigma^2*eye(m))\(P*scene-dP*model0);
        end
        % update model
        model = model0 + basis*param;
        EMtol = norm((E_old-E)/E_old);
        EMiter = EMiter + 1;
    end  % end of iteration/perT
    % Anneal
    sigma = sigma * anneal_rate;
    iter = iter + 1;
    ntol = norm(model_old - model);
end % end of annealing.
% toc
model = cpd_denormalize(model, centroid, scale);


end % end of function
function [P, E] = cpd_P(x, y, sigma, outliers)

if nargin<3, error('cpd_P.m error! Not enough input parameters.'); end;
if ~exist('outliers','var') || isempty(outliers), outliers = 0; end;

k=-2*sigma^2;
[n, d]=size(x);[m, d]=size(y);

P=repmat(x,[1 1 m])-permute(repmat(y,[1 1 n]),[3 2 1]);

P=squeeze(sum(P.^2,2));
P=P/k;
P=exp(P);

% compute column sums -> s
if outliers
  Pn=outliers*(-k*pi)^(0.5*d)*ones(1,m);
  s=sum([P;Pn]);
else
  s=sum(P);
end

if nnz(s)==numel(s)
    E=-sum(log(s));  % log-likelihood
    P=P./repmat(s,n,1); % normalization such that each column sums to 1
%    s=sum(P,2);
%    P=P./repmat(s,1,m); % normalization such that each row sums to 1
else
    P=[];E=[];
end

end
function G=cpd_G(x,y,beta)

if nargin<3, error('cpd_G.m error! Not enough input parameters.'); end;

k=-2*beta^2;
[n, d]=size(x); [m, d]=size(y);

G=repmat(x,[1 1 m])-permute(repmat(y,[1 1 n]),[3 2 1]);
G=squeeze(sum(G.^2,2));
G=G/k;
G=exp(G);

end
function  [X, centroid, scale] = cpd_normalize(x)

[n, d]=size(x);
centroid=mean(x);
x=x-repmat(centroid,n,1);
scale=sqrt(sum(sum(x.^2,2))/n);
X=x/scale;

end
function x =cpd_denormalize(X, centroid, scale)

[m, d]=size(X);
x=X*scale;         % scale
x=x+repmat(centroid,m,1); % move
end
function f = fixFaceOrientation_old(f,v)
% Fixes the orientation of all faces so that the vertices are in
% anti-clockwise order
% Input:
%   f - list of faces
%   v - list of vertices
% Output:
%   f - same as input but with correct order

    epsilon = 10^(-10);
    
%     % For testing invert some faces
%     p = 0.9;
%     fprintf('Inverting %d out of %d faces.\n',round(length(f)*p),length(f));
%     inv = randperm(length(f),round(length(f)*p));
%     f(inv,:) = [f(inv,1) f(inv,3) f(inv,2)];
    
    %!!!
%     fixed = [];
%     load('allfixes.mat')
%     for i = 1:length(fixed)
%         f(fixed(i),:) = [f(fixed(i),1) f(fixed(i),3) f(fixed(i),2)];
%     end
    %!!!
    
    % get faces that may need to be changed
    E = [f(:,1) f(:,2); f(:,2) f(:,3); f(:,3) f(:,1)];
    badEdges = ~ismember([E(:,2) E(:,1)], E, 'rows');
    
    
%     fixed = [newFixed;newFixed2];%!
%     repeated = [];%!

    while(sum(badEdges))
        badFaces = find(sum(reshape(badEdges',length(f),3),2));
%         newFixed = [];%!
        fprintf('Number of bad faces: %d\n',length(badFaces));%!

        % Get all unit normals of badFaces
        n1 = -v(f(badFaces,2),:) + v(f(badFaces,1),:);
        n2 = -v(f(badFaces,3),:) + v(f(badFaces,1),:);
        normal = cross(n1,n2,2);
        unit = sqrt(dot(normal,normal,2));
        normal = normal./[unit unit unit];

        % Centroid of all badFaces
        c1 = mean([v(f(badFaces,1),1) v(f(badFaces,2),1) v(f(badFaces,3),1)],2);
        c2 = mean([v(f(badFaces,1),2) v(f(badFaces,2),2) v(f(badFaces,3),2)],2);
        c3 = mean([v(f(badFaces,1),3) v(f(badFaces,2),3) v(f(badFaces,3),3)],2);
        c = [c1 c2 c3];

        % Edges of each faces
        E1 = v(f(:,2),:) - v(f(:,1),:);
        E2 = v(f(:,3),:) - v(f(:,1),:);

        for i = 1:length(normal)
%             if mod(i,100) == 0%!
%                 fprintf('Checking face: %d\n',i)%!
%             end%!
%             figure;
%             s = 2;
%             idx = v(:,1)<c(i,1)+s & v(:,1)>c(i,1)-s &...
%                   v(:,2)<c(i,2)+s & v(:,2)>c(i,2)-s &...
%                   v(:,3)<c(i,3)+s & v(:,3)>c(i,3)-s;
%             idx = find(idx);
%             closef = getFaces(f,idx);
%             plotAva(closef,v,'y');
%             hold on;
%             plot3(c(i,1),c(i,2),c(i,3),'*r');
%             quiver3(c(i,1),c(i,2),c(i,3),normal(i,1),normal(i,2),normal(i,3),'Color','r','AutoScaleFactor',10)
            
            % This follows the algorithm of Moeller and Trumbore
            D = normal(i,:);
            D = repmat(D,length(f),1);
            c0 = c(i,:);
            T = repmat(c0,length(f),1) - v(f(:,1),:);

            P = cross(D,E2,2);
            Q = cross(T,E1,2);

            PE = dot(P,E1,2);

            t = dot(Q,E2,2)./PE;
            a = dot(P,T,2)./PE;
            b = dot(Q,D,2)./PE;

            intersections = find(a+b < 1 & a > 0 & b > 0 & t>epsilon); % face intersections
            % If the number of intersections is odd, we need to
            % change the orientation.
            if mod(length(intersections),2)
                f(badFaces(i),:) = [f(badFaces(i),1) f(badFaces(i),3) f(badFaces(i),2)];
%                 newFixed = [newFixed;badFaces(i)];

%                 [~,idx] = ismember(badFaces(i),inv);
%                 if idx == 0
%                     stop = 1;
%                 end
%                 inv(idx) = [];
            end
        end
        
%         idx = ismember(newFixed,fixed);
%         repeated = [repeated;newFixed(idx)];
%         fprintf('Number of reapeated bad faces: %d\n',length(repeated))%!
%         fixed = [fixed;newFixed];
%         fprintf('Total number of fixed: %d. Of that new %d\n',length(fixed),length(newFixed))
%         fixed = unique(fixed);
%         figure();%!
%         fminusbad = f;%!
%         [~,idx] = ismember(sort(f(badFaces,:),2),sort(f,2),'rows');%!
%         fminusbad(idx,:) = [];%!
%         plotAva(fminusbad,v,'y');%!
%         plotAva(f(badFaces,:),v,'m')%!
%         plotAva(f(repeated,:),v,'g')%!
%         drawnow%!
        
        E = [f(:,1) f(:,2); f(:,2) f(:,3); f(:,3) f(:,1)];
        badEdges = ~ismember([E(:,2) E(:,1)], E, 'rows');
    end
    
%     if ~isempty(inv)
%         fprintf('%d faces were not detected.',length(inv));
%         stop = 1;
%     end
%     fprintf('All inverted faces were fixed.');
end
function f = fixFaceOrientation(f,v)
% Fixes the orientation of all faces so that the vertices are in
% anti-clockwise order
% Input:
%   f - list of faces
%   v - list of vertices
% Output:
%   f - same as input but with correct order
    
    e1 = f(:,1:2);
    e2 = f(:,2:3);
    e3 = f(:,[3,1]);
    
    % Part 0
    % Exit if all faces are oriented the same way
    badEdges = ~ismember([e1(:,[2,1]);e2(:,[2,1]);e3(:,[2,1])], [e1;e2;e3], 'rows');
    if sum(badEdges) == 0
        return
    end
    
    % Part 1
    % We first check the orientation of several faces with the
    % ray/face intersection method
    p = 0.005; % Percentage of faces to check
    epsilon = 10^(-10);
    preCheckFaces = round(linspace(1,length(f),round(length(f)*p)));
    
    % Get all unit normals of preCheckFaces
    n1 = -v(f(preCheckFaces,2),:) + v(f(preCheckFaces,1),:);
    n2 = -v(f(preCheckFaces,3),:) + v(f(preCheckFaces,1),:);
    normal = cross(n1,n2,2);
    unit = sqrt(dot(normal,normal,2));
    normal = normal./[unit unit unit];
 
    % Centroid of all preCheckFaces
    c1 = mean([v(f(preCheckFaces,1),1) v(f(preCheckFaces,2),1) v(f(preCheckFaces,3),1)],2);
    c2 = mean([v(f(preCheckFaces,1),2) v(f(preCheckFaces,2),2) v(f(preCheckFaces,3),2)],2);
    c3 = mean([v(f(preCheckFaces,1),3) v(f(preCheckFaces,2),3) v(f(preCheckFaces,3),3)],2);
    c = [c1 c2 c3];
 
    % Edges of each faces
    E1 = v(f(:,2),:) - v(f(:,1),:);
    E2 = v(f(:,3),:) - v(f(:,1),:);
 
    for i = 1:length(normal)
        % This follows the algorithm of Moeller and Trumbore
        D = normal(i,:);
        D = repmat(D,length(f),1);
        c0 = c(i,:);
        T = repmat(c0,length(f),1) - v(f(:,1),:);
 
        P = cross(D,E2,2);
        Q = cross(T,E1,2);
 
        PE = dot(P,E1,2);
 
        t = dot(Q,E2,2)./PE;
        a = dot(P,T,2)./PE;
        b = dot(Q,D,2)./PE;
 
        intersections = find(a+b < 1 & a > 0 & b > 0 & t>epsilon); % face intersections
        % If the number of intersections is odd, we need to
        % change the orientation.
        if mod(length(intersections),2)
            f(preCheckFaces(i),:) = f(preCheckFaces(i),[1,3,2]);
        end
    end
    fixed(preCheckFaces) = 0;
    
    % Part 2
    % We now fix the orientation of the adjacent faces of the previously
    % fixed faces.
    newFaces = f(preCheckFaces,:);
    while(~isempty(newFaces))

        edges = [newFaces(:,2) newFaces(:,1); newFaces(:,3) newFaces(:,2); newFaces(:,1) newFaces(:,3)];
        facesNotToAdjust = (ismember(e1,edges,'rows')) | (ismember(e2,edges,'rows')) | (ismember(e3,edges,'rows'));
        facesNotToAdjust = facesNotToAdjust & fixed;
        fixed(facesNotToAdjust) = 0;
        
        edges = edges(:,[2,1]);
        facesToAdjust = (ismember(e1,edges,'rows')) | (ismember(e2,edges,'rows')) | (ismember(e3,edges,'rows'));
        facesToAdjust = facesToAdjust & fixed;
        f(facesToAdjust,:) = f(facesToAdjust,[2,1,3]);
        fixed(facesToAdjust) = 0;
        
        newFaces = f(facesToAdjust | facesNotToAdjust,:);
    end
end
function [f,v] = removeFace(f,v,idx)
% [f,v] = removeFace(f,v,idx) remove the face with index idx.
% It remove also the vertices if they did not belong to any other face.
%
% Example:
%   [f,v] = removeFace(f,v,5)

    face = f(idx,:);
    vertices = sort(unique(face(:)),'descend');
    f(idx,:) =[]; % remove face
    % Remove the vertices if they are not part of any other face
    for i = 1:length(vertices)
        if sum(sum(f == vertices(i))) == 0;
            v(vertices(i),:) = [];
            % Adjust vertex indices in f
            idxToReduce = find(f>vertices(i));
            f(idxToReduce) = f(idxToReduce) - 1;
        end
    end
end
function check = checkFaceOrientation(f)
% check = checkFaceOrientation(f) checks the orientation of all faces.
% 'check == true' means that all faces are oriented the same
% 'check == false' does not
% Note that check will be false if the faces contain holes.
    check = false;
    E = [f(:,[1,2]);f(:,[2,3]);f(:,[3,1])];
    badEdges = ~ismember(E(:,[2,1]), E, 'rows');
    if sum(badEdges) == 0
        check = true;
    end
end
function f = fixFaceOrientation2(f,v)
% Fixes the orientation of all faces so that the vertices are in
% anti-clockwise order
% Input:
%   f - list of faces
%   v - list of vertices
% Output:
%   f - same as input but with correct order
 
    fixed = ones(length(f),1); % 0 = fixed, 1 = not fixed
    
    e1 = f(:,1:2);
    e2 = f(:,2:3);
    e3 = f(:,[3,1]);
    
    % Part 0
    % Exit if all faces are oriented the same way
    if checkFaceOrientation(f)
        return
    end
    
    % Part 1
    % We first check the orientation of several faces with the
    % ray/face intersection method
    p = 0.005; % Percentage of faces to check
    epsilon = 10^(-10);
    preCheckFaces = round(linspace(1,length(f),round(length(f)*p)));
    
    % Get all unit normals of preCheckFaces
    n1 = -v(f(preCheckFaces,2),:) + v(f(preCheckFaces,1),:);
    n2 = -v(f(preCheckFaces,3),:) + v(f(preCheckFaces,1),:);
    normal = cross(n1,n2,2);
    unit = sqrt(dot(normal,normal,2));
    normal = normal./[unit unit unit];
 
    % Centroid of all preCheckFaces
    c1 = mean([v(f(preCheckFaces,1),1) v(f(preCheckFaces,2),1) v(f(preCheckFaces,3),1)],2);
    c2 = mean([v(f(preCheckFaces,1),2) v(f(preCheckFaces,2),2) v(f(preCheckFaces,3),2)],2);
    c3 = mean([v(f(preCheckFaces,1),3) v(f(preCheckFaces,2),3) v(f(preCheckFaces,3),3)],2);
    c = [c1 c2 c3];
 
    % Edges of each faces
    E1 = v(f(:,2),:) - v(f(:,1),:);
    E2 = v(f(:,3),:) - v(f(:,1),:);
 
    for i = 1:length(normal)
    % This follows the algorithm of Moeller and Trumbore
        D = normal(i,:);
        D = repmat(D,length(f),1);
        c0 = c(i,:);
        T = repmat(c0,length(f),1) - v(f(:,1),:);
 
        P = cross(D,E2,2);
        Q = cross(T,E1,2);
 
        PE = dot(P,E1,2);
 
        t = dot(Q,E2,2)./PE;
        a = dot(P,T,2)./PE;
        b = dot(Q,D,2)./PE;
 
        intersections = find(a+b < 1 & a > 0 & b > 0 & t>epsilon); % face intersections
        % If the number of intersections is odd, we need to
        % change the orientation.
        if mod(length(intersections),2)
            f(preCheckFaces(i),:) = f(preCheckFaces(i),[1,3,2]);
        end
    end
    fixed(preCheckFaces) = 0;
    
    % Part 2
    % We now fix the orientation of the adjacent faces of the previously
    % fixed faces.
    newFaces = f(preCheckFaces,:);
    while(~isempty(newFaces))
        
        edges = [newFaces(:,2) newFaces(:,1); newFaces(:,3) newFaces(:,2); newFaces(:,1) newFaces(:,3)];
        facesNotToAdjust = (ismember(e1,edges,'rows')) | (ismember(e2,edges,'rows')) | (ismember(e3,edges,'rows'));
        facesNotToAdjust = facesNotToAdjust & fixed;
        fixed(facesNotToAdjust) = 0;
        
        edges = edges(:,[2,1]);
        facesToAdjust = (ismember(e1,edges,'rows')) | (ismember(e2,edges,'rows')) | (ismember(e3,edges,'rows'));
        facesToAdjust = facesToAdjust & fixed;
        f(facesToAdjust,:) = f(facesToAdjust,[2,1,3]);
        fixed(facesToAdjust) = 0;
        
        newFaces = f(facesToAdjust | facesNotToAdjust,:);
    end
end
function [ volume ] = vol_calc( vv, f_tmp,offset )
    v1 = [];
    v2 = [];
    v3 = [];
    for i =1:size(f_tmp,1)
        v1(i,:) = vv(double(f_tmp(i,1)),:);
        v2(i,:) = vv(double(f_tmp(i,2)),:);
        v3(i,:) = vv(f_tmp(i,3),:);
    end
    v1(:,1) = v1(:,1) + offset;
    v2(:,1) = v2(:,1) + offset;
    v3(:,1) = v3(:,1) + offset;
    v1 = [v1(:,3),v1(:,1),v1(:,2)];
    v2 = [v2(:,3),v2(:,1),v2(:,2)];
    v3 = [v3(:,3),v3(:,1),v3(:,2)];
    volume = abs(sum(SignedVolumeOfTriangle(v1,v2,v3)));
end
function [h, ext] = labelpoints (xpos, ypos, labels, varargin)
%  [h, ext] = labelpoints (xpos, ypos, labels, position, buffer, adjust_axes, varargin)
%
%   Given x and y coordinate vectors (xpos, ypos) and given
%     a vector of labels this function will label all data points
%     and output the label handles in vector, h and the label extents in ext.
%
% ***REQUIRED INPUTS (the basics)***************
%
%   'xpos' and 'ypos' are x and y coordinate vectors for your labels.  xpos & ypos
%     should be the same length.  An exception is when all labels are along the same
%     x or y coordinate.  In that case only one value is needed for that x or y coordinate
%     and it will be replicated in the code.
%     (see examples directly below).
%
%   'labels' should be a 1xN or Nx1 array of strings or numbers of the same length as xpos or ypos.
%     Alternatively it can be a single string or number that will be replicated
%     for each (x,y) coordinate.   (see examples directly below.).
%     Alternatively it can be an NxM cell array of strings and/or numbers where the columns of each
%     row will be joined with a space delimiter to create 1 label per row (see examples directly below).
%
%   Example:
%         x = [0 .1 .2]; y = [.8 .7 .6];
%         labels = {'label 1' 'label 2' 'label 3'};
%         plot(x, y, 'o'); axis([-1 1 0 1]);
%         labelpoints(x, y, labels);
%
%   Example: when all labels fall along the same X value
%         boxplot(1:10)
%         labelpoints(0.8, [3, 5.5, 8], {'25%' '50%' '75%'}, 'center');
%
%   Example: When using only 1 label that is replicated
%         x = [0 .1 .2]; y = [.8 .6 .4];
%         plot(x,y,'o'); axis([-1 1 -1 1])
%         labelpoints(x, y, 'this point')
%
%   Example: When using a 2D cell arry for labels
%        labels = {'a', 'b', 'c'; 1, 2, 3};
%        labelpoints(0.5, 0.5, labels, 'stacked', 'down'); %see options below for 'stacked' input.
%
% *** OPTIONAL INPUTS **************************
%
%   'position' (optional) describes the position of the labels relative to their
%     (x,y) location by entering one of the following abbreviated compass directions
%     in single quotes (default is 'NW'):   N, S, E, W, NE, NW, SE, SW, center, C
%
%   'buffer' (optional) is a number generally between 0:1 that adds distance between
%     the label and the plotted point where 0 (default) is 0 distance and 1 is 1/10th
%     of the axis length.  Ignored for 'center' position.  The value can be greater than
%     1 or less than 0 too where negative buffer will shift in the opposite direction.
%
%   'adjust_axes' (optional, default=0): depending on the positioning of labels,
%     some may fall beyond the axis limits. adjust_axes = 1 will re-adjust xlim
%     & ylim slightly so labels are not beyond axis limits.
%
%   Examples:
%         x = [0 0 0]; y = [.8 .7 .6];
%         labels = {'label 1' 'label 2' 'label 3'};
%         plot(x, y, 'o')
%         labelpoints(x, y, labels, 'SE', 0.5, 1);
%
% *** OPTIONAL PARAMETERS **************************
%   The following parameters may also be entered in any order
%   * 'FontSize', N -   font size of all labels
%   * 'FontWeight', S - font weight 'bold' or 'normal'
%   * 'interpreter', s - set the text interpreter (default is 'none')
%   * 'Color', S    -   font color of all labels ('y' 'm' 'c' 'r' 'g' 'b' 'w' 'k') or RGB vector [x x x]
%                       Multiple colors may be entered as rows of RGB vector and your pattern will be replicated
%                       to match the length of your lables. See example.
%   * 'BackgroundColor', S - RGB triplet or character vector of color name for text background.
%   * 'rotation', N -   will rotate the labels N degrees and will place center of label according to 'position' input.
%                       (positive is counterclockwise). You might slightly increase buffer for rotated text.
%       Example:
%         x = [4 8 19 5 3]; bar(x);
%         labelpoints(1:5, x, {'dog' 'cat' 'bee' 'frog' 'ant'}, 'N', 0.7, 1, 'FontSize', 14, 'Color', 'm', 'rotation', 45)
%       Example with Multiple Colors
%         labelpoints(1:5, x, {'dog' 'cat' 'bee' 'frog' 'ant'}, 'N', 0.7, 1, 'color', [0 0 0; 1 0 0; 0 0 1]);
%
%   'stacked' (optional): Automatically stacks multiple labels vertically or horizontally. 
%       When using 'stacked' only the first xpos and ypos coordinate will be used to plot 
%       the 1st label and the following coordinates will be determined by the desired stack type.  
%       Stack types include:
%       * 'down' - stacks vertically, downward. xpos and ypos indicate position of top most label. (default)
%       * 'up'   - stacks vertically, upward.  xpos and ypos indicate position of bottom label.
%       * 'right'- stacks horizontally, rightward.  xpos and ypos indicate pos of left most label.
%       * 'left' - stacks horizontally, leftward. xpos and ypos indicate pos of right most label.
%       You can also adjust the spacing between the labels like this:
%       * 'down_1.5' increases the inter-label distance by x1.5(default is 1)
%       * 'left_0.5' decreases the inter-label distance by x0.5
%       Values >1 increase the distances, values between 0:1 cause labels to overlap.
%       Note that 'position', 'buffer', 'FontSize', and 'rotation' inputs still affect all labels.
%       You cannot use any outliers when using 'stacked'. 
%
%       Examples:
%         bar([0 4 8 19 5 3])
%         labelpoints(0.5, 14, {'a','b','c','d','e','f','g'}, 'stacked', 'down');
%         labelpoints(0.3, 19, {'A','B','C','D','E','F','G'}, 'stacked', 'right_2.8')
%         labelpoints(max(xlim), min(ylim), {'A','B','C','D','E','F','G'}, 'stacked', 'up')
%
%   'outliers...'
%       This function includes optional parameters to identify and label only the outliers.
%       This can come in handy when you want to avoid cluttered labels or only want to see outlier labels.  
%       NaNs values will be ignored. There are several ways to identify outliers: (may require stats toolbox)
%       'outliers_SD', N   -  will only label points that are greater than N standard deviations from
%                             mean(xpos) or mean(ypos). Alternatively, label N standard deviations from
%                             any (X,Y) point by making N a 3-element vector [N standard dev, X, Y].
%                           * Example:
%                               x = normrnd(0, 10, 1, 200); y=normrnd(0, 10, 1, 200); plot(x,y,'o');
%                               labelpoints(x,y,1:200,'outliers_SD', 2);
%                           * Example:
%                               x = normrnd(0, 10, 1, 200); y=normrnd(0, 10, 1, 200); plot(x,y,'o');
%                               labelpoints(x,y,1:200,'outliers_SD',[2,10,-20]);
%       'outliers_Q', N    -  will only label points that are greater than N times the interquartile range
%                             of xpos or ypos.
%                           * Example:
%                               x = normrnd(0, 10, 1, 200); y=normrnd(0, 10, 1, 200); plot(x,y,'o');
%                               labelpoints(x,y,1:200,'outliers_Q', 1);
%       'outliers_N', N    -  will calculate the distance of each point from the mean point(xpos,ypos)
%                             and will label the N-furthest points. Alternatively, N can be a decimal
%                             to label N% of the points. Alternatively, label the N (or 0.N) furthest
%                             points from any (X,Y) point by making N a 3-element vector [N, X, Y];
%                           * Example: (label 20 furthest points)
%                               x = normrnd(0, 10, 1, 200); y=normrnd(0, 10, 1, 200); plot(x,y,'o');
%                               h = labelpoints(x,y,1:200,'outliers_N', 20);
%                           * Example:  (label 20% furthest points)
%                               x = normrnd(0, 10, 1, 200); y=normrnd(0, 10, 1, 200); plot(x,y,'o');
%                               labelpoints(x,y,1:200,'outliers_N', 0.2);
%                           * Example:  (labels 5 furthest points from (10, -10))
%                               x = normrnd(0, 10, 1, 200); y=normrnd(0, 10, 1, 200); plot(x,y,'o');
%                               labelpoints(x,y,1:200,'outliers_N', [5,10,-10]);
%       'outliers_lim', {[x1 x2; y1 y2]; tag}
%                          -  labels points that are greater and less than the x and/or y value ranges.
%                             [x1 x2; y1 y2] are ranges of x and y values describing a boundary where points
%                             should or should not be labeled.
%                             'tag' can be either 'and', 'or', or 'invert'. (described below).
%                             When 'tag' is 'and', only points outside of x *and* y boundaries are labeled.
%                             When 'tag' is 'or' (default), points that are outside of the x *or* y boundaries
%                             are labeled.  (see examples for clarity).
%                             When 'tag' is 'invert', only points that fall *within* the x and y boundary are labeled.
%                             Use inf or -inf in the x1,x2,y1,y2 values to have a limitless bound.
%                           * Example: ('and')
%                               x = normrnd(0, 10, 1, 100); y=normrnd(0, 10, 1, 100); plot(x,y,'o');
%                               labelpoints(x,y,1:100,'outliers_lim', {[-5, 5; -10, 0]; 'and'}); hold on;
%                               line([-5, -5; 5, 5; min(xlim) max(xlim); min(xlim) max(xlim)]', [min(ylim) max(ylim); min(ylim) max(ylim); -10 -10; 0 0]', 'color', 'r')
%                           * Example: ('or')
%                               (change 'and' to 'or' in the example above)
%                           * Example: ('invert')
%                               (change 'and' to 'invert' in the example above)
%       'outliers_lin', {slope, Y-int, type, num}
%                          -  label points that have high residual values according to a linear
%                             fit of the data.  'Slope' & 'Y-int' are the slope and y intercept of the
%                             line that is being compared to the data.  The 'Slope' and 'Y-int' can be
%                             left out or empty and the regression line will be calculated based on the
%                             data points. Example:  {'sd', 1.5} or {'','','N',0.3}
%                             'TYPE' describes the type of outliers to be detected. When TYPE is 'SD',
%                             'NUM' standard deviations of the y value along the regression line will be labeled
%                             Example:  {1,0,'SD',2}. When 'TYPE' is 'N', if NUM<1, NUM% of the greatest residuals
%                             will be labeled.  If NUM>1 the NUM greatest residuals will be labeled.
%                             Examples: {1,0,'N', 10} or {1,0,'N',0.25}
%                           * Examples:
%                               x=[0:0.01:1]; y=[0:0.01:1]+rand(1,101); labs=[1:101];plot(x,y, 'o');
%                               labelpoints(x,y,labs,'outliers_lin',{'sd', 1.5});
%                               labelpoints(x,y,labs,'outliers_lin',{1,0,'sd',1.5});
%                               labelpoints(x,y,labs,'outliers_lin',{'N', 3});
%       'outliers_rand, N  -  will select N random points (no necessarily outliers) to label when N is a positive
%                             intiger or will select N% random points to label when N is a decimal (ie, 0.4 labels 40%).
%
%   To change text parameters post-hoc, use the output which is a vector of handles for all text elements.
%     Example:  h=labelpoints(....);  set(h, 'interpreter', 'latex');
%   If labelpoints is not writing to the desired axes, make the axes current before labeling using axes(axis-handle).
%   To reposition labels, use the arrow pointer in the plot window, select the label(s) and reposition.
%
%   For more examples, see additional comments within the code.  
%
%  140331 v.1
%  141115 v.2
%  151230 v.3
%  171012 v.4
%  171026 v4.1
% Copyright (c) 2014, Adam Danz  adam.danz@gmail.com
%All rights reserved
% source: http://www.mathworks.com/matlabcentral/fileexchange/46891-labelpoints
% *** MORE EXAMPLES *********************************
%
%      %Make Fake Data:
%       x = 1:10;  y=rand(1,10);
%       scatter(x,y)
%       labs = {'a' 'b' 'c' 'd' 'middle' 'f' 'g' 'h' 'i' 'last'};
%
%      Label Examples
%       txt_h = labelpoints(x, y, labs);                   %Required inputs
%       txt_h = labelpoints(x, y, labs, 'E', 0.15);        %Specify location of label relative to point
%       txt_h = labelpoints(x, y, labs, 'E', 0.15, 1);     %Expand axes if labels extend beyond axes
%       txt_h = labelpoints(x, y, labs, 'W', 0.15, 1, 'FontSize', 14, 'Color', 'r');            %Change font size and color
%       txt_h = labelpoints(x, y, labs, 'W', 0.4, 1, 'FontSize', 12, 'Color', [0 .5 .1], 'rotation', 45);  %Rotate data label
%       txt_h = labelpoints(min(xlim), max(ylim), labs, 'SE', .18, 'stacked', 'down');          %Stack data labels downward
%       txt_h = labelpoints(max(xlim), min(ylim), 1:10, 'NW', .18, 'stacked', 'left_2.5');      %Stack data labels leftward with 2.5 spacing.
%
%      Also try these labels:
%       labs = [1:1:10];
%       labs = {'Sofia' '' '' '' 'Bucharest' '' '' 'Belgrade' '' 'Ankara'}
%       labs = '*';
%       labs = 'any string';
%
%      More Outlier Examples
%       Generate Fake Data:
%            x = [rand(1,30), rand(1,8)*2];
%            y = [rand(1,30), rand(1,8)*2];
%            scatter(x, y)
%            labs = 1:38;
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_N', 5);                  %will label 5 furthest points from mean(x,y)
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_N', 0.1, 'Color', 'r');  %will label 10% of furthest points from mean
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_N', [8,1,0.25]);         %will label 8 furthest points from (1, 0.25)
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_SD', 2);                 %will label all points > 2 SD from mean
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_SD', [2,1,0.25]);        %will label all points > 2 SD from (1, 0.25)
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_Q', 1.5);                %will label points greater that 1.5 x IQR
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_lim', [0 1])             %will label points where x<0 or x>1 or y<0 or y>1
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_lim', [-inf 1; -inf 0.5])  %...where x>1 or y>0.5
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_lim', {[.4 1.2; 0.2 0.6]; 'and'}) %...where x<.4 or >1.2 AND y<0.2 or y>0.6
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_lim', {[.4 1.2; 0.2 0.6]; 'invert'}) %...where x>.4 or <1.2 AND y>0.2 or y<0.6
%        labelpoints(x, y, labs, 'N', 0.1, 1, 'outliers_lim', {[-inf inf; -inf 0.6]}) %Labels any Y value greater than 0.6
%        labelpoints(x, y, labs, 'N', 'outliers_lin', {'sd',1})                  %will label all outliers that are 1sd from regression line  NOTE: MY FAKE DATA IS NOT LINEAR!
%        labelpoints(x, y, labs, 'N', 'outliers_lin', {1,0,'sd',1})              %will label all outliers that are 1sd from unity line (y=x)
%        labelpoints(x, y, labs, 'N', 'outliers_lin', {0,0.5,'N',5})             %will label the top 5 values furthest from y=0.5
%
%   Alternative use:
%     Density Distributions:
%       x = randn(1,100); y = 1:100;
%       scatter(x,y)
%       labs = '|';
%       txt_h = labelpoints(x, 8, labs, 'center');
%
%     Single Labeling
%       x = 2004:2013;  y=rand(1,10);
%       plot(x,y,'-o')
%       labs = 'acquisition';
%       labelpoints(x(3), y(3), labs, 'N', 0.2, 1);
%       labelpoints(2008.5, min(ylim), {['labelpoints.m   ', datestr(now, 'mm/dd/yy')]}, 'N', 0.3, 0, 'fontsize', 12, 'color', 'm');
%
%     Use labels instead of markers
%       x = randn(1,15); y = randn(1,15);
%       labs = char('a'+(1:15)-1)';
%       labelpoints(x, y, labs, 'center', 0, 1, 'color', 'b');
%
%     Place notes on your plot
%       x = randn(1,100); y = 1:100;
%       scatter(x,y); xlim([-6,4])
%       labelpoints(-6, 100, {date, 'New York', '8-day span', 'p<0.001', ' ', 'This is easy'}, 'SE', 0.3, 'stacked', 'down')
%
% Changes history
%   11/02/14    if labels are entered as char, they are convered to cell
%   11/03/14    varargin added to accomodate auto_outlier feature
%   11/03/14    convert inputs to row vector if they are column vectors
%   11/04/14    now position in put is not case sensitive
%   11/04/14    now 1 label can be entered to label multiple points
%   11/04/14    now fontsize and color can be specified by params
%   11/05/14    changed 'outlier' to 'outlier_SD' and added 'outlier_Q'
%   11/07/14    added option to rotate test
%   11/15/14    added 'outliers_N' input option and cleaned up text rotation section.
%   11/19/14    curr_extent is not always a cell.  fixed.
%   11/20/14    when outliers_N is selected N is min(N, lenght(xpos));
%   11/21/14    removes entire point and label when there is an 'inf' value
%   12/14/14    fixed 'curr_extent' in rotation section
%   08/22/15    edited help file (buffer can be <>0/1)
%   09/01/15    commented-out warning that there are no outlier to label.
%   09/13/15    added outliers_lim option & set interpreter to none.
%   09/19/15    added ext output
%   09/20/15    removed validity check for color and allow 3-el vectors now.
%   10/09/15    added if nargout > 0.  !! Changed outlier section to deal with NaNs.
%               The code performs stats (ie, nanmedian, nanstd) on all data but when chosing labels,
%               only considers xpos ypos pairs that do not have NaN. See pairIDX.
%               changed text rotation to always center and added 'factor' section.
%   10/28/15    Changed outlier section from if/then to switch case. Added outlier_lin section!
%   12/21/15    Added 'stacked' input option - major addition, no change to previous functionality.
%   12/30/15    adjustments to help section; combined 2 sections at beginning that deal with text rotation.
%               *Changed medians to means in outlier section.
%               *Added more flexibility to some outlier inputs so user can identify their own center.
%               *added option to use "c" instead of "center" for position.
%               uploaded vs.3 to matlab central
%   02/02/16    Added 'if nlabs>1 ' to deal with text stacks of 1.
%   04/01/16    Improved examples and instructions.  Tested with matlab 2016.
%   09/11/16    Added interpreter as param.
%   10/22/16    added axHand and added auto_offset option. 
%   12/12/16    added fontweight
%   12/15/16    added warning when xscale or yscale is not linear and buffer is not 0. added 'buf'.
%   01/12/17    added ability to enter cell MxN cell arrays for stacked text which joins the cols of the array.
%   01/15/17    added outlier type outlier_rand.
%   01/19/17    allowed for multiple colors.
%   08/14/17    added BackgroundColor param.
% 	08/31/17    added axHand param.
%   10/12/17    adapted code to work with datetime axes. Cleaned up some code and comments.  
%   10/26/17    replaced use of isdatetime() with isa()
%% Input validity
% Check Class of 'labels'
%If 'labels' are numerical, convert to cell
if isnumeric(labels) == 1
    labels = num2cell(labels);
end
% if 'labels' are char, convert to cell
if ischar(labels)
    labels = cellstr(labels);
end
%if user entered a 2D cell array for labels, join column of each row so number of labels == number of rows. (added 171201)
labels_size = size(labels);
if labels_size(1) >1 && labels_size(2) >1
    %determine if any elements of cell are numeric and convert to str
    numericIdx = cellfun(@isnumeric, labels);
    labels(numericIdx) = strsplit(num2str([labels{numericIdx}]));
    %join column of each row with 'space' delimiter
    tempLabs = cell(1, size(labels,1));
    for r = 1:size(labels,1)
        tempLabs{r} = strjoin(labels(r,:), ' ');
    end
    labels = tempLabs;
elseif length(labels_size) > 2
    error('''LABELS'' may be one or two dimensional.')
end
% if all labels share the same xpos or ypos (only 1 value entered in 1 of the position vectors)
if length(xpos)==1 && length(ypos)>1
    xpos = repmat(xpos, size(ypos));
elseif length(ypos)==1 && length(xpos)>1
    ypos = repmat(ypos, size(xpos));
end
% if only one label is entered for all points, replicate it
if length(labels)==1 && length(xpos) > 1
    labels = repmat(labels, [1, length(xpos)]);
end
% ensures xpos, ypos, and labels are all row vectors
if iscolumn(xpos);      xpos = xpos';       end
if iscolumn(ypos);      ypos = ypos';       end
if iscolumn(labels);    labels = labels';   end
%By this point, xpos, ypos and labels should all be same length EXCEPT if optional input 'stacked' is used
% indicate error otherwise.
if isequal(length(xpos), length(ypos), length(labels)) == 0 && sum(strcmp('stacked', varargin))==0
    error('xpos, ypos, and labels must all be the same length unless using one input for labels.')
end
%if an 'inf' value is entered, this will eliminate that entire point and label
if isa(xpos, 'double')
    xinf = find(xpos==inf);
    yinf = find(ypos==inf);
    findinf = [xinf yinf];
    if ~isempty(findinf)
        xpos(findinf)=[];
        ypos(findinf)=[];
        labels(findinf) = [];
    end
end
%Validate inputs and optional parameters
%for help, see https://www.mathworks.com/help/matlab/ref/inputparser.addparameter.html#inputarg_validationFcn
% and https://www.mathworks.com/help/matlab/ref/validateattributes.html
validPositions = {'N' 'NE' 'E' 'SE' 'S' 'SW' 'W' 'NW' 'center' 'C'};
checkPosition = @(x) any(validatestring(x, validPositions));
checkCoordinate = @(x) (isnumeric(x) | isa(x, 'datetime'));
p = inputParser;
p.FunctionName = mfilename;
addRequired(p, 'xpos', checkCoordinate);
addRequired(p, 'ypos', checkCoordinate);
addRequired(p, 'labels');
addOptional(p, 'position', 'NW', checkPosition);
addOptional(p, 'buffer', 0, @isnumeric);
addOptional(p, 'adjust_axes', 0, @isnumeric);
addParameter(p, 'outliers_SD', 3, @isnumeric);
addParameter(p, 'outliers_Q', 1.5, @isnumeric);
addParameter(p, 'outliers_N', 1, @isnumeric);
addParameter(p, 'outliers_lim', [0,0;0,0]);
addParameter(p, 'outliers_lin', {'SD',1});
addParameter(p, 'outliers_rand', 0.5, @isnumeric);
addParameter(p, 'stacked', 'down');
addParameter(p, 'axHand', 0, @ishandle);
addParameter(p, 'FontSize', 10, @isnumeric);
addParameter(p, 'FontWeight', 'normal');
addParameter(p, 'Color', 'k');
addParameter(p, 'BackgroundColor', 'none');
addParameter(p, 'rotation', 0, @isnumeric);
addParameter(p, 'interpreter', 'none');
parse(p, xpos, ypos, labels, varargin{:})
%get current axis handel
axHand = p.Results.axHand;
if axHand == 0
    axHand = gca;
end
%if xscale or yscale is not 'linear' then buffer value should==0; throw warning otherwise. 161215
buf = p.Results.buffer;
xscl = get(axHand, 'xscale');
yscl = get(axHand, 'yscale');
if (~strcmp(xscl, 'linear') || ~strcmp(yscl, 'linear')) && buf~=0
    warning('Buffer size changed to 0 due to non linear axis scales.')
    buf = 0;
end
%Indicate when plot uses datetime for x or y axes
if isa(xlim(axHand), 'datetime') && ~isa(ylim(axHand), 'datetime')
    bufferUnits = 'x_datetime';
elseif isa(ylim(axHand), 'datetime') && ~isa(xlim(axHand), 'datetime')
    bufferUnits = 'y_datetime';
elseif isa(xlim(axHand), 'datetime') && isa(ylim(axHand), 'datetime')
    bufferUnits = 'xy_datetime';
else
    bufferUnits = 'other';
end
%Set flag if user entered > 1 color and repeate the color pattern if necessary
% to match length of labels. This only works on vector format (not str format) (added 170119)
multiColor = false; %default
labelColors = p.Results.Color;
nColors = size(labelColors,1);
if isnumeric(labelColors) && nColors>1
    multiColor = true;
    if nColors < length(labels)
        labelColors = repmat(labelColors, ceil(length(labels)/nColors), 1);
    end
end
%assign position
[va, ha, u1, u2] = get_compass(upper(p.Results.position), buf, bufferUnits);
    function [va, ha, u1, u2] = get_compass(compass_str, buffer, bufferUnits)
        %calculate buffer
        switch bufferUnits
            case 'normalize'
                a = [0 1 0 1]/10;
            case 'x_datetime'
                a = [0 0, ylim(axHand)/10];
                if buf~=0 && ismember(compass_str, {'E', 'W', 'NE', 'NW', 'SE', 'SW'})
                    warning('X axis is in datetime units so label buffer must be 0 for East/West orientations.')
                end
            case 'y_datetime'
                a = [0 0, xlim(axHand)/10];
                if buf~=0 && ismember(compass_str, {'N', 'S', 'NE', 'NW', 'SE', 'SW'})
                    warning('Y axis is in datetime units so label buffer must be 0 for North/South orientations.')
                end
            case 'xy_datetime'
                a = [0 0 0 0];
                if buf~=0
                    warning('X & Y axes are in datetime units so label buffer must be 0.')
                end
            otherwise
                a = axis(axHand)/10;% I've somewhat arbitrarily divided by 10 to make 'buffer' more sensitive
        end
        
        %default u values
        u1 = 0; u2 = 0;
        
        switch upper(compass_str)
            case 'E',       va = 'middle'; ha = 'left';         u1 = a(2)-a(1);
            case 'W',       va = 'middle'; ha = 'right';        u1 = (a(2)-a(1))*-1;
            case 'N',       va = 'bottom'; ha = 'center';       u2 = a(4)-a(3);
            case 'S',       va = 'top';    ha = 'center';       u2 = (a(4)-a(3))*-1;
            case 'NE',      va = 'bottom'; ha = 'left';         u1 = (a(2)-a(1))/2;     u2 = (a(4)-a(3))/2;
            case 'NW',      va = 'bottom'; ha = 'right';        u1 = (a(2)-a(1))*-0.5;  u2 = (a(4)-a(3))/2;
            case 'SE',      va = 'top';    ha = 'left';         u1 = (a(2)-a(1))/2;     u2 = (a(4)-a(3))*-0.5;
            case 'SW',      va = 'top';    ha = 'right';        u1 = (a(2)-a(1))*-0.5;  u2 = (a(4)-a(3))*-0.5;
            case {'CENTER', 'C'},  va = 'middle'; ha = 'center';
        end
        
        %Factor in buffer (in data units)
        u1 = u1*buffer;   %applied to X
        u2 = u2*buffer;   %applied to Y
    end
%adjust u1, u2 if rotation is enabled
%  rotation centers text on data point no matter what the position input is.
%  so here we increase the distance from the point a tad to compensate for the lack of true alignment.
if p.Results.rotation~=0
    factor = 1.2; %arbitrary, 1=no change, <1 means shift toward dot, >1 means shift away from dot.
    u1 = u1*factor;
    u2 = u2*factor;
    %if we are rotating the text, we'll initially plot it centered - this must happen before plotting labells.
    va = 'middle'; ha = 'center';
end
%% stacked text section (should come before outlier section)
if sum(strcmp('stacked', varargin))==1
    %if optional input 'stacked' is being used, make sure user isn't confused by using optional incompatable params
    %if more than 1 xpos or ypos is entered...
    if length(xpos)>1 || length(ypos)>1
        warning('Only the first xpos and ypos will be used to initiate stacked text');
    end
    %if outliers are entered in input, remove them from varargin
    if ~isempty(cell2mat(regexp(varargin(cellfun(@ischar, varargin)), 'outlier')))
        warning('Cannot use stacked and outlier input parameters at the same time.  Outliers input will be ignored');
        tempvarargin = varargin;
        tempvarargin(cellfun(@isnumeric, tempvarargin)) = {'temp_replace'};    %numeric values prevent use of regexp()
        varargIDX = find(~cellfun(@isempty,regexp(tempvarargin, 'outlier')));
        varargin([varargIDX, varargIDX+1]) = [];    %removes outlier input and its associated param from vararin
    end
    
    %internal parameter defaults
    spacing = 1;
    stacktype = lower(p.Results.stacked);
    
    %detect if user added optional spacing parameter (example: 'down_1.5')
    %if detected, this separates the stacktype from spacing
    if ~isempty(strfind(stacktype, '_'))
        spacing = str2double(stacktype(strfind(stacktype, '_')+1:end));
        stacktype = stacktype(1:strfind(stacktype, '_')-1);
    end
    
    %Check that user entered a valid stacktype
    if ~any(strcmp(stacktype, {'up', 'down', 'left', 'right'}))
        error('Text stacking options are:  up, down, left, or right (not case sensitive)');
    end
    
    %clear xpos and ypos vectors after initial points
    nlabs = length(labels); %number of labels
    if nlabs>1              %this is needed if user only has 1 element in stack.
        xpos(min(2, nlabs):nlabs) = nan;
        ypos(min(2, nlabs):nlabs) = nan;
    end
    
    %get xpos and ypos for all other labels
    for s = 2:nlabs
        
        %Temperarily plot the s-1 label, get its position, then delete it.
        labhand = text(xpos(s-1)+u1 , ypos(s-1)+u2, labels(s-1), 'VerticalAlignment',va, 'HorizontalAlignment',ha, 'FontSize', p.Results.FontSize, 'Parent', axHand);
        label_extnt_norm = get(labhand, 'extent');
        delete(labhand)
        
        %Calculate xpos and ypos for label S (this came from stackedtext.m which is now obsolete)
        switch stacktype
            case 'down'
                xpos(s) = xpos(s-1);
                ypos(s) = ypos(s-1) - label_extnt_norm(4)*spacing;
            case 'right'
                ypos(s) = ypos(s-1);
                xpos(s) = xpos(s-1) + label_extnt_norm(3)*spacing;
            case 'up'
                xpos(s) = xpos(s-1);
                ypos(s) = ypos(s-1) + label_extnt_norm(4)*spacing;
            case 'left'
                ypos(s) = ypos(s-1);
                xpos(s) = xpos(s-1) - label_extnt_norm(3)*spacing;
        end %switch
    end %for s = 2
end %if sum()
%% Outlier section
%If outliers parameters are selected
outlierNames = {'outliers_SD', 'outliers_Q', 'outliers_N', 'outliers_lim', 'outliers_lin', 'outliers_rand'}; %add new outlier options here
outlier_flag = false;
%identify if/which outlier inputs is (or isn't) present
for c = 1:length(outlierNames)
    if sum(strcmp(varargin, outlierNames{c}))>0
        outlier_flag = true;
        outliertype = varargin{strcmp(varargin, outlierNames{c})};  %cell naming outlier type
    end
end
%executes chosen outlier type.  The idea is that each case (or type) serves the purpose of identifying what outliers to keep.
%so the output of each subsection is the 'outlier_idx' variable which is an index of all labels'
if outlier_flag
    % Index (1/0) of all paired data (ie, if 1 or both coordinates are NaN, pairIDX(i) is 0.
    % This may be used in identifying outliers but should NOT be used to calculate stats on xpos or ypos
    % Only works with class - double (will fail for datetime data).
    pairIDX = ~isnan(xpos) & ~isnan(ypos);
    switch outliertype
        
        case 'outliers_SD'
            SDs = p.Results.outliers_SD(1);
            % if user specified the 'center' of her data:
            if length(p.Results.outliers_SD) > 1
                Xcnt = p.Results.outliers_SD(2);
                Ycnt = p.Results.outliers_SD(3);
            else % if user did not specify center of her data:
                Xcnt = nanmean(xpos);
                Ycnt = nanmean(ypos);
            end
            outlier_idx = logical(abs(xpos - Xcnt) > SDs*nanstd(xpos)  |  abs(ypos - Ycnt) > SDs*nanstd(ypos)); %index of outliers
            
        case 'outliers_Q'
            xbounds = [prctile(xpos,25) - p.Results.outliers_Q * iqr(xpos) , prctile(xpos, 75) + p.Results.outliers_Q * iqr(xpos)];   %[lower upper] bounds of outliers
            ybounds = [prctile(ypos,25) - p.Results.outliers_Q * iqr(ypos) , prctile(ypos, 75) + p.Results.outliers_Q * iqr(ypos)];   %[lower upper] bounds of outliers
            outlier_idx = logical(ypos<ybounds(1) | ypos>ybounds(2) |  xpos<xbounds(1) | xpos>xbounds(2));
            
        case 'outliers_lim'
            %assign limits and qualifier
            limvars = p.Results.outliers_lim; %ie, {[5 5; 5 5]; 'or'}
            if iscell(limvars)
                lims = limvars{1};
                if size(limvars,1) == 1 %default
                    qualifier = 'or';
                else
                    qualifier = lower(limvars{2});
                end
            else
                lims = limvars;
                qualifier = 'or';
            end
            
            if size(lims,1) == 1
                lims = [lims;lims];
            end
            
            x_outliers = xpos<lims(1,1) | xpos>lims(1,2);  %idx of x points outside of safe zone
            y_outliers = ypos<lims(2,1) | ypos>lims(2,2);  %idx of y points outside of safe zone
            switch qualifier
                case 'or'
                    outlier_idx = x_outliers | y_outliers;
                case 'and'
                    outlier_idx = x_outliers & y_outliers;
                case 'invert'
                    x_outliers = xpos>lims(1,1) & xpos<lims(1,2);
                    y_outliers = ypos>lims(2,1) & ypos<lims(2,2);
                    outlier_idx = x_outliers & y_outliers;
                otherwise
                    error('The inputs you entered for Outliers_lim wasn''t recognized.')
            end
            
        case 'outliers_N'
            Npts = p.Results.outliers_N(1);
            % if user specified the 'center' of her data:
            if length(p.Results.outliers_N) > 1
                Xcnt = p.Results.outliers_N(2);
                Ycnt = p.Results.outliers_N(3);
            else % if user did not specify center of her data:
                Xcnt = nanmean(xpos);
                Ycnt = nanmean(ypos);
            end
            if p.Results.outliers_N<1
                N = ceil(length(xpos(pairIDX)) * Npts);
            else
                N = min(Npts, length(xpos(pairIDX)));        %ensures that user cannot label more outliers than coordinates.
            end
            meanpoint = repmat([Xcnt Ycnt], [length(xpos),1]);
            paired = horzcat(xpos', ypos');
            distances = (((meanpoint(:,1)-paired(:,1)).^2)  +  ((meanpoint(:,2)-paired(:,2)).^2)).^(1/2);       %all distances from mean
            [sorted, idx] = sort(distances, 'descend');
            idx = idx(~isnan(sorted)); %this is to ignore any NaN values in xpos or ypos that would cause an nan distance.
            outlier_idx = false(1,length(xpos));
            outlier_idx(idx(1:N))=1;
            
        case 'outliers_lin'
            %user can either enter {slope, yint, outlier type, threshold} -or- {outlier type, threshold}
            %here we control for what user entered or didn't enter
            linvars = p.Results.outliers_lin; %ie, {1,1.1,'sd',1} or {'sd',1}
            if isnumeric(linvars{1}) && ~isempty(linvars{1}) %if user entered own slope and y-int
                slope = linvars{1};
                yint = linvars{2};
                outtype = upper(linvars{3}); %outlier type (sd or n)
                outthresh = linvars{4};      %threshold (values > threshold are outliers)
            else
                %calculated slope and y-int of the (x,y) data.
                if isempty(linvars{1}); linvars = linvars([3,4]); end
                slope = nansum((xpos-nanmean(xpos)).*(ypos-nanmean(ypos))) / nansum((xpos-nanmean(xpos)).^2);
                yint = nanmean(ypos) - (slope * nanmean(xpos));                     %refline(slope,yint)   %To test
                outtype = upper(linvars{1});
                outthresh = linvars{2};
            end
            %now calculate residuals (linear estimate - y val)^2
            Yest = slope*xpos + yint;
            resid = (Yest - ypos).^2;
            %now sort residuals > to < similar to 'outliers_N'
            [sorted, idx] = sort(resid, 'descend');
            idx = idx(~isnan(sorted)); %this is to ignore any NaN values in xpos or ypos that would cause an nan distance.
            %finally, chose the outliers based on chosen method
            if strcmp(outtype, 'SD')
                outlier_idx = idx(1:sum(nanstd(sorted)*outthresh <= sorted));     %figure; plot(sorted, 'o'); addhorz(nanstd(sorted)*outthresh);  %to view sorted outliers and treshold
            elseif strcmp(outtype, 'N')
                if outthresh<1
                    N = ceil(length(idx) * outthresh);
                else
                    N = min(outthresh, length(idx));        %ensures that user cannot label more outliers than coordinates.
                end
                outlier_idx = idx(1:N);
            end
            
        case 'outliers_rand'
            %user enters either a positive intiger to label x random points or a decimal to label % random points.
            Npts = p.Results.outliers_rand(1);
            if p.Results.outliers_rand<1
                N = ceil(length(xpos(pairIDX)) * Npts);
            else
                N = min(Npts, length(xpos(pairIDX)));        %ensures that user cannot label more outliers than coordinates.
            end
            %randomly select N indicies from range of data
            outlier_idx = randsample(length(xpos(pairIDX)),N);
            
            
    end %outlier switch
    
    xpos = xpos(outlier_idx);
    ypos = ypos(outlier_idx);
    labels = labels(outlier_idx);
    
    % if any(outlier_idx) == 0;           %dispay msg if there are no outliers to label
    %     mfile = [mfilename,'.m'];
    %     disp(['There are no outliers to label in ', mfile,'.'])
    %     disp('Change outlier value for less sensitivity; See help file.');
    % end
end %outlier_flag
%% WRITE TEXT
%If there is more than 1 color element we'll need to loop through each label since matlab's text() function only allows for 1 color.
if multiColor
    hand = zeros(size(labels));
    for k = 1:length(labels)
        hand(k) = text(xpos(k)+u1 , ypos(k)+u2, labels{k}, 'VerticalAlignment',va, 'HorizontalAlignment',ha, 'FontSize', p.Results.FontSize, 'color', labelColors(k,:), 'FontWeight', p.Results.FontWeight, 'Parent', axHand);
    end
else
    %Label points all with 1 color (faster)
    hand = text(xpos+u1 , ypos+u2, labels, 'VerticalAlignment',va, 'HorizontalAlignment',ha, 'FontSize', p.Results.FontSize, 'color', labelColors, 'BackgroundColor', p.Results.BackgroundColor, 'FontWeight', p.Results.FontWeight, 'Parent', axHand);
end
extnt = get(hand, 'extent');
%Rotate text if specified
if p.Results.rotation~=0           %if rotation input is something other than 0  (use to be:  sum(strcmp(varargin, 'rotation')) == 1 )
    xl = xlim;      yl = ylim;                          %In case text rotation auto adjusts axes.
    curr_extent = get(hand, 'extent');                     %Need to store current center point of all labels since text rotation relocates position
    if iscell(curr_extent); curr_extent = cell2mat(curr_extent); end
    hold on
    curr_position = [curr_extent(:,1)+(curr_extent(:,3)/2),curr_extent(:,2)+(curr_extent(:,4)/2)];          %uses extent to locate center of label
    set(hand, 'rotation', p.Results.rotation, 'VerticalAlignment','middle', 'HorizontalAlignment','center');  	%note: text rotation changes alignment which is why they need to be centered back to specifications.
    for i = 1:length(hand)                                 %after rotation, reposition labels back to desired location
        set(hand(i), 'position', curr_position(i,:))
    end
    set(axHand, 'xlim', xl); set(axHand, 'ylim', yl);         %In case text rotation auto adjusts axes.
end
%Determine if any labels go beyond axis limits and adjust if desired  (adjust_axes = 0 or 1)
if p.Results.adjust_axes == 1   &&   ~isempty(hand)
    x_adj = sign(u1+0.0000001);                 %the addition is to avoid '0'
    y_adj = sign(u2+0.0000001);                 %the addition is to avoid '0'
    
    labelextent = get(hand, 'extent');
    if isequal(class(labelextent),'cell')
        labelextent = cat(1, labelextent{:});
    end
    xl = xlim;      yl = ylim;
    lablimX = [min(labelextent(:,1)), max(labelextent(:,1)+(labelextent(:,3).*x_adj))] +u1;
    lablimY = [min(labelextent(:,2)), max(labelextent(:,2)+(labelextent(:,4).*y_adj))] +u2;
    
    xlim([min(min(xl), min(lablimX)), max(max(xl), max(lablimX))])
    ylim([min(min(yl), min(lablimY)), max(max(yl), max(lablimY))])
    %Warning: Negative data ignored (If you're getting this it's likely becase you're using a log scale in 1 of your axes or using datetime.
end
%Turn off Latex interpreter to avoid subscripts with an underscore is used in a label
set(hand, 'interpreter', p.Results.interpreter)
%Outputs
if nargout > 0
    h   = hand;
    ext = extnt;
end

end

function self = OriginalRoatationLandmarks (CaseRot,alpha2,alpha3,self)

    self.nose_tip = returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.nose_tip);
                self.rwrist_front = returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rwrist_front);
                       self.rwrist_back= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rwrist_back);      
                       self.rwrist_lateral= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rwrist_lateral);     
                       self.rwrist_medial= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rwrist_medial);      
                       self.lwrist_front= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lwrist_front);       
                       self.lwrist_back= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lwrist_back);        
                       self.lwrist_lateral= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lwrist_lateral);     
                       self.lwrist_medial= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lwrist_medial);      
                       self.rforearm_front= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rforearm_front);
                       self.rforearm_back= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rforearm_back);
                       self.rforearm_lateral = returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rforearm_lateral);
                       self.rforearm_medial= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rforearm_medial);   
                       self.lforearm_front= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lforearm_front);     
                       self.lforearm_back= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lforearm_back);      
                       self.lforearm_lateral= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lforearm_lateral);  
                       self.lforearm_medial= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lforearm_medial);   
                       self.rbicep_front= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rbicep_front);      
                       self.rbicep_back= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rbicep_back);       
                       self.rbicep_lateral= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rbicep_lateral);    
                       self.rbicep_medial= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rbicep_medial);      
                       self.lbicep_front= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lbicep_front);      
                       self.lbicep_back= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lbicep_back);       
                       self.lbicep_lateral= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lbicep_lateral);    
                        self.lbicep_medial= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lbicep_medial);    
                        self.lowerBack= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lowerBack);        
                        self.r_hip= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.r_hip);            
                        self.l_hip= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.l_hip);            
                        self.rtoe_tip= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rtoe_tip);            
                        self.rheel_tip= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rheel_tip);           
                        self.ltoe_tip= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.ltoe_tip);            
                        self.lheel_tip = returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lheel_tip);           
                        self.r_ankle= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.r_ankle);             
                        self.rankle_medialPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rankle_medialPoint);  
                        self.rankle_lateralPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rankle_lateralPoint); 
                        self.l_ankle= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.l_ankle);             
                        self.lankle_medialPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lankle_medialPoint);  
                        self.lankle_lateralPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lankle_lateralPoint); 
                        self.rcalf_backPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rcalf_backPoint);     
                        self.rcalf_medialPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rcalf_medialPoint);   
                        self.rcalf_lateralPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rcalf_lateralPoint);  
                        self.lcalf_backPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lcalf_backPoint);     
                        self.lcalf_medialPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lcalf_medialPoint);   
                        self.lcalf_lateralPoint= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lcalf_lateralPoint);  
                        self.rthigh_front= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rthigh_front);        
                        self.rthigh_back= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rthigh_back);         
                        self.rthigh_lateral= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rthigh_lateral);      
                        self.rthigh_medial= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rthigh_medial);       
                        self.lthigh_front= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lthigh_front);        
                        self.lthigh_back= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lthigh_back);         
                        self.lthigh_lateral= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lthigh_lateral);      
                        self.lthigh_medial= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lthigh_medial);       
                        self.crotch= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.crotch);              
                        self.rShoulder= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.rShoulder);           
                        self.lShoulder= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.lShoulder);           
                        self.r_armpit= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.r_armpit);            
                        self.l_armpit= returnToOrginialOrientation(CaseRot,alpha2,alpha3,self,self.l_armpit);
end