%%
% %%%% Fit3D (3D01)/SS20 (3D02)
% UnitConvertorDist = 10;
% UnitConvertorSA = 1000000;
% UnitConvertorVol = 1000000;
% % %%%%

% %%%% Styku (3D03)
UnitConvertorDist = 1/2.54;
UnitConvertorSA = 1550;
UnitConvertorVol = 61.024;
%%%%

% % % %%%% Naked (3D04) / Human Solutions (3DO1)
% UnitConvertorDist = 1/100;
% UnitConvertorSA = 1;
% UnitConvertorVol = 1/1000;
%%

for i = [1:2]
          i
         Path = DataStorageUniversalPaper.DO1_path(i);
        % DataStorageUniversalPaper.DO1_verif(i) = DataStorageUniversalPaper.Stykuobj(i);
         %addpath(char(Path));
         Subject = char(DataStorageUniversalPaper.Stykuobj(i));
         
     if ~isempty(Subject)
                try
                    tic
                    U=Avatar(Subject,'steps',[3],'Vol_SA','on');
                    toc
                    worked = 1;
                catch exception
                    msg_error = getReport(exception)
                    %DataStorageUniversalPaper.DO1_verif(i) = msg_error;
                end
                if(worked == 1)
                    % Volume
                        DataStorageUniversalPaper.VOL_TOT(i) = U.volume.total/UnitConvertorVol;
                        DataStorageUniversalPaper.VOL_Trunk(i) = U.volume.trunk/UnitConvertorVol;	
                        DataStorageUniversalPaper.VOL_Arm_R(i)	= U.volume.rArm/UnitConvertorVol;
                        DataStorageUniversalPaper.VOL_Arm_L(i) = U.volume.lArm/UnitConvertorVol;
                        DataStorageUniversalPaper.VOL_Leg_R(i) = U.volume.rleg/UnitConvertorVol;
                        DataStorageUniversalPaper.VOL_Leg_L(i) = U.volume.lleg/UnitConvertorVol;
                    %%%%
                    
%                     % Ellipse Ratio
%                         DataStorageUniversalPaper.DA_ER_Ch(i) = U.chestCircumference.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_W(i) = U.waistCircumference.a_over_b;
%                         DataStorageUniversalPaper.DA_ER_H(i) = U.hipCircumference.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_B_R(i) = U.r_bicepgirth.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_B_L(i)	= U.l_bicepgirth.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_F_R(i) = U.r_forearmgirth.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_F_L(i) = U.l_forearmgirth.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_Wr_R(i) = U.r_wristgirth.a_over_b;
%                         DataStorageUniversalPaper.DA_ER_Wr_L(i)	= U.l_wristgirth.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_Th_R(i)	= U.rThighGirth.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_Th_L(i)	= U.lThighGirth.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_C_R(i) = U.rCalfCircumference.a_over_b;
%                         DataStorageUniversalPaper.DA_ER_C_L(i) = U.lCalfCircumference.a_over_b;
%                         DataStorageUniversalPaper.DA_ER_A_R(i) = U.r_ankle_girth.a_over_b; 
%                         DataStorageUniversalPaper.DA_ER_A_L(i) = U.l_ankle_girth.a_over_b; 
%                     %%%%%

                    %DataStorageUniversalPaper.DA_SA_TOT(i) = U.surfaceArea.total/UnitConvertorSA;
                    %DataStorageUniversalPaper.DA_SA_Trunk(i) = U.surfaceArea.trunk/UnitConvertorSA;
                    %DataStorageUniversalPaper.DA_SA_Arm_R(i) = U.surfaceArea.rArm/UnitConvertorSA;
                    %DataStorageUniversalPaper.DA_SA_Arm_L(i) = U.surfaceArea.lArm/UnitConvertorSA;
                    %DataStorageUniversalPaper.DA_SA_Leg_R(i) = U.surfaceArea.rleg/UnitConvertorSA;
                    %DataStorageUniversalPaper.DA_SA_Leg_L(i) = U.surfaceArea.lleg/UnitConvertorSA;
                    
                    DataStorageUniversalPaper.DA_CIRC_Ch(i) = U.chestCircumference.value/UnitConvertorDist; 
                    DataStorageUniversalPaper.DA_CIRC_W(i) = U.waistCircumference.value/UnitConvertorDist; 
                    DataStorageUniversalPaper.DA_CIRC_H(i) = U.hipCircumference.value/UnitConvertorDist; 
                    DataStorageUniversalPaper.DA_CIRC_B_R(i) = U.r_bicepgirth.value/UnitConvertorDist; 
                    DataStorageUniversalPaper.DA_CIRC_B_L(i) = U.l_bicepgirth.value/UnitConvertorDist;
                    DataStorageUniversalPaper.DA_CIRC_F_R(i) = U.r_forearmgirth.value/UnitConvertorDist;	
                    DataStorageUniversalPaper.DA_CIRC_F_L(i) = U.l_forearmgirth.value/UnitConvertorDist;
                    DataStorageUniversalPaper.DA_CIRC_Wr_R(i) = U.r_wristgirth.value/UnitConvertorDist;
                    DataStorageUniversalPaper.DA_CIRC_Wr_L(i) = U.l_wristgirth.value/UnitConvertorDist; 
                    DataStorageUniversalPaper.DA_CIRC_Th_R(i) =  U.rThighGirth.value/UnitConvertorDist;
                    DataStorageUniversalPaper.DA_CIRC_Th_L(i) = U.lThighGirth.value/UnitConvertorDist; 	
                    DataStorageUniversalPaper.DA_CIRC_C_R(i) = U.rCalfCircumference.value/UnitConvertorDist; 
                    DataStorageUniversalPaper.DA_CIRC_C_L(i) = U.lCalfCircumference.value/UnitConvertorDist; 
                    DataStorageUniversalPaper.DA_CIRC_A_R(i) = U.r_ankle_girth.value/UnitConvertorDist; 
                    DataStorageUniversalPaper.DA_CIRC_A_L(i) = U.l_ankle_girth.value/UnitConvertorDist; 

                    %DataStorageUniversalPaper.DA_LEN_Arm_R(i) =  U.rightArmLength/UnitConvertorDist; 
                    %DataStorageUniversalPaper.DA_LEN_Arm_L(i) = U.leftArmLength/UnitConvertorDist; 
                    %DataStorageUniversalPaper.DA_LEN_Leg_R(i) = U.rLegLength/UnitConvertorDist; 
                    %DataStorageUniversalPaper.DA_LEN_Leg_L(i) = U.lLegLength/UnitConvertorDist; 
                    
                end
         end

end 