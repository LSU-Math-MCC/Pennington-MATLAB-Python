%% Fit_3D

T=fieldStr';
writetable(cell2table(T),'Anthro_baby.xlsx','WriteVariableNames',0)

for i = [1:31]
    Values = [];
    
    for j=[1:31]
        Values = [Values;ShapeUpAdultsDataStorage(i).(fieldStr{j})];
    end
    
    writetable(array2table(Values'),'Anthro_baby.xlsx','WriteRowNames',0,'WriteVariableNames',0,'Range',['A' num2str(i+1)]);
end

%% SS20

T=fieldStr';
writetable(cell2table(T),'DAs_SS20.xlsx','WriteVariableNames',0)

for i = [20:36]
    Values = [];
    
    for j=[1:22,24:30]
        Values = [Values;DAs(i).DAs.(fieldStr{j})];
    end
    
    writetable(array2table(Values'),'DAs_SS20.xlsx','WriteRowNames',0,'WriteVariableNames',0,'Range',['A' num2str(i+1)]);
end

%% Styku

T=fieldStr';
writetable(cell2table(T),'DAs_Styku.xlsx','WriteVariableNames',0)

for i = [1:35]
    Values = [];
    
    for j=[1:22,24:30]
        Values = [Values;DAs(i).DAs.(fieldStr{j})];
    end
    
    writetable(array2table(Values'),'DAs_Styku.xlsx','WriteRowNames',0,'WriteVariableNames',0,'Range',['A' num2str(i+1)]);
end
