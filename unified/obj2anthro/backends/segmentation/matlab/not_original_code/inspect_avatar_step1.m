% inspect_avatar_step1.m
% Construct Avatar with steps=[1] and print detailed property info

% Add original_code directory to path to access Avatar.m
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'original_code'));

try
    avatar = Avatar('../../model_files/man.obj','steps',[1]);
    fprintf('\n--- Detailed Avatar Inspection ---\n');
    fields = {'list_of_properties','v','f','volume','surfaceArea','chestCircumference','waistCircumference','hipCircumference','crotch'};
    for i=1:length(fields)
        f = fields{i};
        if isprop(avatar,f)
            val = avatar.(f);
            if isnumeric(val)
                s = size(val);
                if numel(val) > 1
                    fprintf('%s: numeric, size=%s\n', f, mat2str(s));
                else
                    fprintf('%s: %g\n', f, val);
                end
            elseif isstruct(val)
                fprintf('%s: struct with fields: %s\n', f, strjoin(fieldnames(val),', '));
            else
                fprintf('%s: %s\n', f, mat2str(val));
            end
        else
            fprintf('%s: <not present>\n', f);
        end
    end
catch ME
    fprintf('Error during inspection: %s\n', ME.message);
    rethrow(ME);
end
fprintf('\nInspection finished.\n');
