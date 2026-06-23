% run_avatar_demo_skip_repair.m
% Build an Avatar from model_files/man.obj but skip mesh repair (steps omit 2)

% Add original_code directory to path to access Avatar.m
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'original_code'));

modelPath = fullfile('..','..','model_files','man.obj');
fprintf('Loading model (skip repair): %s\n', modelPath);

try
    avatar = Avatar(modelPath,'steps',[1,3]);
catch ME
    fprintf('Error constructing Avatar (skip repair): %s\n', ME.message);
    rethrow(ME);
end

fprintf('\n=== Avatar Summary (repair skipped) ===\n');
fields = {'volume','surfaceArea','chestCircumference','waistCircumference','hipCircumference','leftArmLength','rightArmLength'};
for i=1:length(fields)
    f = fields{i};
    if isprop(avatar,f)
        val = avatar.(f);
        if isnumeric(val)
            fprintf('%s: %g\n', f, val);
        elseif isstruct(val)
            fprintf('%s: struct with fields: %s\n', f, strjoin(fieldnames(val),', '));
        else
            fprintf('%s: %s\n', f, mat2str(val));
        end
    else
        fprintf('%s: <not present>\n', f);
    end
end

if isprop(avatar,'v') && isprop(avatar,'f')
    fprintf('Vertices: %d, Faces: %d\n', size(avatar.v,1), size(avatar.f,1));
end

fprintf('\nDemo (skip repair) completed.\n');
