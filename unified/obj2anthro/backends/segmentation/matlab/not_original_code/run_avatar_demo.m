% run_avatar_demo.m
% Build an Avatar from model_files/man.obj and print key properties

% Add original_code directory to path to access Avatar.m
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'original_code'));

modelPath = fullfile('..','..','model_files','man.obj');
fprintf('Loading model: %s\n', modelPath);

try
    avatar = Avatar(modelPath);
catch ME
    fprintf('Error constructing Avatar: %s\n', ME.message);
    rethrow(ME);
end

fprintf('\n=== Avatar Summary ===\n');
% Print a few representative properties if they exist
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

% Print count of vertices/faces
if isprop(avatar,'v') && isprop(avatar,'f')
    fprintf('Vertices: %d, Faces: %d\n', size(avatar.v,1), size(avatar.f,1));
end

% Optionally show the avatar if display available
try
    if ismethod(avatar,'show')
        fprintf('Calling avatar.show() to display mesh (close figure to continue)...\n');
        avatar.show();
    end
catch
    % ignore display errors
end

fprintf('\nDemo completed.\n');
