% run_avatar_demo_raw.m
% Read the OBJ file and inspect raw vertex/face counts; construct Avatar with minimal steps

% Add original_code directory to path to access Avatar.m
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'original_code'));

objPath = fullfile('..','..','model_files','man.obj');
fprintf('Reading OBJ: %s\n', objPath);

try
    myObj = readObj(objPath);
    v = myObj.v;
    f = uint32(myObj.f);
    fprintf('Raw: Vertices: %d, Faces: %d\n', size(v,1), size(f,1));
catch ME
    fprintf('Error reading OBJ: %s\n', ME.message);
    rethrow(ME);
end

% Try constructing Avatar from a struct and only run step 1 (cleaning)
inputStruct.v = v;
inputStruct.f = f;

try
    avatar = Avatar(inputStruct,'steps',[1]);
    fprintf('Avatar constructed with steps=[1]\n');
    fprintf('Vertices after step1: %d, Faces after step1: %d\n', size(avatar.v,1), size(avatar.f,1));
catch ME
    fprintf('Error constructing Avatar from struct: %s\n', ME.message);
    rethrow(ME);
end

fprintf('\nDemo (raw) completed.\n');
