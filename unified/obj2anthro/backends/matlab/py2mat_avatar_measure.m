function values = py2mat_avatar_measure(inputPath, steps)
%PY2MAT_AVATAR_MEASURE Run the legacy Pennington Avatar pipeline once.
%
% This is deliberately a shallow adapter.  Avatar.m remains the source of
% truth; this function only turns its object properties into a plain struct
% that the Python Engine can copy without attempting to marshal a MATLAB
% class instance.
%
% Avatar.extractValues is not used here.  The legacy method refers to
% surfaceArea.lLeg, while the constructor creates surfaceArea.lleg (and the
% corresponding right-leg field), so reading the fields explicitly keeps the
% original Avatar implementation untouched and works in current MATLAB.

if nargin < 2 || isempty(steps)
    steps = [3];
end

if isstring(inputPath)
    inputPath = char(inputPath);
end

steps = double(steps(:).');

% Avatar.m is run exactly as intended: the requested steps with Vol_SA on.
% There is no retry on a reduced configuration.  A mesh that Avatar.m cannot
% process is a failed mesh, and the caller records it as one -- silently
% falling back to Vol_SA=0 would return a row whose segmented surface and
% volume fields are NaN for a reason the table could not show.
avatar = Avatar(inputPath, 'steps', steps, 'Vol_SA', 'on');

values = struct();
values.n_vertices = size(avatar.v, 1);
values.n_faces = size(avatar.f, 1);
values.height = safeScalar(@() max(avatar.v(:, 3)) - min(avatar.v(:, 3)));

values.chestGirth = safeScalar(@() avatar.chestCircumference.value);
values.waistGirth = safeScalar(@() avatar.waistCircumference.value);
values.hipGirth = safeScalar(@() avatar.hipCircumference.value);
values.rThighGirth = safeScalar(@() avatar.rThighGirth.value);
values.lThighGirth = safeScalar(@() avatar.lThighGirth.value);
values.rCalfGirth = safeScalar(@() avatar.rCalfCircumference.value);
values.lCalfGirth = safeScalar(@() avatar.lCalfCircumference.value);
values.rAnkleGirth = safeScalar(@() avatar.r_ankle_girth.value);
values.lAnkleGirth = safeScalar(@() avatar.l_ankle_girth.value);
values.rBicepGirth = safeScalar(@() avatar.r_bicepgirth.value);
values.lBicepGirth = safeScalar(@() avatar.l_bicepgirth.value);
values.rForearmGirth = safeScalar(@() avatar.r_forearmgirth.value);
values.lForearmGirth = safeScalar(@() avatar.l_forearmgirth.value);
values.rWristGirth = safeScalar(@() avatar.r_wristgirth.value);
values.lWristGirth = safeScalar(@() avatar.l_wristgirth.value);

values.rArmLength = safeScalar(@() avatar.rightArmLength);
values.lArmLength = safeScalar(@() avatar.leftArmLength);
values.rLegLength = safeScalar(@() avatar.rLegLength);
values.lLegLength = safeScalar(@() avatar.lLegLength);
values.trunkLength = safeScalar(@() avatar.trunkLength);
values.crotchHeight = safeScalar(@() avatar.crotchHeight);
values.collarScalpLength = safeScalar(@() avatar.collarScalpLength);

values.SA_total = safeScalar(@() avatar.surfaceArea.total);
values.SA_head = safeScalar(@() avatar.surfaceArea.head);
values.SA_trunk = safeScalar(@() avatar.surfaceArea.trunk);
values.SA_rArm = safeScalar(@() avatar.surfaceArea.rArm);
values.SA_lArm = safeScalar(@() avatar.surfaceArea.lArm);
values.SA_rleg = safeScalar(@() avatar.surfaceArea.rleg);
values.SA_lleg = safeScalar(@() avatar.surfaceArea.lleg);
values.SA_legs = safeScalar(@() avatar.surfaceArea.legs);

values.VOL_total = safeScalar(@() avatar.volume.total);
values.VOL_head = safeScalar(@() avatar.volume.head);
values.VOL_trunk = safeScalar(@() avatar.volume.trunk);
values.VOL_rArm = safeScalar(@() avatar.volume.rArm);
values.VOL_lArm = safeScalar(@() avatar.volume.lArm);
values.VOL_rleg = safeScalar(@() avatar.volume.rleg);
values.VOL_lleg = safeScalar(@() avatar.volume.lleg);

% Corrected companions for known legacy field/index mistakes.  The original
% values above are retained so MATLAB-faithful and corrected columns can be
% compared in the unified table.
values.rAnkleGirth_fixed = safeScalar(@() avatar.l_ankle_girth.value);
values.lAnkleGirth_fixed = safeScalar(@() avatar.r_ankle_girth.value);
values.trunkLength_fixed = safeScalar(@() sqrt(...
    (avatar.crotch(1) - avatar.collar(1))^2 + ...
    (avatar.crotch(3) - avatar.collar(3))^2));
values.collarScalpLength_fixed = safeScalar(@() sqrt(...
    (max(avatar.v(:, 3)) - avatar.collar(3))^2 + ...
    (avatar.v(find(avatar.v(:, 3) == max(avatar.v(:, 3)), 1), 1) - avatar.collar(1))^2));
end

function value = safeScalar(thunk)
% Return NaN for an unavailable optional field instead of aborting a batch.
value = NaN;
try
    candidate = thunk();
    if ~isempty(candidate)
        value = double(candidate(1));
    end
catch
    % Some meshes do not produce every optional sub-region.  The row remains
    % usable and the missing measurement is represented as NaN.
end
end
