NAMES=[14:size(SS20_HoleFilling,2)];

for i = 1:size(NAMES,2)
    fileName{i}=sprintf('Fit3D_HF_%d.obj',NAMES(i));
    %sprintf('P%d_%d_pre',P,D)
end

%open file identifier
for k=1:length(fileName)
    fid=fopen(fileName{k},'w');

    for i=14:size(SS20_HoleFilling(k).v,1)
        fprintf(fid,'v %d %d %d\n',SS20_HoleFilling(k).v(i,1) , SS20_HoleFilling(k).v(i,2),SS20_HoleFilling(k).v(i,3));
    end
    for i=14:size(SS20_HoleFilling(k).f,1)
        fprintf(fid,'f %d %d %d\n',SS20_HoleFilling(k).f(i,1) , SS20_HoleFilling(k).f(i,2),SS20_HoleFilling(k).f(i,3));
    end
    fclose(fid)
end



%%%%% Single subject
for j = 7:205
    j
    Subject = char(Naked.ID(j));
    U = Avatar(Subject,'steps',[3],'WB_SA_only','on');
    Subject = [Subject(1:end-4), '.obj']
    fid=fopen(Subject,'w');
    for i=1:size(U.v,1)
        fprintf(fid,'v %d %d %d\n',U.v(i,1) , U.v(i,2), U.v(i,3));
    end
    for i=1:size(U.f,1)
        fprintf(fid,'f %d %d %d\n',U.f(i,1) , U.f(i,2), U.f(i,3));
    end
    fclose(fid)
end 

for j = 1:8
    j
    Subject = char(TA_pose_org.ObjFile(j));
    U = Avatar_temp(Subject,'armpits_old','on','steps',[1 2 3],'Vol_SA','on');
    U = AvatarTpose(Subject,'armpits_old','on','steps',[1 2 3],'Vol_SA','on');
%     Subject = [Subject(1:end-4), '.obj']
%     fid=fopen(Subject,'w');
%     for i=1:size(U.v,1)
%         fprintf(fid,'v %d %d %d\n',U.v(i,1) , U.v(i,2), U.v(i,3));
%     end
%     for i=1:size(U.f,1)
%         fprintf(fid,'f %d %d %d\n',U.f(i,1) , U.f(i,2), U.f(i,3));
%     end
%     fclose(fid)
end 