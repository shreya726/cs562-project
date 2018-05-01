bfdir = "C:\Users\jrhil\Desktop\UCR_TS_Archive_2015\BeetleFly";
bfdata = "C:\Users\jrhil\Desktop\UCR_TS_Archive_2015\BeetleFly\BeetleFly_TRAIN";
eddir = "C:\Users\jrhil\Desktop\UCR_TS_Archive_2015\ElectricDevices";
eddata = "C:\Users\jrhil\Desktop\UCR_TS_Archive_2015\ElectricDevices\ElectricDevices_TRAIN";
rddir= "C:\Users\jrhil\Desktop\UCR_TS_Archive_2015\RefrigerationDevices";
rddata = "C:\Users\jrhil\Desktop\UCR_TS_Archive_2015\RefrigerationDevices\RefrigerationDevices_TRAIN";
slcdir = "C:\Users\jrhil\Desktop\UCR_TS_Archive_2015\StarLightCurves";
slcdata = "C:\Users\jrhil\Desktop\UCR_TS_Archive_2015\StarLightCurves\StarLightCurves_TRAIN";

%sets = {[bfdir bfdata] [eddir eddata] [rddir rddata] [slcdir slcdata]};
sets = {[slcdir slcdata]};
for s= 1:numel(sets)
    disp(s);
    dir = sets{s}(1);
    datafile = sets{s}(2);
    data=load(datafile);
    dsize = size(data);
    dsize = dsize(1);
    outputpaths = fullfile(dir,'shapelets',{'locations.txt','features.txt','accuracy.txt','time.txt'});
    tic
    [loc, feat, acc] = jshapelets(data);
    et = toc;%elapsed time
    fid1 = fopen(outputpaths{4},'w');
    fprintf(fid1,'%d\n',et);
    fclose(fid1);
    %dlmwrite(outputpaths{1},loc);
 %   [~,featnum] = size(feat);
 %   for j = 1:featnum
    dlmwrite(outputpaths{1},loc,'newline','pc');
    dlmwrite(outputpaths{2},feat,'newline','pc');
       
 %   end
    dlmwrite(outputpaths{3},acc,'newline','pc');
    
    for cnums = 1:20  
        tic;
        [c,U,obj_fun] = fcm(data,cnums,[2.0 100 1e-5 0]);%; the first three values are the default, according to the documentation, the 0 is to turn off verbose output
        et = toc;
        outputpaths = fullfile(dir,'fcm',{strcat(strcat('centers',num2str(cnums)),'.txt'),...
        strcat(strcat('membership',num2str(cnums)),'.txt'),strcat(strcat('time',num2str(cnums)),'.txt')});
        dlmwrite(outputpaths{3},et);
        dlmwrite(outputpaths{1},c,'newline','pc');
        dlmwrite(outputpaths{2},U,'newline','pc');
    end
    
end
fclose('all');