function [locout,featout,accout] = jshapelets(data)

    cls1=data(:,1);

    locout = [];
    featout = [];
    accout = [];

    ts = 1;

    data2=data;
    ql=50;

    %figure;
    DIS=[];
    ACC=[];
    FTR=1;
    iter=1;
    while true
        [r,c]=size(data2);
        cls2 = [];
        cls2(r,1)=0;
        seps=[];
        cnt=1;
        for ql=50:50   %60,   %sony=10:35
            for i=2:c-ql
               ftr = data2(ts,i:i+ql-1);
               dis = computeMatrix(ftr,data2);
               dis = dis/sqrt(ql);

               seps(cnt,1)=0;   %gap
               seps(cnt,2)=0;   %corr
               seps(cnt,3)=0;   %q=length(ind1)/length(ind2)
               seps(cnt,4)=ql;  %ql= qlen
               seps(cnt,5)=i;   %index
               for corr=0.95:-0.01:0.65        
                    ind1 = find(dis(:,1)<sqrt(2*(1-corr)));
                    ind2 = find(dis(:,1)>sqrt(2*(1-corr)));
                    if(length(ind1)==0 || length(ind2)==0)
                        continue;
                    end
                    m1 = mean(dis(ind1,1)); 
                    s1 = std(dis(ind1,1),1); 
                    m2 = mean(dis(ind2,1));
                    s2 = std(dis(ind2,1),1);

                    q = length(ind1)/length(ind2);

                    if( q > 0.2 && q < 5 )    
                        curr = (m2-s2-(m1+s1));
                    else
                        curr = 0;
                    end
                    if curr>seps(cnt,1)
                        seps(cnt,1) = curr;
                        seps(cnt,2) = corr;
                        seps(cnt,3) = q;
                    end
               end 
               cnt = cnt+1;
            end
        end
        [a,b]=sortrows(seps,-1);
        indx = 0;
        for i=1:size(a,1)
            q=a(i,3);
            if(q>0.2 && q<5)  %0.42
                indx = i;
                break;
            end
        end
        if(indx==0)
          break; %indx=1;
        end
        ql=a(indx,4);
        ftr = data2(ts,a(indx,5):a(indx,5)+ql-1);
        dis = computeMatrix(ftr,data2);
        dis = dis/sqrt(ql);
        corr = a(indx,2);
        ind1 = find(dis(:,1)<sqrt(2*(1-corr)));
        ind2 = find(dis(:,1)>sqrt(2*(1-corr)));
        m1 = mean(dis(ind1,1)); 
        s1 = std(dis(ind1,1),1); 
        m2 = mean(dis(ind2,1));
        s2 = std(dis(ind2,1),1);

        dist=computeMatrix(ftr,data);
        dist=dist/sqrt(ql);
        DIS = [DIS dist];
        ACC(FTR) = classifyFeats(DIS,data(:,1));
%        fprintf(fid2,'%f\n',ACC(FTR));
        accout = [accout;ACC(FTR)];
        accuracy = classifyFeats(dis,cls1); 



        iter=iter+1;

        I = find(dis(:,1)<(m1+0.5*s1));
        cls2(I)=1;
        X = find(cls2==0);
        cls1 = cls1(X);
        data2=data2(X,:);
        FTR = FTR+1;

  %      fprintf(fid,'%d\t%d\t%f\n',ts,a(indx,5),a(1,2));
        locout = [locout;[ts, a(indx,5),a(1,2)]];
  %      fprintf(fid1,'%f\t',ftr);
        featout = [featout;ftr];
  %      fprintf(fid1,'\n');

        if(min(length(ind1),length(ind2))<2)
            break;
        end


        [x,y]=sort(dis,'descend');
        for i=1:r
            if(cls2(y(i))==0)
                ts=find(X==y(i));
                break;
            end
        end

    end

end
