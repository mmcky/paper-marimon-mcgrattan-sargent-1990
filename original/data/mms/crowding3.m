function mostsimilar=crowding(child,CS,crowdingfactor,crowdingsubpop, ..
                              nclass,lcond)
if crowdingfactor<1; 
  crowdingfactor=1; 
end;
matchmax=-1; 
mostsimilar=0;
for j=1:crowdingfactor;
  worst=1+floor(rand(crowdingsubpop*nclass,1)*nclass);
  worst=[worst,CS(worst,lcond+2)];
  worststr=find(worst(:,2)<=min(worst(:,2)));
  popmember=worst(worststr(1),1);
  match=sum([child(1:lcond)==CS(popmember,1:lcond),  ..
             child(lcond+1)~=CS(popmember,lcond+1)] );
  if match>matchmax;
    matchmax=match;
    mostsimilar=popmember;
  end;
end;

