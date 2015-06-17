function m=marker_type2(i)
% MARKER_TYPE Returns marker type.
%
% Synopsis:
%  m=marker_type(i)
%
% See also MARKER_COLOR. 
%

% About: Statistical Pattern Recognition Toolbox
% (C) 1999-2003, Written by Vojtech Franc and Vaclav Hlavac
% Czech Technical University Prague
% Faculty of Electrical Engineering
% Center for Machine Perception

% Modifications:
%  7-jan-2003, VF, created

MARKERS=['x','*','^','p','h','<','d','>','v'];

m=MARKERS(mod(i-1,length(MARKERS))+1);

return;
