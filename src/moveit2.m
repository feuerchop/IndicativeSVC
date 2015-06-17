function moveit2(h);
%MOVEIT   Move a graphical object in 2-D.
%   Move an object in 2-D. Modify this function to add more functionality
%   when e.g. the object is dropped. It is not perfect but could perhaps
%   inspire some people to do better stuff.
%
%   % Example:
%   t = 0:2*pi/20:2*pi;
%   X = 3 + sin(t); Y = 2 + cos(t); Z = X*0;
%   h = patch(X,Y,Z,'g')
%   axis([-10 10 -10 10]);
%   moveit2(h);
%
% Author: Anders Brun, anders@cb.uu.se
%


% Unpack gui object
gui = get(gcf,'UserData');

% Make a fresh figure window
set(h,'ButtonDownFcn',@startmovit);

% Store gui object
set(gcf,'UserData',gui);


function startmovit(src,evnt)
% Unpack gui object
gui = get(gcf,'UserData');

% Remove mouse pointer
set(gcf,'PointerShapeCData',nan(16,16));
set(gcf,'Pointer','custom');

% Set callbacks
gui.currenthandle = src;
thisfig = gcbf();
set(thisfig,'WindowButtonMotionFcn',@movit);
set(thisfig,'WindowButtonUpFcn',@stopmovit);

% Store starting point of the object
gui.startpoint = get(gca,'CurrentPoint');
set(gui.currenthandle,'UserData',{get(gui.currenthandle,'XData') get(gui.currenthandle,'YData')});

% Store gui object
set(gcf,'UserData',gui);



function movit(src,evnt)
% Unpack gui object
gui = get(gcf,'UserData');

try
if isequal(gui.startpoint,[])
    return
end
catch
end

% Do "smart" positioning of the object, relative to starting point...
pos = get(gca,'CurrentPoint')-gui.startpoint;
XYData = get(gui.currenthandle,'UserData');

set(gui.currenthandle,'XData',XYData{1} + pos(1,1));
set(gui.currenthandle,'YData',XYData{2} + pos(1,2));

drawnow;

% Store gui object
set(gcf,'UserData',gui);


function stopmovit(src,evnt)

% Clean up the evidence ...

thisfig = gcbf();
gui = get(gcf,'UserData');
set(gcf,'Pointer','arrow');
set(thisfig,'WindowButtonUpFcn','');
set(thisfig,'WindowButtonMotionFcn','');
drawnow;
set(gui.currenthandle,'UserData','');
set(gcf,'UserData',[]);

