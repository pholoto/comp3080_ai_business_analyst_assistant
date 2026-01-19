import { format, differenceInDays, eachMonthOfInterval, startOfMonth, isWithinInterval } from 'date-fns';

interface Project {
  id: string;
  name: string;
  startDate: string;
  endDate: string;
  currentPhase: number;
  totalPhases: number;
}

interface Props {
  projects: Project[];
}

export const ProjectTimelineChart = ({ projects }: Props) => {
  const today = new Date();
  
  // Find the overall date range
  const allDates = projects.flatMap(p => [new Date(p.startDate), new Date(p.endDate)]);
  if (allDates.length === 0) {
    return (
      <div className="glass-card rounded-2xl p-6">
        <h3 className="text-lg font-display font-bold mb-4">Project Timeline</h3>
        <p className="text-sm text-muted-foreground">No projects to display</p>
      </div>
    );
  }
  
  const minDate = new Date(Math.min(...allDates.map(d => d.getTime())));
  const maxDate = new Date(Math.max(...allDates.map(d => d.getTime())));
  const totalDays = differenceInDays(maxDate, minDate) || 1;
  
  // Generate month markers
  const months = eachMonthOfInterval({ start: minDate, end: maxDate });

  const getBarPosition = (startDate: string, endDate: string) => {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const left = (differenceInDays(start, minDate) / totalDays) * 100;
    const width = (differenceInDays(end, start) / totalDays) * 100;
    return { left: `${Math.max(0, left)}%`, width: `${Math.max(2, width)}%` };
  };

  const getProgressWidth = (project: Project) => {
    const progress = (project.currentPhase / project.totalPhases) * 100;
    return `${progress}%`;
  };

  const getTodayPosition = () => {
    if (today < minDate || today > maxDate) return null;
    return `${(differenceInDays(today, minDate) / totalDays) * 100}%`;
  };

  const todayPos = getTodayPosition();

  return (
    <div className="glass-card rounded-2xl p-6 col-span-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-display font-bold">Project Timeline</h3>
          <p className="text-sm text-muted-foreground">Gantt view of all project schedules</p>
        </div>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm bg-primary/30" />
            <span>Duration</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm bg-primary" />
            <span>Progress</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-0.5 h-3 bg-destructive" />
            <span>Today</span>
          </div>
        </div>
      </div>
      
      {/* Month headers */}
      <div className="relative h-8 mb-2 border-b border-border">
        {months.map((month, idx) => {
          const left = (differenceInDays(startOfMonth(month), minDate) / totalDays) * 100;
          return (
            <div 
              key={idx}
              className="absolute text-xs text-muted-foreground font-medium"
              style={{ left: `${Math.max(0, left)}%` }}
            >
              {format(month, 'MMM yyyy')}
            </div>
          );
        })}
      </div>

      {/* Timeline grid and bars */}
      <div className="relative">
        {/* Today marker */}
        {todayPos && (
          <div 
            className="absolute top-0 bottom-0 w-0.5 bg-destructive z-10"
            style={{ left: todayPos }}
          >
            <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] text-destructive font-medium whitespace-nowrap">
              Today
            </div>
          </div>
        )}

        {/* Project bars */}
        <div className="space-y-3">
          {projects.map((project) => {
            const position = getBarPosition(project.startDate, project.endDate);
            const isOverdue = new Date(project.endDate) < today && project.currentPhase < project.totalPhases;
            const start = new Date(project.startDate);
            const end = new Date(project.endDate);
            
            return (
              <div key={project.id} className="flex items-center gap-4">
                {/* Project name */}
                <div className="w-32 flex-shrink-0">
                  <p className="text-sm font-medium truncate">{project.name}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {format(start, 'MMM d')} - {format(end, 'MMM d')}
                  </p>
                </div>
                
                {/* Timeline bar */}
                <div className="flex-1 relative h-8">
                  {/* Background bar (full duration) */}
                  <div 
                    className={`absolute h-6 top-1 rounded-md ${isOverdue ? 'bg-destructive/20' : 'bg-primary/20'}`}
                    style={{ left: position.left, width: position.width }}
                  >
                    {/* Progress bar */}
                    <div 
                      className={`h-full rounded-md ${isOverdue ? 'bg-destructive/60' : 'bg-primary'}`}
                      style={{ width: getProgressWidth(project) }}
                    />
                    
                    {/* Phase indicator */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-[10px] font-medium text-primary-foreground drop-shadow-sm">
                        Phase {project.currentPhase}/{project.totalPhases}
                      </span>
                    </div>
                  </div>
                  
                  {/* End date marker */}
                  <div 
                    className="absolute top-0 h-8 flex items-center"
                    style={{ left: `calc(${position.left} + ${position.width})` }}
                  >
                    <div className={`w-1.5 h-1.5 rounded-full ${isOverdue ? 'bg-destructive' : 'bg-primary'}`} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
