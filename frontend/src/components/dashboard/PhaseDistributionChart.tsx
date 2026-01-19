import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

interface Project {
  currentPhase: number;
}

interface Props {
  projects: Project[];
}

const PHASES = [
  { name: 'Problem Definition', phase: 1 },
  { name: 'Requirements', phase: 2 },
  { name: 'Market Analysis', phase: 3 },
  { name: 'Solution Design', phase: 4 },
  { name: 'Prototype', phase: 5 },
  { name: 'Testing', phase: 6 },
  { name: 'Documentation', phase: 7 },
];

const COLORS = [
  'hsl(var(--primary))',
  'hsl(var(--primary) / 0.85)',
  'hsl(var(--primary) / 0.7)',
  'hsl(var(--primary) / 0.55)',
  'hsl(var(--primary) / 0.4)',
  'hsl(var(--primary) / 0.25)',
  'hsl(var(--muted-foreground))',
];

export const PhaseDistributionChart = ({ projects }: Props) => {
  const phaseCount = PHASES.map(phase => ({
    name: phase.name,
    value: projects.filter(p => p.currentPhase === phase.phase).length,
    phase: phase.phase
  })).filter(p => p.value > 0);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="font-medium text-foreground">{payload[0].payload.name}</p>
          <p className="text-sm text-primary">{payload[0].value} project(s)</p>
        </div>
      );
    }
    return null;
  };

  const totalProjects = projects.length;

  return (
    <div className="glass-card rounded-2xl p-6">
      <h3 className="text-lg font-display font-bold mb-4">Phase Distribution</h3>
      <p className="text-sm text-muted-foreground mb-4">Projects by current phase</p>
      <div className="h-48 flex items-center justify-center">
        {totalProjects > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={phaseCount}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={70}
                paddingAngle={2}
                dataKey="value"
              >
                {phaseCount.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[entry.phase - 1]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-muted-foreground text-sm">No projects yet</p>
        )}
        {totalProjects > 0 && (
          <div className="absolute flex flex-col items-center">
            <span className="text-3xl font-display font-bold">{totalProjects}</span>
            <span className="text-xs text-muted-foreground">Projects</span>
          </div>
        )}
      </div>
      <div className="grid grid-cols-2 gap-2 mt-4">
        {phaseCount.slice(0, 4).map((phase, index) => (
          <div key={index} className="flex items-center gap-2 text-xs">
            <div 
              className="w-2 h-2 rounded-full" 
              style={{ backgroundColor: COLORS[phase.phase - 1] }}
            />
            <span className="text-muted-foreground truncate">{phase.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
