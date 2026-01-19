import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface Project {
  currentPhase: number;
  totalPhases: number;
}

interface Props {
  projects: Project[];
}

export const ProgressTrendChart = ({ projects }: Props) => {
  // Simulate weekly progress data (in a real app, this would come from the database)
  const weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'];
  
  // Calculate current average progress
  const currentAvg = projects.length > 0 
    ? Math.round(projects.reduce((acc, p) => acc + (p.currentPhase / p.totalPhases) * 100, 0) / projects.length)
    : 0;
  
  // Generate realistic trend data leading to current progress
  const data = weeks.map((week, index) => {
    const factor = (index + 1) / weeks.length;
    const variation = Math.sin(index * 0.8) * 5; // Add some natural variation
    const progress = Math.min(100, Math.max(0, Math.round(currentAvg * factor + variation)));
    
    return {
      week,
      progress,
      target: Math.round(((index + 1) / weeks.length) * 100) // Linear target
    };
  });

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="font-medium text-foreground">{label}</p>
          <p className="text-sm text-primary">Progress: {payload[0].value}%</p>
          <p className="text-sm text-muted-foreground">Target: {payload[1]?.value}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-card rounded-2xl p-6">
      <h3 className="text-lg font-display font-bold mb-4">Progress Trend</h3>
      <p className="text-sm text-muted-foreground mb-4">Average completion over time</p>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="progressGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis 
              dataKey="week" 
              axisLine={false} 
              tickLine={false}
              tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
            />
            <YAxis 
              axisLine={false} 
              tickLine={false}
              tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area 
              type="monotone" 
              dataKey="progress" 
              stroke="hsl(var(--primary))" 
              strokeWidth={2}
              fill="url(#progressGradient)" 
            />
            <Area 
              type="monotone" 
              dataKey="target" 
              stroke="hsl(var(--muted-foreground))" 
              strokeWidth={1}
              strokeDasharray="4 4"
              fill="transparent" 
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center gap-4 mt-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <div className="w-3 h-1 rounded-full bg-primary" />
          <span>Actual Progress</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 border-t border-dashed border-muted-foreground" />
          <span>Target</span>
        </div>
      </div>
    </div>
  );
};
