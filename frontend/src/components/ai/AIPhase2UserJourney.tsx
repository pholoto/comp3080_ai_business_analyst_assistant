import { useState } from "react";
import { 
  User, MousePointer, Eye, CheckCircle2, ArrowRight, 
  AlertCircle, Lightbulb, Copy, Check, Smartphone, Monitor,
  MessageSquare, Zap, Target, Clock
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export interface JourneyStep {
  id: string;
  stepNumber: number;
  action: string;
  userGoal: string;
  systemResponse: string;
  uiElement: string;
  emotionalState: 'positive' | 'neutral' | 'negative' | 'frustrated';
  device: 'desktop' | 'mobile' | 'both';
  notes?: string;
  alternatives?: string[];
}

export interface Phase2JourneyResult {
  featureName: string;
  journeySteps: JourneyStep[];
  happyPathSummary: string;
  edgeCases: string[];
  accessibilityNotes: string[];
}

interface AIPhase2UserJourneyProps {
  result: Phase2JourneyResult;
}

const emotionColors: Record<string, { bg: string; border: string; icon: React.ReactNode }> = {
  positive: { 
    bg: 'bg-green-500/20', 
    border: 'border-green-500/30',
    icon: <CheckCircle2 className="w-4 h-4 text-green-400" />
  },
  neutral: { 
    bg: 'bg-muted', 
    border: 'border-border',
    icon: <Eye className="w-4 h-4 text-muted-foreground" />
  },
  negative: { 
    bg: 'bg-orange-500/20', 
    border: 'border-orange-500/30',
    icon: <AlertCircle className="w-4 h-4 text-orange-400" />
  },
  frustrated: { 
    bg: 'bg-destructive/20', 
    border: 'border-destructive/30',
    icon: <AlertCircle className="w-4 h-4 text-destructive" />
  },
};

const deviceIcons: Record<string, React.ReactNode> = {
  desktop: <Monitor className="w-4 h-4" />,
  mobile: <Smartphone className="w-4 h-4" />,
  both: <><Monitor className="w-3 h-3" /><Smartphone className="w-3 h-3" /></>,
};

export const AIPhase2UserJourney = ({ result }: AIPhase2UserJourneyProps) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedStep, setExpandedStep] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const copyFullJourney = () => {
    const journeyText = result.journeySteps.map(step => 
      `Step ${step.stepNumber}: ${step.action}\nGoal: ${step.userGoal}\nSystem: ${step.systemResponse}\nUI: ${step.uiElement}`
    ).join('\n\n');
    copyToClipboard(journeyText, 'full-journey');
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-purple-500 flex items-center justify-center">
            <Target className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <h3 className="font-display font-bold text-lg">User Journey: {result.featureName}</h3>
            <p className="text-sm text-muted-foreground">{result.journeySteps.length} steps in happy path</p>
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={copyFullJourney} className="gap-2">
          {copiedId === 'full-journey' ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
          Copy Flow
        </Button>
      </div>

      {/* Happy Path Summary */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-primary/10 via-background to-purple-500/10 border border-primary/20">
        <div className="flex items-start gap-3">
          <Zap className="w-5 h-5 text-primary mt-0.5" />
          <div>
            <h4 className="font-semibold text-sm mb-1">Happy Path Summary</h4>
            <p className="text-sm text-muted-foreground">{result.happyPathSummary}</p>
          </div>
        </div>
      </div>

      {/* Journey Flow Diagram */}
      <div className="relative">
        {/* Timeline connector */}
        <div className="absolute left-6 top-12 bottom-12 w-0.5 bg-gradient-to-b from-primary via-purple-500 to-primary/30" />

        <div className="space-y-6">
          {result.journeySteps.map((step, index) => {
            const emotion = emotionColors[step.emotionalState];
            const isExpanded = expandedStep === step.id;
            const isLast = index === result.journeySteps.length - 1;

            return (
              <div key={step.id} className="relative">
                {/* Step Node */}
                <div 
                  className={`ml-12 p-4 rounded-xl border transition-all cursor-pointer hover:shadow-lg ${emotion.bg} ${emotion.border}`}
                  onClick={() => setExpandedStep(isExpanded ? null : step.id)}
                >
                  {/* Step Number Badge */}
                  <div className="absolute left-0 top-4 w-12 flex justify-center">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                      step.emotionalState === 'positive' ? 'bg-green-500 text-white' :
                      step.emotionalState === 'frustrated' ? 'bg-destructive text-white' :
                      'bg-primary text-primary-foreground'
                    }`}>
                      {step.stepNumber}
                    </div>
                  </div>

                  {/* Main Content */}
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 space-y-2">
                      {/* Action */}
                      <div className="flex items-center gap-2">
                        <MousePointer className="w-4 h-4 text-primary" />
                        <span className="font-semibold">{step.action}</span>
                      </div>

                      {/* User Goal */}
                      <div className="flex items-start gap-2 text-sm">
                        <User className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
                        <span className="text-muted-foreground"><strong>Goal:</strong> {step.userGoal}</span>
                      </div>

                      {/* System Response */}
                      <div className="flex items-start gap-2 text-sm">
                        <MessageSquare className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
                        <span className="text-muted-foreground"><strong>System:</strong> {step.systemResponse}</span>
                      </div>
                    </div>

                    {/* Right Side Badges */}
                    <div className="flex flex-col items-end gap-2">
                      <Badge variant="outline" className="flex items-center gap-1.5">
                        {deviceIcons[step.device]}
                        {step.device}
                      </Badge>
                      <div className={`p-1.5 rounded-lg ${emotion.bg}`}>
                        {emotion.icon}
                      </div>
                    </div>
                  </div>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div className="mt-4 pt-4 border-t border-border/50 space-y-3 animate-fade-in">
                      {/* UI Element */}
                      <div className="p-3 rounded-lg bg-background/50">
                        <div className="flex items-center gap-2 mb-1">
                          <Eye className="w-4 h-4 text-primary" />
                          <span className="text-sm font-medium">UI Element</span>
                        </div>
                        <p className="text-sm text-muted-foreground">{step.uiElement}</p>
                      </div>

                      {/* Notes */}
                      {step.notes && (
                        <div className="p-3 rounded-lg bg-primary/10">
                          <div className="flex items-center gap-2 mb-1">
                            <Lightbulb className="w-4 h-4 text-primary" />
                            <span className="text-sm font-medium">Design Notes</span>
                          </div>
                          <p className="text-sm text-muted-foreground">{step.notes}</p>
                        </div>
                      )}

                      {/* Alternatives */}
                      {step.alternatives && step.alternatives.length > 0 && (
                        <div className="space-y-2">
                          <span className="text-sm font-medium flex items-center gap-2">
                            <ArrowRight className="w-4 h-4" />
                            Alternative Paths
                          </span>
                          <div className="flex flex-wrap gap-2">
                            {step.alternatives.map((alt, i) => (
                              <Badge key={i} variant="secondary" className="text-xs">
                                {alt}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Arrow to next step */}
                {!isLast && (
                  <div className="absolute left-6 -bottom-3 transform -translate-x-1/2">
                    <ArrowRight className="w-4 h-4 text-primary rotate-90" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Edge Cases */}
      {result.edgeCases.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-orange-500/20 flex items-center justify-center">
              <AlertCircle className="w-4 h-4 text-orange-400" />
            </div>
            <div>
              <h3 className="font-display font-bold text-lg">Edge Cases to Handle</h3>
              <p className="text-sm text-muted-foreground">Scenarios outside the happy path</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {result.edgeCases.map((edge, index) => (
              <div key={index} className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/20 flex items-start gap-3">
                <AlertCircle className="w-4 h-4 text-orange-400 mt-0.5 shrink-0" />
                <span className="text-sm">{edge}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Accessibility Notes */}
      {result.accessibilityNotes.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
              <Eye className="w-4 h-4 text-purple-400" />
            </div>
            <div>
              <h3 className="font-display font-bold text-lg">Accessibility Considerations</h3>
              <p className="text-sm text-muted-foreground">Inclusive design requirements</p>
            </div>
          </div>

          <div className="space-y-2">
            {result.accessibilityNotes.map((note, index) => (
              <div key={index} className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-purple-400 mt-0.5 shrink-0" />
                <span className="text-sm">{note}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
