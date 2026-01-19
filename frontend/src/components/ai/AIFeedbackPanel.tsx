import { useState } from "react";
import { Sparkles, RefreshCw, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AIAnalysisResult } from "@/lib/mockAIData";
import { AILoadingSkeleton } from "./AILoadingSkeleton";
import { AIQualityScore } from "./AIQualityScore";
import { AIFeedbackCard } from "./AIFeedbackCard";
import { AISuggestionChips } from "./AISuggestionChips";
import { AITypingAnimation } from "./AITypingAnimation";
import { AIPhase1DetailedAnalysis } from "./AIPhase1DetailedAnalysis";
import { AIPhase2FeatureAnalysis } from "./AIPhase2FeatureAnalysis";
import { AIPhase2UserJourney } from "./AIPhase2UserJourney";

interface AIFeedbackPanelProps {
  isLoading: boolean;
  result: AIAnalysisResult | null;
  onAnalyze: () => void;
  onRetry?: () => void;
  title?: string;
  description?: string;
  disabled?: boolean;
  phaseId?: number;
  subFunction?: 'feature-analyzer' | 'user-journey' | 'tech-stack' | 'wireframe';
}

export const AIFeedbackPanel = ({
  isLoading,
  result,
  onAnalyze,
  onRetry,
  title = "AI Analysis",
  description = "Get intelligent insights on your work",
  disabled = false,
  phaseId = 1,
  subFunction,
}: AIFeedbackPanelProps) => {
  const [showTyping, setShowTyping] = useState(true);
  const [dismissedFeedback, setDismissedFeedback] = useState<string[]>([]);

  const handleDismiss = (id: string) => {
    setDismissedFeedback([...dismissedFeedback, id]);
  };

  const handleAction = (action: string) => {
    console.log("Action triggered:", action);
  };

  const visibleFeedback = result?.feedback.filter(f => !dismissedFeedback.includes(f.id)) || [];

  // Check if this is Phase 1 with detailed analysis
  const hasPhase1Details = phaseId === 1 && result?.phase1Details;
  const hasPhase2Details = phaseId === 2 && result?.phase2Details;
  const isPhase3 = phaseId === 3;
  const isPhase4 = phaseId === 4;

  const getTypingText = () => {
    if (hasPhase1Details) {
      return "I've performed a critical BA-style analysis of your problem definition. Here's my detailed assessment with rewrites, diagnosis, and strategic recommendations:";
    }
    if (hasPhase2Details && subFunction === 'feature-analyzer') {
      return "I've normalized your feature list, categorized requirements using MoSCoW prioritization, and identified potential scope risks:";
    }
    if (hasPhase2Details && subFunction === 'user-journey') {
      return "I've mapped out the complete user journey with emotional states, UI elements, and edge cases to consider:";
    }
    return "I've analyzed your input and found some insights. Here's my assessment:";
  };

  return (
    <div className="glass-card rounded-xl p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center relative">
            <Bot className="w-6 h-6 text-primary" />
            {isLoading && (
              <span className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-primary animate-pulse" />
            )}
          </div>
          <div>
            <h3 className="font-display font-bold text-lg">{title}</h3>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {result && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onRetry || onAnalyze}
              className="gap-1.5"
            >
              <RefreshCw className="w-4 h-4" />
              Re-analyze
            </Button>
          )}
          {!result && (
            <Button
              onClick={onAnalyze}
              className="aiba-button-primary gap-2"
              disabled={isLoading || disabled}
            >
              {isLoading ? (
                <>Analyzing...</>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Analyze
                </>
              )}
            </Button>
          )}
        </div>
      </div>

      {/* Loading State */}
      {isLoading && <AILoadingSkeleton variant="panel" />}

      {/* Results */}
      {!isLoading && result && (
        <div className="space-y-6 animate-fade-in">
          {/* Intro message with typing effect */}
          {showTyping && (
            <div className="flex items-start gap-3 p-4 rounded-xl bg-secondary/50">
              <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-4 h-4 text-primary" />
              </div>
              <div className="text-sm text-foreground/80">
                <AITypingAnimation
                  text={getTypingText()}
                  speed={12}
                  onComplete={() => setShowTyping(false)}
                />
              </div>
            </div>
          )}

          {/* Phase 1 Detailed Analysis */}
          {hasPhase1Details && result.phase1Details && (
            <AIPhase1DetailedAnalysis result={result.phase1Details} />
          )}

          {/* Phase 2 Feature Analysis */}
          {hasPhase2Details && subFunction === 'feature-analyzer' && result.phase2Details && (
            <AIPhase2FeatureAnalysis result={result.phase2Details.featureAnalysis} />
          )}

          {/* Phase 2 User Journey */}
          {hasPhase2Details && subFunction === 'user-journey' && result.phase2Details && (
            <AIPhase2UserJourney result={result.phase2Details.userJourney} />
          )}

          {/* Generic feedback for other phases (excluding Phase 3 and Phase 4 which render results separately) */}
          {!hasPhase1Details && !hasPhase2Details && !isPhase3 && !isPhase4 && (
            <>
              {/* Quality Score */}
              <AIQualityScore
                score={result.overallScore}
                breakdown={result.breakdown}
                animated={true}
              />

              {/* Feedback Cards */}
              {visibleFeedback.length > 0 && (
                <div className="space-y-3">
                  <h4 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                    Feedback ({visibleFeedback.length})
                  </h4>
                  {visibleFeedback.map((feedback) => (
                    <AIFeedbackCard
                      key={feedback.id}
                      feedback={feedback}
                      onAction={handleAction}
                      onDismiss={() => handleDismiss(feedback.id)}
                    />
                  ))}
                </div>
              )}

              {/* Suggested Questions */}
              <AISuggestionChips
                suggestions={result.suggestedQuestions}
                onSelect={(q) => console.log("Selected question:", q)}
              />
            </>
          )}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !result && (
        <div className="text-center py-8">
          <div className="w-16 h-16 mx-auto rounded-full bg-secondary flex items-center justify-center mb-4">
            <Sparkles className="w-8 h-8 text-muted-foreground" />
          </div>
          <p className="text-muted-foreground">
            Click "Analyze" to get AI-powered feedback on your work
          </p>
        </div>
      )}
    </div>
  );
};
