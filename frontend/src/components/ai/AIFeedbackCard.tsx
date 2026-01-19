import { useState } from "react";
import { 
  Lightbulb, AlertTriangle, HelpCircle, TrendingUp,
  ChevronDown, ChevronUp, Check, X
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AIFeedbackItem } from "@/lib/mockAIData";

interface AIFeedbackCardProps {
  feedback: AIFeedbackItem;
  onAction?: (action: string) => void;
  onDismiss?: () => void;
}

export const AIFeedbackCard = ({ feedback, onAction, onDismiss }: AIFeedbackCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getIcon = () => {
    switch (feedback.type) {
      case "suggestion":
        return <Lightbulb className="w-5 h-5" />;
      case "warning":
        return <AlertTriangle className="w-5 h-5" />;
      case "question":
        return <HelpCircle className="w-5 h-5" />;
      case "insight":
        return <TrendingUp className="w-5 h-5" />;
      default:
        return <Lightbulb className="w-5 h-5" />;
    }
  };

  const getTypeStyles = () => {
    switch (feedback.type) {
      case "suggestion":
        return "bg-primary/10 text-primary border-primary/20";
      case "warning":
        return "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/20";
      case "question":
        return "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20";
      case "insight":
        return "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20";
      default:
        return "bg-primary/10 text-primary border-primary/20";
    }
  };

  const getConfidenceColor = () => {
    if (feedback.confidence >= 90) return "bg-green-500";
    if (feedback.confidence >= 75) return "bg-primary";
    if (feedback.confidence >= 60) return "bg-yellow-500";
    return "bg-orange-500";
  };

  return (
    <div className={`rounded-xl border p-4 transition-all duration-200 ${getTypeStyles()}`}>
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
          feedback.type === "warning" ? "bg-yellow-500/20" :
          feedback.type === "question" ? "bg-purple-500/20" :
          feedback.type === "insight" ? "bg-green-500/20" :
          "bg-primary/20"
        }`}>
          {getIcon()}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div>
              <h4 className="font-semibold text-foreground">{feedback.title}</h4>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="outline" className="text-xs capitalize">
                  {feedback.category}
                </Badge>
                <div className="flex items-center gap-1">
                  <div className={`w-1.5 h-1.5 rounded-full ${getConfidenceColor()}`} />
                  <span className="text-xs text-muted-foreground">
                    {feedback.confidence}% confidence
                  </span>
                </div>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </div>

          {/* Expandable content */}
          <div className={`overflow-hidden transition-all duration-200 ${isExpanded ? "mt-3 max-h-96" : "max-h-0"}`}>
            <p className="text-sm text-foreground/80 leading-relaxed">
              {feedback.content}
            </p>
            
            {/* Actions */}
            {feedback.actions && feedback.actions.length > 0 && (
              <div className="flex items-center gap-2 mt-4">
                {feedback.actions.map((action) => (
                  <Button
                    key={action.action}
                    variant="secondary"
                    size="sm"
                    className="gap-1.5"
                    onClick={() => onAction?.(action.action)}
                  >
                    <Check className="w-3.5 h-3.5" />
                    {action.label}
                  </Button>
                ))}
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-1.5 text-muted-foreground"
                  onClick={onDismiss}
                >
                  <X className="w-3.5 h-3.5" />
                  Dismiss
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
