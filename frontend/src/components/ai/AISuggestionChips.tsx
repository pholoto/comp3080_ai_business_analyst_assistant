import { HelpCircle } from "lucide-react";

interface AISuggestionChipsProps {
  suggestions: string[];
  onSelect?: (suggestion: string) => void;
}

export const AISuggestionChips = ({ suggestions, onSelect }: AISuggestionChipsProps) => {
  if (!suggestions.length) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <HelpCircle className="w-4 h-4" />
        <span>Questions to consider:</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSelect?.(suggestion)}
            className="px-3 py-1.5 text-sm rounded-full bg-secondary hover:bg-secondary/80 
                     text-secondary-foreground border border-border/50 
                     transition-all duration-200 hover:border-primary/30 hover:shadow-sm"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};
