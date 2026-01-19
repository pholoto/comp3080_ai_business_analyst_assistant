export interface AIFeedbackItem {
  id: string;
  type: 'suggestion' | 'warning' | 'insight' | 'question' | 'rewrite' | 'diagnosis' | 'recommendation';
  title: string;
  content: string;
  confidence: number;
  category: string;
  actions?: { label: string; action: string }[];
  items?: string[];
}

// Phase 1 Specific Interfaces
export interface NormalizedProblemSummary {
  rewrittenProblem: string;
  purpose: string;
}

export interface UserPersona {
  role: string;
  context: string;
  goal: string;
  urgency: string;
  failureImpact: string;
}

export interface SecondaryPersona {
  role: string;
  reason: string;
}

export interface UserPersonaClarification {
  primaryUser: UserPersona;
  secondaryPersonas: SecondaryPersona[];
}

export interface PainMoment {
  id: string;
  moment: string;
  trigger: string;
  currentBehavior: string;
  whyItHurts: string;
}

export interface RootCause {
  id: string;
  category: 'Knowledge' | 'Process' | 'Access' | 'Psychology';
  cause: string;
  description: string;
}

export interface ExistingSolution {
  name: string;
  gap: string;
}

export interface ProblemImpact {
  time: string;
  quality: string;
  emotional: string;
  opportunity: string;
}

export interface RiskFlag {
  id: string;
  type: 'unclear' | 'validate';
  description: string;
}

export interface Phase1DetailedResult {
  overallScore: number;
  breakdown: { category: string; score: number; color: string }[];
  normalizedProblem: NormalizedProblemSummary;
  personaClarification: UserPersonaClarification;
  painMoments: PainMoment[];
  rootCauses: RootCause[];
  existingSolutions: ExistingSolution[];
  problemImpact: ProblemImpact;
  riskFlags: RiskFlag[];
  transitionQuestions: string[];
}

// Phase 2 Specific Interfaces
export interface NormalizedFeature {
  id: string;
  original: string;
  normalized: string;
  category: 'core' | 'enhancement' | 'nice-to-have';
  duplicateOf?: string;
}

export interface RequirementRow {
  id: string;
  requirement: string;
  type: 'functional' | 'non-functional';
  category: string;
  moscow: 'Must' | 'Should' | 'Could' | 'Won\'t';
  rationale: string;
  complexity: 'Low' | 'Medium' | 'High';
}

export interface Phase2FeatureResult {
  normalizedFeatures: NormalizedFeature[];
  requirements: RequirementRow[];
  mvpScope: string[];
  warnings: string[];
}

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

export interface Phase2DetailedResult {
  overallScore: number;
  breakdown: { category: string; score: number; color: string }[];
  featureAnalysis: Phase2FeatureResult;
  userJourney: Phase2JourneyResult;
}

export interface AIAnalysisResult {
  overallScore: number;
  breakdown: { category: string; score: number; color: string }[];
  feedback: AIFeedbackItem[];
  suggestedQuestions: string[];
  phase1Details?: Phase1DetailedResult;
  phase2Details?: Phase2DetailedResult;
}

// Detailed Phase 1 mock response (BA-style critical analysis)
export const phase1DetailedResponse: Phase1DetailedResult = {
  overallScore: 62,
  breakdown: [
    { category: "Problem Clarity", score: 55, color: "hsl(var(--destructive))" },
    { category: "User Specificity", score: 48, color: "hsl(25 95% 53%)" },
    { category: "Pain Point Validity", score: 72, color: "hsl(142 76% 36%)" },
    { category: "Scope Definition", score: 65, color: "hsl(var(--primary))" },
    { category: "Market Readiness", score: 70, color: "hsl(262 83% 58%)" },
  ],

  // Section 1: Normalized Problem Summary (AI-Rewritten)
  normalizedProblem: {
    purpose: "Eliminate solution bias, buzzwords, and ambiguity.",
    rewrittenProblem: "High school students applying for universities struggle to produce structured, authentic personal statements and CVs without access to experienced mentors. They often feel overwhelmed at the starting stage, uncertain about expectations, and unable to evaluate the quality of their drafts, leading to wasted time, stress, and suboptimal applications."
  },

  // Section 2: User Persona Clarification (Forced Choice)
  personaClarification: {
    primaryUser: {
      role: "High school students preparing university applications independently",
      context: "Writing essays and CVs without 1-1 mentoring",
      goal: "Produce application-ready documents with confidence",
      urgency: "High (fixed deadlines, one-shot submissions)",
      failureImpact: "Missed admissions or weaker scholarship chances"
    },
    secondaryPersonas: [
      { role: "Parents assisting their children", reason: "Excluded from MVP - different needs, less urgency" },
      { role: "School counselors", reason: "Excluded from MVP - B2B sales cycle complexity" },
      { role: "Gap year students reapplying", reason: "Lower priority - smaller segment" }
    ]
  },

  // Section 3: Concrete Pain Moments (Not Pain Points)
  painMoments: [
    {
      id: "pm-1",
      moment: "Starting paralysis",
      trigger: "Student opens a blank Google Doc to write their personal statement",
      currentBehavior: "Spends 30-60 minutes rereading prompts, opens YouTube/Reddit, then closes the document without writing a single sentence",
      whyItHurts: "Time is consumed without progress, increasing anxiety and procrastination cycle"
    },
    {
      id: "pm-2",
      moment: "Feedback void",
      trigger: "Student finishes first draft and wants to know if it's good",
      currentBehavior: "Sends to friends who say \"looks good\" or parents who nitpick grammar, neither providing substantive feedback",
      whyItHurts: "False confidence or frustration, no actual improvement in essay quality"
    },
    {
      id: "pm-3",
      moment: "Structure confusion",
      trigger: "Student reads 10 sample essays and notices they're all different",
      currentBehavior: "Tries to copy elements from multiple samples, creating a Frankenstein essay with no coherent narrative",
      whyItHurts: "Essay loses authenticity and personal voice, sounds generic to admissions officers"
    },
    {
      id: "pm-4",
      moment: "Deadline panic",
      trigger: "7 days left before submission, essay still feels \"not right\"",
      currentBehavior: "Rewrites entire essay from scratch, stays up late, submits something barely reviewed",
      whyItHurts: "Quality suffers under pressure, emotional toll affects other applications"
    },
    {
      id: "pm-5",
      moment: "CV overwhelm",
      trigger: "Student needs to list achievements but feels they have \"nothing impressive\"",
      currentBehavior: "Either leaves sections blank or inflates minor activities with buzzwords",
      whyItHurts: "CV appears either empty or inauthentic, neither representing the student well"
    }
  ],

  // Section 4: Root Cause Analysis
  rootCauses: [
    {
      id: "rc-1",
      category: "Knowledge",
      cause: "Knowledge Gap",
      description: "Students don't understand evaluation criteria—what makes an essay \"good\" vs \"great\" is opaque"
    },
    {
      id: "rc-2",
      category: "Process",
      cause: "Process Gap",
      description: "No clear writing stages (brainstorm → structure → refine). Students jump straight to drafting without scaffolding"
    },
    {
      id: "rc-3",
      category: "Access",
      cause: "Access Gap",
      description: "Lack of affordable, timely mentorship. Good counselors cost $200+/hour or have month-long waitlists"
    },
    {
      id: "rc-4",
      category: "Psychology",
      cause: "Cognitive Load",
      description: "Too many examples/templates without guidance creates paradox of choice and analysis paralysis"
    }
  ],

  // Section 5: Existing Solutions & Why They Fail
  existingSolutions: [
    {
      name: "ESAI",
      gap: "Provides automated feedback but lacks context awareness and application strategy"
    },
    {
      name: "ChatGPT",
      gap: "Flexible but requires strong prompting skills and offers inconsistent guidance"
    },
    {
      name: "Paid Counselors",
      gap: "High quality but inaccessible due to cost ($150-500/hour) and availability"
    },
    {
      name: "Sample Essay Databases",
      gap: "Useful for inspiration but no personalized guidance on how to adapt"
    }
  ],

  // Section 6: Problem Impact & Stakes
  problemImpact: {
    time: "10-20 hours lost per application due to rewrites and uncertainty",
    quality: "Essays remain generic and uncompetitive without expert feedback",
    emotional: "High stress close to deadlines, affecting sleep and other coursework",
    opportunity: "Reduced admission and scholarship probability—potentially $10k-100k lifetime impact"
  },

  // Section 7: Readiness & Risk Flags
  riskFlags: [
    { id: "rf-1", type: "unclear", description: "Students may still over-rely on AI instead of developing their own thinking" },
    { id: "rf-2", type: "unclear", description: "Essay quality metrics are subjective—how will you measure success?" },
    { id: "rf-3", type: "validate", description: "Users may expect \"done-for-you\" writing, not guidance" },
    { id: "rf-4", type: "validate", description: "Willingness to pay: Do students/parents see this as a $10 or $100 problem?" },
    { id: "rf-5", type: "unclear", description: "Competitive advantage: What stops ChatGPT from being \"good enough\"?" }
  ],

  // Section 8: Transition Questions to Phase 2
  transitionQuestions: [
    "What signals tell a student their essay is \"good enough\"?",
    "Where do students abandon the process most often?",
    "What feedback type causes the biggest revision improvement?",
    "Which essay sections are hardest to write (intro, body, conclusion)?",
    "How do students currently track progress across multiple applications?",
    "What triggers the decision to seek external help?"
  ]
};

// Detailed Phase 2 mock response
export const phase2DetailedResponse: Phase2DetailedResult = {
  overallScore: 72,
  breakdown: [
    { category: "Feature Clarity", score: 68, color: "hsl(var(--primary))" },
    { category: "Scope Feasibility", score: 75, color: "hsl(142 76% 36%)" },
    { category: "User-Centricity", score: 80, color: "hsl(262 83% 58%)" },
    { category: "Technical Viability", score: 65, color: "hsl(25 95% 53%)" },
  ],
  featureAnalysis: {
    normalizedFeatures: [
      { id: "f1", original: "AI essay feedback", normalized: "Get instant, actionable feedback on essay drafts", category: "core" },
      { id: "f2", original: "Essay checker", normalized: "Get instant, actionable feedback on essay drafts", category: "core", duplicateOf: "f1" },
      { id: "f3", original: "Writing prompts generator", normalized: "Receive personalized brainstorming prompts to overcome writer's block", category: "core" },
      { id: "f4", original: "Sample essays library", normalized: "Browse curated examples with annotations explaining what works", category: "enhancement" },
      { id: "f5", original: "Progress tracker", normalized: "Track completion status across all applications in one dashboard", category: "core" },
      { id: "f6", original: "CV builder", normalized: "Build achievement-focused CV with guided prompts", category: "core" },
      { id: "f7", original: "University deadline reminders", normalized: "Receive proactive deadline alerts based on target schools", category: "enhancement" },
      { id: "f8", original: "Dark mode", normalized: "Switch between light and dark themes for comfort", category: "nice-to-have" },
      { id: "f9", original: "Share with counselor", normalized: "Invite advisors to review and comment on drafts", category: "enhancement" },
    ],
    requirements: [
      { id: "r1", requirement: "User can submit essay draft and receive structured AI feedback within 30 seconds", type: "functional", category: "AI/ML", moscow: "Must", rationale: "Core value proposition - without this, the product has no differentiation", complexity: "High" },
      { id: "r2", requirement: "System provides feedback on structure, clarity, authenticity, and grammar separately", type: "functional", category: "AI/ML", moscow: "Must", rationale: "Granular feedback helps users prioritize revisions", complexity: "High" },
      { id: "r3", requirement: "User can create account using email or Google OAuth", type: "functional", category: "Authentication", moscow: "Must", rationale: "Required for saving progress and personalization", complexity: "Low" },
      { id: "r4", requirement: "User can save multiple essay drafts with version history", type: "functional", category: "Data", moscow: "Must", rationale: "Students iterate many times; losing work is catastrophic", complexity: "Medium" },
      { id: "r5", requirement: "System suggests 3-5 brainstorming prompts based on user profile", type: "functional", category: "AI/ML", moscow: "Should", rationale: "Addresses 'starting paralysis' pain point", complexity: "Medium" },
      { id: "r6", requirement: "Dashboard shows progress across all applications with deadlines", type: "functional", category: "User Interface", moscow: "Should", rationale: "Reduces cognitive load of tracking multiple schools", complexity: "Medium" },
      { id: "r7", requirement: "CV builder with section-by-section guidance", type: "functional", category: "User Interface", moscow: "Should", rationale: "Bundled value; reuses AI capabilities", complexity: "Medium" },
      { id: "r8", requirement: "API response time < 2 seconds for non-AI operations", type: "non-functional", category: "Performance", moscow: "Must", rationale: "User experience degrades significantly beyond 2s", complexity: "Low" },
      { id: "r9", requirement: "AI feedback response time < 30 seconds", type: "non-functional", category: "Performance", moscow: "Must", rationale: "Users expect near-instant for short essays", complexity: "Medium" },
      { id: "r10", requirement: "System handles 1000 concurrent users", type: "non-functional", category: "Performance", moscow: "Could", rationale: "MVP unlikely to hit this; nice for scaling", complexity: "High" },
      { id: "r11", requirement: "All user data encrypted at rest and in transit", type: "non-functional", category: "Security", moscow: "Must", rationale: "Essays contain sensitive personal information", complexity: "Low" },
      { id: "r12", requirement: "WCAG 2.1 AA compliance for accessibility", type: "non-functional", category: "User Interface", moscow: "Should", rationale: "Inclusive design for students with disabilities", complexity: "Medium" },
    ],
    mvpScope: [
      "AI essay feedback with structured response (structure, clarity, authenticity)",
      "User authentication (email + Google OAuth)",
      "Essay draft saving with basic version history",
      "Single dashboard showing all essays and status",
      "Basic brainstorming prompt generator"
    ],
    warnings: [
      "AI feedback quality heavily depends on prompt engineering—allocate time for iteration",
      "Version history adds complexity; consider simplified 'last 5 versions' for MVP",
      "Google OAuth requires verification if accessing sensitive scopes",
      "'30 second response' may not be achievable for long essays without async processing"
    ]
  },
  userJourney: {
    featureName: "AI Essay Feedback",
    happyPathSummary: "User pastes essay → Clicks 'Analyze' → Receives structured feedback in 4 categories → Applies suggestions → Resubmits for improved score",
    journeySteps: [
      {
        id: "j1",
        stepNumber: 1,
        action: "Land on essay input page",
        userGoal: "Start getting feedback on my essay",
        systemResponse: "Display clean input area with clear CTA and character count",
        uiElement: "Full-screen textarea with 'Paste your essay here' placeholder, character counter, and prominent 'Analyze Essay' button",
        emotionalState: "neutral",
        device: "both",
        notes: "Keep interface minimal to reduce cognitive load for anxious users"
      },
      {
        id: "j2",
        stepNumber: 2,
        action: "Paste or type essay content",
        userGoal: "Input my draft quickly without friction",
        systemResponse: "Accept text, update character count, enable analyze button when minimum length reached",
        uiElement: "Textarea expands as content grows, subtle validation message if too short",
        emotionalState: "neutral",
        device: "both",
        notes: "Support Ctrl+V paste and drag-drop for .docx files",
        alternatives: ["Upload .docx file", "Connect Google Docs"]
      },
      {
        id: "j3",
        stepNumber: 3,
        action: "Click 'Analyze Essay' button",
        userGoal: "Get AI feedback as quickly as possible",
        systemResponse: "Show loading state with progress indicator and estimated time",
        uiElement: "Button transforms to loading spinner, progress bar appears with '~20 seconds remaining'",
        emotionalState: "neutral",
        device: "both",
        notes: "Use skeleton loaders for feedback sections to set expectations"
      },
      {
        id: "j4",
        stepNumber: 4,
        action: "View feedback results",
        userGoal: "Understand what's good and what needs improvement",
        systemResponse: "Display structured feedback with scores and specific suggestions",
        uiElement: "Four feedback cards (Structure, Clarity, Authenticity, Grammar) each with score, summary, and expandable details",
        emotionalState: "positive",
        device: "both",
        notes: "Lead with positive feedback to build confidence before critiques"
      },
      {
        id: "j5",
        stepNumber: 5,
        action: "Expand 'Authenticity' feedback section",
        userGoal: "Understand specific issues with my voice/tone",
        systemResponse: "Show detailed suggestions with highlighted essay excerpts",
        uiElement: "Accordion expands to show: issue description, quoted excerpt from essay, specific rewrite suggestion",
        emotionalState: "neutral",
        device: "desktop",
        notes: "Use side-by-side view on desktop: essay on left, feedback on right"
      },
      {
        id: "j6",
        stepNumber: 6,
        action: "Apply suggested edit",
        userGoal: "Improve my essay with one click",
        systemResponse: "Update essay text with suggestion, highlight change, add to revision history",
        uiElement: "'Apply' button next to each suggestion, applied text highlighted in green briefly",
        emotionalState: "positive",
        device: "both",
        notes: "Allow undo within 10 seconds of applying",
        alternatives: ["Manually edit instead", "Dismiss suggestion"]
      },
      {
        id: "j7",
        stepNumber: 7,
        action: "Click 'Re-analyze' after edits",
        userGoal: "See if my changes improved the score",
        systemResponse: "Run new analysis, show score comparison with previous version",
        uiElement: "Side-by-side score comparison: 'Structure: 72 → 85 (+13)', celebration animation if improved",
        emotionalState: "positive",
        device: "both",
        notes: "Gamification element—progress feels rewarding"
      },
      {
        id: "j8",
        stepNumber: 8,
        action: "Save final version",
        userGoal: "Keep this version safe for submission",
        systemResponse: "Save to user account, add to 'Ready for Review' section, prompt to add to specific university application",
        uiElement: "Success toast, essay card moves to 'Complete' column in dashboard",
        emotionalState: "positive",
        device: "both"
      }
    ],
    edgeCases: [
      "Essay is too short (<100 words) — show minimum length requirement with helpful message",
      "Essay is too long (>2000 words) — warn about common application limits, offer to truncate",
      "AI service timeout — show friendly error with 'Retry' button, save draft automatically",
      "User navigates away mid-analysis — save draft, show 'Continue analysis?' on return",
      "Network disconnects while typing — auto-save to local storage, sync when reconnected",
      "User submits non-English text — detect language and show appropriate feedback or limitation message"
    ],
    accessibilityNotes: [
      "All feedback scores must have text alternatives, not just color indicators",
      "Loading states announced via aria-live regions for screen readers",
      "Keyboard navigation through all feedback sections with clear focus indicators",
      "High contrast mode support for score visualizations",
      "Essay input textarea must support voice-to-text input"
    ]
  }
};

export const mockAIResponses: Record<number, AIAnalysisResult> = {
  1: {
    overallScore: 62,
    breakdown: phase1DetailedResponse.breakdown,
    feedback: [],
    suggestedQuestions: phase1DetailedResponse.transitionQuestions,
    phase1Details: phase1DetailedResponse
  },
  2: {
    overallScore: 72,
    breakdown: phase2DetailedResponse.breakdown,
    feedback: [],
    suggestedQuestions: [
      "What happens when users lose connection mid-analysis?",
      "How will you handle data conflicts between devices?",
      "What's the expected concurrent user load at peak?",
    ],
    phase2Details: phase2DetailedResponse
  },
  3: {
    overallScore: 85,
    breakdown: [
      { category: "Research Depth", score: 88, color: "hsl(var(--primary))" },
      { category: "Differentiation", score: 82, color: "hsl(142 76% 36%)" },
      { category: "Market Size", score: 85, color: "hsl(262 83% 58%)" },
      { category: "Competition", score: 84, color: "hsl(25 95% 53%)" },
    ],
    feedback: [
      {
        id: "3-1",
        type: "insight",
        title: "Market Gap Identified",
        content: "There's a clear gap in the market for AI-powered project management tailored to engineering students.",
        confidence: 91,
        category: "Opportunity",
      },
    ],
    suggestedQuestions: [
      "What's your pricing strategy vs competitors?",
      "How will you acquire first 100 users?",
    ],
  },
  4: {
    overallScore: 80,
    breakdown: [
      { category: "Scalability", score: 85, color: "hsl(var(--primary))" },
      { category: "Cost Efficiency", score: 78, color: "hsl(142 76% 36%)" },
      { category: "Compatibility", score: 82, color: "hsl(262 83% 58%)" },
      { category: "Maintenance", score: 75, color: "hsl(25 95% 53%)" },
    ],
    feedback: [
      {
        id: "4-1",
        type: "suggestion",
        title: "Consider Serverless Architecture",
        content: "For your use case, serverless functions could reduce costs by 40-60%.",
        confidence: 88,
        category: "Architecture",
      },
    ],
    suggestedQuestions: [
      "Do you need real-time features?",
      "What's your expected data volume?",
    ],
  },
  5: {
    overallScore: 74,
    breakdown: [
      { category: "Timeline", score: 70, color: "hsl(var(--primary))" },
      { category: "Resource Allocation", score: 75, color: "hsl(142 76% 36%)" },
      { category: "Risk Assessment", score: 78, color: "hsl(262 83% 58%)" },
      { category: "Dependencies", score: 72, color: "hsl(25 95% 53%)" },
    ],
    feedback: [
      {
        id: "5-1",
        type: "warning",
        title: "Timeline May Be Aggressive",
        content: "Based on similar projects, your timeline appears 20-30% shorter than typical.",
        confidence: 85,
        category: "Planning",
      },
    ],
    suggestedQuestions: [
      "What's your biggest technical risk?",
      "Do you have backup resources?",
    ],
  },
  6: {
    overallScore: 82,
    breakdown: [
      { category: "Test Coverage", score: 85, color: "hsl(var(--primary))" },
      { category: "Edge Cases", score: 78, color: "hsl(142 76% 36%)" },
      { category: "Performance", score: 80, color: "hsl(262 83% 58%)" },
      { category: "Security", score: 84, color: "hsl(25 95% 53%)" },
    ],
    feedback: [
      {
        id: "6-1",
        type: "suggestion",
        title: "Add Integration Tests",
        content: "Your test plan focuses on unit tests. Consider adding integration tests.",
        confidence: 88,
        category: "Testing Strategy",
      },
    ],
    suggestedQuestions: [
      "What's your acceptable test coverage %?",
      "How will you test mobile?",
    ],
  },
  7: {
    overallScore: 88,
    breakdown: [
      { category: "Completeness", score: 90, color: "hsl(var(--primary))" },
      { category: "Clarity", score: 88, color: "hsl(142 76% 36%)" },
      { category: "Structure", score: 85, color: "hsl(262 83% 58%)" },
      { category: "Accessibility", score: 89, color: "hsl(25 95% 53%)" },
    ],
    feedback: [
      {
        id: "7-1",
        type: "suggestion",
        title: "Add API Documentation",
        content: "Include OpenAPI/Swagger documentation for your backend endpoints.",
        confidence: 94,
        category: "Documentation",
      },
    ],
    suggestedQuestions: [
      "Who is the primary audience?",
      "How will you keep docs updated?",
    ],
  },
};
