/**
 * Transformers to convert backend API responses to frontend expected format
 */

import type {
  Phase1Response,
  Phase2FeatureAnalyzerResponse,
  Phase2UserJourneyResponse,
  Phase3Response,
  PorterFiveForces,
  PorterForce,
  BackendFunctionalRequirement,
  BackendNonFunctionalRequirement
} from './api';

import type {
  AIAnalysisResult,
  ExistingSolution,
  JourneyStep,
  NormalizedFeature,
  NormalizedProblemSummary,
  PainMoment,
  Phase1DetailedResult,
  Phase2FeatureResult,
  Phase2JourneyResult,
  ProblemImpact,
  RequirementRow,
  RiskFlag,
  RootCause,
  SecondaryPersona,
  UserPersona,
  UserPersonaClarification
} from './mockAIData';

// ============================================================================
// Phase 1 Transformer
// ============================================================================

/**
 * Transform backend Phase 1 response to frontend format
 */
export function transformPhase1Response(response: Phase1Response): AIAnalysisResult {
  const phase1Details = transformToPhase1Details(response);

  return {
    overallScore: response.quality_score.overall_score,
    breakdown: response.quality_score.dimensions.map((dim, index) => ({
      category: dim.name,
      score: dim.score,
      color: getColorForIndex(index),
    })),
    feedback: [], // Feedback items can be derived from various sections if needed
    suggestedQuestions: response.transition_questions,
    phase1Details,
  };
}

function transformToPhase1Details(response: Phase1Response): Phase1DetailedResult {
  // Transform normalized problem
  const normalizedProblem: NormalizedProblemSummary = {
    rewrittenProblem: response.normalized_summary.summary,
    purpose: "Eliminate solution bias, buzzwords, and ambiguity.",
  };

  // Transform user personas
  const primaryUser: UserPersona = {
    role: response.personas.primary_user.role,
    context: response.personas.primary_user.context,
    goal: response.personas.primary_user.goal,
    urgency: response.personas.primary_user.urgency,
    failureImpact: response.personas.primary_user.failure_consequence,
  };

  const secondaryPersonas: SecondaryPersona[] = response.personas.secondary_users.map(user => ({
    role: user.role,
    reason: user.context || user.failure_consequence,
  }));

  const personaClarification: UserPersonaClarification = {
    primaryUser,
    secondaryPersonas,
  };

  // Transform pain moments
  const painMoments: PainMoment[] = response.pain_moments.map((pm, index) => ({
    id: `pm-${index + 1}`,
    moment: pm.moment,
    trigger: pm.trigger,
    currentBehavior: pm.current_behavior,
    whyItHurts: pm.why_it_hurts,
  }));

  // Transform root causes
  const rootCauses: RootCause[] = response.root_causes.map((rc, index) => ({
    id: `rc-${index + 1}`,
    category: mapRootCauseCategory(rc.category),
    cause: rc.cause,
    description: rc.explanation,
  }));

  // Transform existing solutions
  const existingSolutions: ExistingSolution[] = response.existing_solutions_analysis.map(sol => ({
    name: sol.name,
    gap: sol.gap,
  }));

  // Transform impact
  const problemImpact: ProblemImpact = transformImpactStakes(response.impact_stakes);

  // Transform risk flags from scope boundary and other sources
  const riskFlags: RiskFlag[] = response.scope_boundary.exclusions.map((exclusion, index) => ({
    id: `rf-${index + 1}`,
    type: 'unclear' as const,
    description: exclusion,
  }));

  return {
    overallScore: response.quality_score.overall_score,
    breakdown: response.quality_score.dimensions.map((dim, index) => ({
      category: dim.name,
      score: dim.score,
      color: getColorForIndex(index),
    })),
    normalizedProblem,
    personaClarification,
    painMoments,
    rootCauses,
    existingSolutions,
    problemImpact,
    riskFlags,
    transitionQuestions: response.transition_questions,
  };
}

function mapRootCauseCategory(category: string): 'Knowledge' | 'Process' | 'Access' | 'Psychology' {
  const categoryMap: Record<string, 'Knowledge' | 'Process' | 'Access' | 'Psychology'> = {
    'Knowledge': 'Knowledge',
    'Process': 'Process',
    'Access': 'Access',
    'Psychology': 'Psychology',
    'Technical': 'Process',
    'Other': 'Knowledge',
  };
  return categoryMap[category] || 'Knowledge';
}

function transformImpactStakes(stakes: Phase1Response['impact_stakes']): ProblemImpact {
  const impact: ProblemImpact = {
    time: '',
    quality: '',
    emotional: '',
    opportunity: '',
  };

  stakes.forEach(stake => {
    const category = stake.category.toLowerCase();
    const description = `${stake.description} (${stake.quantification})`;

    if (category.includes('time')) {
      impact.time = description;
    } else if (category.includes('academic') || category.includes('quality')) {
      impact.quality = description;
    } else if (category.includes('emotional')) {
      impact.emotional = description;
    } else if (category.includes('opportunity') || category.includes('financial')) {
      impact.opportunity = description;
    }
  });

  return impact;
}

// ============================================================================
// Phase 2 Transformers
// ============================================================================

/**
 * Transform backend Phase 2 Feature Analyzer response to frontend format
 */
export function transformPhase2FeatureResponse(response: Phase2FeatureAnalyzerResponse): AIAnalysisResult {
  const featureAnalysis = transformToFeatureAnalysis(response);

  // Calculate scores based on the analysis
  const scores = calculatePhase2FeatureScores(response);

  return {
    overallScore: scores.overall,
    breakdown: [
      { category: "Feature Clarity", score: scores.featureClarity, color: "hsl(var(--primary))" },
      { category: "Scope Feasibility", score: scores.scopeFeasibility, color: "hsl(142 76% 36%)" },
      { category: "User-Centricity", score: scores.userCentricity, color: "hsl(262 83% 58%)" },
      { category: "Technical Viability", score: scores.technicalViability, color: "hsl(25 95% 53%)" },
    ],
    feedback: [],
    suggestedQuestions: [
      "What happens when users lose connection mid-operation?",
      "How will you handle data conflicts between devices?",
      "What's the expected concurrent user load at peak?",
    ],
    phase2Details: {
      overallScore: scores.overall,
      breakdown: [
        { category: "Feature Clarity", score: scores.featureClarity, color: "hsl(var(--primary))" },
        { category: "Scope Feasibility", score: scores.scopeFeasibility, color: "hsl(142 76% 36%)" },
        { category: "User-Centricity", score: scores.userCentricity, color: "hsl(262 83% 58%)" },
        { category: "Technical Viability", score: scores.technicalViability, color: "hsl(25 95% 53%)" },
      ],
      featureAnalysis,
      userJourney: {
        featureName: "",
        journeySteps: [],
        happyPathSummary: "",
        edgeCases: [],
        accessibilityNotes: [],
      },
    },
  };
}

function transformToFeatureAnalysis(response: Phase2FeatureAnalyzerResponse): Phase2FeatureResult {
  // Transform normalized features
  const normalizedFeatures: NormalizedFeature[] = response.normalized_features.map((f, index) => ({
    id: `f-${index + 1}`,
    original: f.original_name,
    normalized: f.normalized_name,
    category: mapFeatureCategory(f.category),
    // duplication info not currently provided by backend
  }));

  // Map functional requirements
  const functionalReqs: RequirementRow[] = response.functional_requirements.map(r => ({
    id: r.id,
    requirement: `${r.name}: ${r.description}`,
    type: 'functional',
    category: r.category,
    moscow: mapMoscowPriority(r.moscow_priority),
    rationale: r.rationale,
    complexity: mapComplexity(r.complexity),
  }));

  // Map non-functional requirements
  const nonFunctionalReqs: RequirementRow[] = response.non_functional_requirements.map(r => ({
    id: r.id,
    requirement: `${r.attribute}: ${r.requirement}`,
    type: 'non-functional',
    category: r.category,
    moscow: mapMoscowPriority(r.moscow_priority),
    rationale: r.rationale,
    complexity: mapComplexity(r.complexity),
  }));

  const requirements: RequirementRow[] = [...functionalReqs, ...nonFunctionalReqs];

  // Transform MVP scope to list of strings
  const mvpScope: string[] = response.mvp_scope.included_features.map(
    f => `${f.feature_name} - ${f.justification}`
  );

  // Transform warnings
  const warnings: string[] = response.scope_warnings.map(
    w => `[${w.severity.toUpperCase()}] ${w.warning_type}: ${w.description}`
  );

  return {
    normalizedFeatures,
    requirements,
    mvpScope,
    warnings,
  };
}

function mapMoscowPriority(priority: string): 'Must' | 'Should' | 'Could' | 'Won\'t' {
  const p = priority.toLowerCase();
  if (p.includes('must')) return 'Must';
  if (p.includes('should')) return 'Should';
  if (p.includes('could')) return 'Could';
  if (p.includes('wont') || p.includes("won't")) return 'Won\'t';
  return 'Could';
}

function mapComplexity(complexity: string): 'Low' | 'Medium' | 'High' {
  const c = complexity.toLowerCase();
  if (c.includes('low')) return 'Low';
  if (c.includes('high')) return 'High';
  return 'Medium';
}

function mapFeatureCategory(category: string): 'core' | 'enhancement' | 'nice-to-have' {
  const categoryLower = category.toLowerCase();
  if (categoryLower.includes('core') || categoryLower.includes('must')) {
    return 'core';
  } else if (categoryLower.includes('enhance') || categoryLower.includes('should')) {
    return 'enhancement';
  }
  return 'nice-to-have';
}

function calculatePhase2FeatureScores(response: Phase2FeatureAnalyzerResponse) {
  // Feature clarity based on normalization (can be improved if backend adds quality metrics)
  const featureClarity = 85;

  // Calculate scope feasibility based on warnings
  const highSeverityWarnings = response.scope_warnings.filter(w => w.severity === 'high').length;
  const scopeFeasibility = Math.max(50, 100 - (highSeverityWarnings * 15));

  // Calculate user-centricity based on functional requirements ratio
  const functionalRatio = response.functional_requirements.length /
    Math.max(response.functional_requirements.length + response.non_functional_requirements.length, 1);
  const userCentricity = Math.round(functionalRatio * 100 * 0.3 + 70);

  // Calculate technical viability based on complexity distribution
  const allReqs: (BackendFunctionalRequirement | BackendNonFunctionalRequirement)[] = [
    ...response.functional_requirements,
    ...response.non_functional_requirements
  ];
  const highComplexity = allReqs.filter(r => r.complexity.toLowerCase().includes('high')).length;
  const technicalViability = Math.max(50, 100 - (highComplexity * 10));

  const overall = Math.round(
    (featureClarity + scopeFeasibility + userCentricity + technicalViability) / 4
  );

  return {
    overall,
    featureClarity,
    scopeFeasibility,
    userCentricity,
    technicalViability,
  };
}

/**
 * Transform backend Phase 2 User Journey response to frontend format
 */
export function transformPhase2JourneyResponse(response: Phase2UserJourneyResponse): AIAnalysisResult {
  const userJourney = transformToUserJourney(response);

  return {
    overallScore: 75, // Default score for journey
    breakdown: [
      { category: "Journey Completeness", score: 80, color: "hsl(var(--primary))" },
      { category: "UX Consideration", score: 78, color: "hsl(142 76% 36%)" },
      { category: "Edge Case Coverage", score: 70, color: "hsl(262 83% 58%)" },
      { category: "Accessibility", score: 72, color: "hsl(25 95% 53%)" },
    ],
    feedback: [],
    suggestedQuestions: [
      "How will the journey differ for mobile vs desktop users?",
      "What's the expected drop-off rate at each step?",
      "How will you track user progress through this journey?",
    ],
    phase2Details: {
      overallScore: 75,
      breakdown: [
        { category: "Journey Completeness", score: 80, color: "hsl(var(--primary))" },
        { category: "UX Consideration", score: 78, color: "hsl(142 76% 36%)" },
        { category: "Edge Case Coverage", score: 70, color: "hsl(262 83% 58%)" },
        { category: "Accessibility", score: 72, color: "hsl(25 95% 53%)" },
      ],
      featureAnalysis: {
        normalizedFeatures: [],
        requirements: [],
        mvpScope: [],
        warnings: [],
      },
      userJourney,
    },
  };
}

function transformToUserJourney(response: Phase2UserJourneyResponse): Phase2JourneyResult {
  const journeySteps: JourneyStep[] = response.steps.map(step => ({
    id: `j${step.step_number}`,
    stepNumber: step.step_number,
    action: step.user_action,
    userGoal: step.goal,
    systemResponse: step.system_response,
    uiElement: step.title,
    emotionalState: 'neutral',
    device: 'both',
    notes: step.success_criteria,
    alternatives: step.potential_issues,
  }));

  return {
    featureName: response.feature_name,
    journeySteps,
    happyPathSummary: response.overview,
    edgeCases: response.error_scenarios,
    accessibilityNotes: [],
  };
}

// ============================================================================
// Phase 3 Transformer
// ============================================================================

/**
 * Transform backend Phase 3 response to frontend format
 */
export function transformPhase3Response(response: Phase3Response): AIAnalysisResult {
  // Calculate scores based on the analysis
  const scores = calculatePhase3Scores(response);

  // Map backend Porter forces to frontend format
  const porterFiveForces: PorterFiveForces = {
    competitive_rivalry: transformBackendPorterForce(response.porters_analysis.competitive_rivalry),
    supplier_power: transformBackendPorterForce(response.porters_analysis.supplier_power),
    buyer_power: transformBackendPorterForce(response.porters_analysis.buyer_power),
    threat_of_substitutes: transformBackendPorterForce(response.porters_analysis.threat_of_substitution),
    threat_of_new_entrants: transformBackendPorterForce(response.porters_analysis.threat_of_new_entry),
    overall_attractiveness: response.porters_analysis.overall_assessment,
  };

  return {
    overallScore: scores.overall,
    breakdown: [
      { category: "Research Depth", score: scores.researchDepth, color: "hsl(var(--primary))" },
      { category: "Differentiation", score: scores.differentiation, color: "hsl(142 76% 36%)" },
      { category: "Market Size", score: scores.marketSize, color: "hsl(262 83% 58%)" },
      { category: "Competition", score: scores.competition, color: "hsl(25 95% 53%)" },
    ],
    feedback: [
      {
        id: "3-1",
        type: "insight",
        title: "Market Gap Identified",
        content: response.usp_generation.primary_usp.usp,
        confidence: 91,
        category: "Opportunity",
      },
      {
        id: "3-2",
        type: "insight",
        title: "Competitive Positioning",
        content: response.usp_generation.positioning_statement,
        confidence: 88,
        category: "Strategy",
      },
    ],
    suggestedQuestions: [
      "What's your pricing strategy vs competitors?",
      "How will you acquire first 100 users?",
      "What partnerships could accelerate market entry?",
    ],
    // Add phase 3 specific details if needed
    phase3Details: {
      marketResearch: response.market_research,
      porterFiveForces: porterFiveForces,
      competitorAnalysis: response.competitor_analysis,
      uspGeneration: response.usp_generation,
    },
  };
}

function transformBackendPorterForce(f: any): PorterForce {
  const levelMap: Record<string, 'High' | 'Medium' | 'Low'> = {
    'very-low': 'Low',
    'low': 'Low',
    'moderate': 'Medium',
    'high': 'High',
    'very-high': 'High',
  };

  return {
    force: f.force || '',
    level: levelMap[f.strength] || 'Medium',
    description: f.analysis || '',
    key_factors: f.key_factors || [],
  };
}

function calculatePhase3Scores(response: Phase3Response) {
  // Calculate research depth based on trends and market info completeness
  const researchDepth = Math.min(95, 70 + response.market_research.key_trends.length * 5);

  // Calculate differentiation based on USPs
  const differentiation = Math.min(95, 75 + response.usp_generation.secondary_usps.length * 5);

  // Calculate market size score (simplified)
  const marketSize = 85;

  // Calculate competition score based on gaps identified
  const competition = Math.min(95, 70 + response.competitor_analysis.market_gaps.length * 5);

  const overall = Math.round(
    (researchDepth + differentiation + marketSize + competition) / 4
  );

  return {
    overall,
    researchDepth,
    differentiation,
    marketSize,
    competition,
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

function getColorForIndex(index: number): string {
  const colors = [
    "hsl(var(--destructive))",
    "hsl(25 95% 53%)",
    "hsl(142 76% 36%)",
    "hsl(var(--primary))",
    "hsl(262 83% 58%)",
  ];
  return colors[index % colors.length];
}

// Extend AIAnalysisResult type to include phase3Details
declare module './mockAIData' {
  interface AIAnalysisResult {
    phase3Details?: {
      marketResearch: Phase3Response['market_research'];
      porterFiveForces: PorterFiveForces;
      competitorAnalysis: Phase3Response['competitor_analysis'];
      uspGeneration: Phase3Response['usp_generation'];
    };
  }
}
