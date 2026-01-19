/**
 * API Service for connecting to the backend AI Business Analyst Assistant
 */

// Backend API base URL - can be configured via environment variable
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

// API Endpoints
const ENDPOINTS = {
  // Phase 1: Problem Definition
  phase1ProblemDefinition: (userId: string) => `/api/v2/phases/users/${userId}/phase1/problem-definition`,

  // Phase 2: Requirements Analysis
  phase2FeatureAnalyzer: (userId: string) => `/api/v2/phases/users/${userId}/phase2/feature-analyzer`,
  phase2UserJourney: (userId: string) => `/api/v2/phases/users/${userId}/phase2/user-journey`,

  // Phase 3: Market Analysis
  phase3MarketAnalysis: (userId: string) => `/api/v2/phases/users/${userId}/phase3/market-analysis`,

  // Phase 7: Documentation
  phase7Documentation: (userId: string) => `/api/v2/phases/users/${userId}/phase7/documentation`,

  // Session Management
  getSession: (userId: string) => `/api/v2/phases/users/${userId}/session`,
  clearSession: (userId: string) => `/api/v2/phases/users/${userId}/session`,
};

// ============================================================================
// Request Types
// ============================================================================

export interface Phase1Request {
  problem_description: string;
  target_users: string;
  why_it_matters: string;
  pain_points?: string[];
  has_existing_solutions?: boolean;
  current_solutions?: string | null;
}

export interface Phase2FeatureAnalyzerRequest {
  desired_features: string[];
  mvp_goal: string;
  deadline?: string | null;
  team_skill_level?: 'solo-developer' | 'junior-team' | 'senior-team' | 'mixed-experience' | null;
  additional_constraints?: string | null;
}

export interface Phase2UserJourneyRequest {
  selected_feature: string;
}

export interface Phase3Request {
  geographic_scope?: 'local' | 'regional' | 'multi-regional' | 'national' | 'international' | 'global' | null;
  industry_context?: string | null;
  competitors?: string[] | null;
}

export interface Phase7Request {
  document_type: 'academic-report' | 'software-engineering' | 'business-proposal' | 'startup-pitch';
  project_title?: string | null;
  author_name?: string | null;
  additional_context?: string | null;
}

// ============================================================================
// Response Types (from Backend)
// ============================================================================

export interface QualityDimension {
  name: string;
  score: number;
  feedback: string;
}

export interface QualityScore {
  overall_score: number;
  dimensions: QualityDimension[];
  summary: string;
}

export interface NormalizedProblemSummary {
  summary: string;
  key_elements: Record<string, string>;
}

export interface UserPersona {
  role: string;
  context: string;
  goal: string;
  urgency: string;
  failure_consequence: string;
}

export interface UserPersonaClarification {
  primary_user: UserPersona;
  secondary_users: UserPersona[];
  mvp_focus_rationale: string;
}

export interface PainMoment {
  moment: string;
  trigger: string;
  current_behavior: string;
  why_it_hurts: string;
}

export interface RootCause {
  cause: string;
  category: string;
  explanation: string;
}

export interface ExistingSolution {
  name: string;
  description: string;
  strengths: string[];
  weaknesses: string[];
  gap: string;
}

export interface ImpactStake {
  category: string;
  description: string;
  quantification: string;
}

export interface ScopeBoundary {
  exclusions: string[];
  rationale: string;
}

export interface Phase1Response {
  quality_score: QualityScore;
  normalized_summary: NormalizedProblemSummary;
  personas: UserPersonaClarification;
  pain_moments: PainMoment[];
  root_causes: RootCause[];
  existing_solutions_analysis: ExistingSolution[];
  impact_stakes: ImpactStake[];
  scope_boundary: ScopeBoundary;
  transition_questions: string[];
}

// Phase 2 Response Types
export interface BackendNormalizedFeature {
  original_name: string;
  normalized_name: string;
  category: string;
  description: string;
}

export interface BackendFunctionalRequirement {
  id: string;
  name: string;
  description: string;
  category: string;
  moscow_priority: string;
  complexity: string;
  rationale: string;
  acceptance_criteria: string[];
}

export interface BackendNonFunctionalRequirement {
  id: string;
  attribute: string;
  requirement: string;
  category: string;
  moscow_priority: string;
  complexity: string;
  metric?: string;
  rationale: string;
}

export interface BackendMvpFeature {
  feature_name: string;
  justification: string;
  estimated_effort: string;
}

export interface BackendMvpScope {
  included_features: BackendMvpFeature[];
  excluded_features: string[];
  mvp_rationale: string;
  estimated_timeline?: string;
}

export interface BackendScopeWarning {
  warning_type: string;
  description: string;
  severity: string;
  mitigation: string;
}

export interface Phase2FeatureAnalyzerResponse {
  normalized_features: BackendNormalizedFeature[];
  functional_requirements: BackendFunctionalRequirement[];
  non_functional_requirements: BackendNonFunctionalRequirement[];
  mvp_scope: BackendMvpScope;
  scope_warnings: BackendScopeWarning[];
}

export interface BackendUserJourneyStep {
  step_number: number;
  title: string;
  goal: string;
  user_action: string;
  system_response: string;
  success_criteria: string;
  potential_issues: string[];
}

export interface Phase2UserJourneyResponse {
  feature_name: string;
  journey_title: string;
  overview: string;
  preconditions: string[];
  steps: BackendUserJourneyStep[];
  postconditions: string[];
  alternative_flows: string[];
  error_scenarios: string[];
}

// Phase 3 Response Types
export interface MarketResearchSummary {
  market_size: string;
  growth_rate: string;
  key_trends: string[];
  market_maturity: string;
}

export interface PorterForce {
  force: string;
  level: 'High' | 'Medium' | 'Low';
  description: string;
  key_factors: string[];
}

export interface PorterFiveForces {
  competitive_rivalry: PorterForce;
  supplier_power: PorterForce;
  buyer_power: PorterForce;
  threat_of_substitutes: PorterForce;
  threat_of_new_entrants: PorterForce;
  overall_attractiveness: string;
}

export interface Competitor {
  name: string;
  description: string;
  business_model: string;
  target_customers: string;
  strengths: string[];
  weaknesses: string[];
  market_position: string;
}

export interface CompetitorAnalysis {
  competitors: Competitor[];
  market_gaps: string[];
  competitive_summary: string;
}

export interface USP {
  usp: string;
  rationale: string;
  supporting_evidence: string;
}

export interface USPGeneration {
  primary_usp: USP;
  secondary_usps: USP[];
  positioning_statement: string;
}

// Phase 3 Backend Types
export interface BackendMarketStatistic {
  metric: string;
  value: string;
  source: string;
  year?: string;
}

export interface BackendMarketResearchSummary {
  overview: string;
  market_size?: BackendMarketStatistic;
  growth_rate?: BackendMarketStatistic;
  key_statistics: BackendMarketStatistic[];
  key_trends: string[];
  market_drivers: string[];
  market_challenges: string[];
  sources: string[];
}

export interface BackendPorterForce {
  force: string;
  strength: 'very-low' | 'low' | 'moderate' | 'high' | 'very-high';
  analysis: string;
  key_factors: string[];
}

export interface BackendPortersFiveForces {
  supplier_power: BackendPorterForce;
  buyer_power: BackendPorterForce;
  competitive_rivalry: BackendPorterForce;
  threat_of_substitution: BackendPorterForce;
  threat_of_new_entry: BackendPorterForce;
  overall_assessment: string;
  strategic_implications: string[];
}

export interface BackendCompetitorProfile {
  name: string;
  description: string;
  business_model: string;
  target_customer: string;
  strengths: string[];
  weaknesses: string[];
  opportunities: string[];
  threats: string[];
  pricing_model?: string;
  market_share?: string;
  key_differentiators: string[];
}

export interface BackendCompetitorAnalysis {
  competitors: BackendCompetitorProfile[];
  market_gaps: string[];
  competitive_landscape_summary: string;
}

export interface BackendUniqueSellingPoint {
  usp: string;
  target_audience: string;
  supporting_evidence: string;
  differentiation_level: string;
}

export interface BackendUspGeneration {
  primary_usp: BackendUniqueSellingPoint;
  secondary_usps: BackendUniqueSellingPoint[];
  positioning_statement: string;
  value_proposition_canvas: Record<string, any>;
}

export interface Phase3Response {
  market_research: BackendMarketResearchSummary;
  porters_analysis: BackendPortersFiveForces;
  competitor_analysis: BackendCompetitorAnalysis;
  usp_generation: BackendUspGeneration;
}

export interface CompetitorResearchRequest {
  industry: string;
  geographic_scope?: string;
  known_competitors?: string;
}

export interface CompetitorResearchResponse {
  competitors: Array<{
    name: string;
    description: string;
    business_model: string;
    target_customers: string;
  }>;
}


// Phase 7 Response Types
export interface DocumentSection {
  title: string;
  content: string;
  subsections?: DocumentSection[];
}

export interface Phase7Response {
  document_type: string;
  title: string;
  generated_at: string;
  sections: DocumentSection[];
  metadata: Record<string, string>;
  download_url?: string;
}

// Session Response Types
export interface SessionSummary {
  user_id: string;
  has_phase1: boolean;
  has_phase2: boolean;
  has_phase3: boolean;
  phase1_summary?: Record<string, unknown>;
  phase2_summary?: Record<string, unknown>;
  phase3_summary?: Record<string, unknown>;
}

// ============================================================================
// API Error Handling
// ============================================================================

export class APIError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public details?: unknown
  ) {
    super(message);
    this.name = 'APIError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP error! status: ${response.status}`;
    let details: unknown;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
      details = errorData;
    } catch {
      // Ignore JSON parsing errors
    }
    throw new APIError(errorMessage, response.status, details);
  }
  return response.json();
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Phase 1: Problem Definition Analysis
 */
export const analyzePhase1 = async (userId: string, data: Phase1Request): Promise<Phase1Response> => {
  const url = `${API_BASE_URL}${ENDPOINTS.phase1ProblemDefinition(userId)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<Phase1Response>(response);
};

/**
 * Phase 2: Feature Analyzer
 */
export const analyzePhase2Features = async (userId: string, data: Phase2FeatureAnalyzerRequest): Promise<Phase2FeatureAnalyzerResponse> => {
  const url = `${API_BASE_URL}${ENDPOINTS.phase2FeatureAnalyzer(userId)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<Phase2FeatureAnalyzerResponse>(response);
};

/**
 * Phase 2: User Journey Generator
 */
export const generatePhase2UserJourney = async (userId: string, data: Phase2UserJourneyRequest): Promise<Phase2UserJourneyResponse> => {
  const url = `${API_BASE_URL}${ENDPOINTS.phase2UserJourney(userId)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<Phase2UserJourneyResponse>(response);
};

/**
 * Phase 3: Market Analysis
 */
export const analyzePhase3Market = async (userId: string, data: Phase3Request): Promise<Phase3Response> => {
  const url = `${API_BASE_URL}${ENDPOINTS.phase3MarketAnalysis(userId)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<Phase3Response>(response);
};

export const researchCompetitorsApi = async (userId: string, request: CompetitorResearchRequest): Promise<CompetitorResearchResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/v2/phases/users/${userId}/phase3/research-competitors`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new APIError(errorData.detail || 'Competitor research failed', response.status);
  }

  return response.json();
};

// ============================================================================
// Phase 7: Documentation
// ============================================================================

export async function generatePhase7Documentation(
  userId: string,
  request: Phase7Request
): Promise<Phase7Response> {
  const url = `${API_BASE_URL}${ENDPOINTS.phase7Documentation(userId)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return handleResponse<Phase7Response>(response);
}

// ============================================================================
// Session Management
// ============================================================================

/**
 * Get Session Summary
 */
export async function getSessionSummary(userId: string): Promise<SessionSummary> {
  const url = `${API_BASE_URL}${ENDPOINTS.getSession(userId)}`;
  const response = await fetch(url, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  return handleResponse<SessionSummary>(response);
}

/**
 * Clear Session
 */
export async function clearSession(userId: string): Promise<{ status: string; message: string }> {
  const url = `${API_BASE_URL}${ENDPOINTS.clearSession(userId)}`;
  const response = await fetch(url, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
  });
  return handleResponse<{ status: string; message: string }>(response);
}
