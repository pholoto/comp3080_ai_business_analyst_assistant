"""Conversation harness for exercising the backend stack with real LLM.

python -m back_end.conversation_simulator

This tests the new phase-based API with structured inputs for:
- Phase 1: Problem Definition
- Phase 2: Requirements Analysis (Feature Analyzer + User Journey)
- Phase 3: Market Analysis
- Phase 7: Documentation (SRS generation)

Uses a Federated Learning Vietnamese Keyboard project as the test case.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from AI.llm import get_default_client
# Import the new phase modules
from back_end.phases import (DocumentationInput, DocumentationPhase,
                             DocumentationType, FeatureAnalyzerInput,
                             MarketAnalysisInput, MarketAnalysisPhase,
                             ProblemDefinitionInput, ProblemDefinitionPhase,
                             RequirementsAnalysisPhase, UserJourneyInput)
from back_end.phases.phase2_requirements_analysis import TeamSkillLevel
from back_end.phases.phase3_market_analysis import GeographicScope

SIMULATED_USER_ID = "test_user"


# ============================================================================
# Test Case: Federated Learning Vietnamese Keyboard
# ============================================================================

# Phase 1 Input: Problem Definition
PHASE1_INPUT = ProblemDefinitionInput(
    problem_description="""
    Vietnamese mobile users struggle with keyboard predictions that don't understand 
    Vietnamese language nuances, diacritics, and regional expressions. Current keyboard 
    apps are trained on generic datasets that fail to capture the unique typing patterns 
    of Vietnamese speakers. Users constantly need to manually correct predictions, 
    leading to slower typing and frustration. Additionally, users are concerned about 
    their typing data being collected and sent to cloud servers, raising privacy concerns 
    especially for sensitive communications.
    """,
    target_users="""
    Vietnamese-speaking mobile phone users aged 18-45, primarily using Android devices, 
    who type frequently in Vietnamese for personal messaging, social media, and work 
    communication. This includes students, young professionals, and business users who 
    value both typing efficiency and data privacy.
    """,
    why_it_matters="""
    Vietnam has over 70 million smartphone users, and inefficient typing directly impacts 
    productivity and user experience. Poor predictions lead to communication errors, 
    slower message composition, and user frustration. Privacy concerns prevent users from 
    fully utilizing cloud-based AI features. A solution that improves predictions while 
    keeping data on-device would significantly enhance the mobile experience for millions 
    of Vietnamese users and set a new standard for privacy-preserving AI applications.
    """,
    pain_points=[
        "Keyboard fails to predict Vietnamese words with correct diacritics (tones)",
        "Regional slang and informal expressions are not recognized",
        "Constant manual corrections slow down typing speed by 30-40%",
        "Privacy concerns about typing data being uploaded to cloud servers",
        "Existing keyboards don't learn from user's personal typing patterns effectively",
        "Switching between Vietnamese and English is clunky and error-prone",
    ],
    has_existing_solutions=True,
    current_solutions="""
    Current solutions include Gboard (Google), Laban Key, and OpenKey. Issues:
    - Gboard: Good general predictions but weak on Vietnamese diacritics and slang, 
      requires cloud processing which raises privacy concerns
    - Laban Key: Vietnamese-focused but outdated ML models, limited personalization
    - OpenKey: Focuses on input methods rather than intelligent predictions
    All existing solutions require sending typing data to servers for model improvement,
    which many privacy-conscious users find unacceptable.
    """
)

# Phase 2 Input: Feature Analyzer
PHASE2_FEATURE_INPUT = FeatureAnalyzerInput(
    desired_features=[
        "Vietnamese diacritic prediction with tone marks",
        "On-device federated learning for personalization",
        "Privacy-preserving model updates",
        "Regional slang and informal expression support",
        "Bilingual Vietnamese-English seamless switching",
        "Offline mode with full functionality",
        "User typing pattern learning",
        "Secure aggregation for model updates",
        "Custom dictionary for user-specific terms",
        "Swipe typing support for Vietnamese",
        "Voice-to-text with Vietnamese dialect support",
        "Emoji and sticker predictions based on context",
    ],
    mvp_goal="""
    Enable Vietnamese mobile users to type messages with 85% prediction accuracy 
    for Vietnamese words including diacritics, while keeping all personal typing 
    data on-device, within 6 months of development.
    """,
    deadline="6 months",
    team_skill_level=TeamSkillLevel.MIXED_EXPERIENCE,
    additional_constraints="""
    - Must work on Android devices with 2GB+ RAM
    - Model size must be under 50MB for on-device deployment
    - Battery impact should be minimal (less than 5% additional drain)
    - Must comply with Vietnamese data privacy regulations
    - Team has experience with Python/TensorFlow but limited mobile development experience
    """
)

# Phase 2 Input: User Journey (for a specific feature)
PHASE2_JOURNEY_INPUT = UserJourneyInput(
    selected_feature="On-device federated learning for personalization"
)

# Phase 3 Input: Market Analysis
PHASE3_INPUT = MarketAnalysisInput(
    geographic_scope=GeographicScope.NATIONAL,
    industry_context="Mobile Technology / EdTech / AI Keyboards",
    competitors=["Google Gboard", "Laban Key", "OpenKey", "SwiftKey", "Fleksy"]
)

# Phase 7 Input: Documentation (All 4 types)
PHASE7_SRS_INPUT = DocumentationInput(
    document_type=DocumentationType.SOFTWARE_ENGINEERING,
    project_title="Vietnamese Federated Keyboard Assistant",
    author_name="COMP3080 Business Analyst Assistant Team",
    user_id=SIMULATED_USER_ID,
)

PHASE7_ACADEMIC_INPUT = DocumentationInput(
    document_type=DocumentationType.ACADEMIC_REPORT,
    project_title="Vietnamese Federated Keyboard Assistant",
    author_name="COMP3080 Business Analyst Assistant Team",
    user_id=SIMULATED_USER_ID,
)

PHASE7_BUSINESS_INPUT = DocumentationInput(
    document_type=DocumentationType.BUSINESS_PROPOSAL,
    project_title="Vietnamese Federated Keyboard Assistant",
    author_name="COMP3080 Business Analyst Assistant Team",
    user_id=SIMULATED_USER_ID,
)

PHASE7_PITCH_INPUT = DocumentationInput(
    document_type=DocumentationType.STARTUP_PITCH,
    project_title="Vietnamese Federated Keyboard Assistant",
    author_name="COMP3080 Business Analyst Assistant Team",
    user_id=SIMULATED_USER_ID,
)


def simulate_phase_conversation() -> None:
    """Run the phase-based conversation simulation."""
    
    print("=" * 80)
    print("PHASE-BASED ANALYSIS SIMULATION")
    print("Project: Vietnamese Federated Keyboard Assistant")
    print("=" * 80)
    
    # Initialize real LLM client
    llm = get_default_client()
    print("\n[LLM] Using real LLM client for analysis")
    
    # Storage for phase outputs (simulates session state)
    phase_outputs: Dict[str, Any] = {}
    
    # =========================================================================
    # PHASE 1: Problem Definition
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: PROBLEM DEFINITION")
    print("=" * 80)
    print("\n📥 INPUT:")
    print(f"  Problem Description: {PHASE1_INPUT.problem_description[:100]}...")
    print(f"  Target Users: {PHASE1_INPUT.target_users[:80]}...")
    print(f"  Why It Matters: {PHASE1_INPUT.why_it_matters[:80]}...")
    print(f"  Pain Points: {len(PHASE1_INPUT.pain_points)} items")
    print(f"  Has Existing Solutions: {PHASE1_INPUT.has_existing_solutions}")
    
    print("\n⏳ Processing Phase 1...")
    phase1 = ProblemDefinitionPhase(llm_client=llm)
    
    try:
        phase1_output = phase1.run(PHASE1_INPUT)
        phase_outputs["phase1"] = phase1_output.model_dump()
        
        print("\n📤 OUTPUT:")
        print("\n1️⃣ Quality Score:")
        print(f"   Overall: {phase1_output.quality_score.overall_score}%")
        for dim in phase1_output.quality_score.dimensions:
            print(f"   - {dim.name}: {dim.score}%")
        
        print("\n2️⃣ Normalized Problem Summary:")
        print(f"   {phase1_output.normalized_summary.summary[:200]}...")
        
        print("\n3️⃣ Primary User Persona:")
        print(f"   Role: {phase1_output.personas.primary_user.role}")
        print(f"   Goal: {phase1_output.personas.primary_user.goal}")
        print(f"   Urgency: {phase1_output.personas.primary_user.urgency}")
        
        print("\n4️⃣ Pain Moments:")
        for i, pm in enumerate(phase1_output.pain_moments[:3], 1):
            print(f"   {i}. {pm.moment}")
        
        print("\n5️⃣ Root Causes:")
        for rc in phase1_output.root_causes[:3]:
            print(f"   - [{rc.category}] {rc.cause}")
        
        print("\n6️⃣ Transition Questions for Phase 2:")
        for q in phase1_output.transition_questions[:3]:
            print(f"   • {q}")
            
    except Exception as e:
        print(f"\n❌ Phase 1 failed: {e}")
        return
    
    print("\n✅ Phase 1 completed. Waiting before Phase 2...")
    time.sleep(5)
    
    # =========================================================================
    # PHASE 2: Requirements Analysis - Feature Analyzer
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2A: FEATURE ANALYZER")
    print("=" * 80)
    print("\n📥 INPUT:")
    print(f"  Desired Features: {len(PHASE2_FEATURE_INPUT.desired_features)} items")
    for f in PHASE2_FEATURE_INPUT.desired_features[:5]:
        print(f"    - {f}")
    print(f"    ... and {len(PHASE2_FEATURE_INPUT.desired_features) - 5} more")
    print(f"  MVP Goal: {PHASE2_FEATURE_INPUT.mvp_goal[:80]}...")
    print(f"  Deadline: {PHASE2_FEATURE_INPUT.deadline}")
    if PHASE2_FEATURE_INPUT.team_skill_level:
        print(f"  Team: {PHASE2_FEATURE_INPUT.team_skill_level.value}")
    
    # Add primary persona from Phase 1
    if phase_outputs.get("phase1"):
        primary_persona = phase_outputs["phase1"].get("personas", {}).get("primary_user", {}).get("role", "")
        PHASE2_FEATURE_INPUT.primary_user_persona = primary_persona
        print(f"  Primary Persona (from Phase 1): {primary_persona[:50]}...")
    
    print("\n⏳ Processing Phase 2A (Feature Analyzer)...")
    phase2 = RequirementsAnalysisPhase(
        llm_client=llm,
        phase1_context=phase_outputs.get("phase1"),
    )
    
    try:
        phase2_features_output = phase2.analyze_features(PHASE2_FEATURE_INPUT)
        phase_outputs["phase2"] = phase2_features_output.model_dump()
        
        print("\n📤 OUTPUT:")
        print("\n1️⃣ Normalized Features:")
        for nf in phase2_features_output.normalized_features[:5]:
            print(f"   [{nf.category.value}] {nf.normalized_name}")
        
        print("\n2️⃣ Functional Requirements (Must-Have):")
        must_have = [r for r in phase2_features_output.functional_requirements 
                     if r.moscow_priority.value == "must-have"]
        for fr in must_have[:5]:
            print(f"   {fr.id}: {fr.name} [{fr.complexity.value}]")
        
        print("\n3️⃣ Non-Functional Requirements:")
        for nfr in phase2_features_output.non_functional_requirements[:3]:
            print(f"   {nfr.id}: {nfr.attribute} - {nfr.requirement[:50]}...")
        
        print("\n4️⃣ MVP Scope:")
        print(f"   Included Features: {len(phase2_features_output.mvp_scope.included_features)}")
        for f in phase2_features_output.mvp_scope.included_features[:3]:
            print(f"   - {f.feature_name}")
        print(f"   Excluded: {len(phase2_features_output.mvp_scope.excluded_features)} features")
        
        print("\n5️⃣ Scope Warnings:")
        for sw in phase2_features_output.scope_warnings[:3]:
            print(f"   ⚠️ [{sw.severity}] {sw.warning_type}: {sw.description[:60]}...")
            
    except Exception as e:
        print(f"\n❌ Phase 2A failed: {e}")
        import traceback
        traceback.print_exc()
        return  # Stop execution - Phase 2 is required for later phases
    
    print("\n✅ Phase 2A completed. Waiting before Phase 2B...")
    time.sleep(5)
    
    # =========================================================================
    # PHASE 2B: User Journey Generator
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2B: USER JOURNEY GENERATOR")
    print("=" * 80)
    print("\n📥 INPUT:")
    print(f"  Selected Feature: {PHASE2_JOURNEY_INPUT.selected_feature}")
    
    print("\n⏳ Processing Phase 2B (User Journey)...")
    
    try:
        phase2_journey_output = phase2.generate_user_journey(PHASE2_JOURNEY_INPUT)
        
        print("\n📤 OUTPUT:")
        print(f"\n🗺️ Journey: {phase2_journey_output.journey_title}")
        print(f"   Overview: {phase2_journey_output.overview[:100]}...")
        
        print("\n   Steps:")
        for step in phase2_journey_output.steps[:5]:
            print(f"   {step.step_number}. {step.title}")
            print(f"      Goal: {step.goal[:60]}...")
            print(f"      User Action: {step.user_action[:60]}...")
            
    except Exception as e:
        print(f"\n❌ Phase 2B failed: {e}")
    
    print("\n✅ Phase 2B completed. Waiting before Phase 3...")
    time.sleep(5)
    
    # =========================================================================
    # PHASE 3: Market Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: MARKET ANALYSIS")
    print("=" * 80)
    print("\n📥 INPUT:")
    if PHASE3_INPUT.geographic_scope:
        print(f"  Geographic Scope: {PHASE3_INPUT.geographic_scope.value}")
    print(f"  Industry: {PHASE3_INPUT.industry_context}")
    print(f"  Competitors: {', '.join(PHASE3_INPUT.competitors or [])}")
    
    print("\n⏳ Processing Phase 3...")
    phase3 = MarketAnalysisPhase(
        llm_client=llm,
        phase1_context=phase_outputs.get("phase1"),
        phase2_context=phase_outputs.get("phase2"),
    )
    
    try:
        phase3_output = phase3.run(PHASE3_INPUT)
        phase_outputs["phase3"] = phase3_output.model_dump()
        
        print("\n📤 OUTPUT:")
        print("\n1️⃣ Market Research:")
        print(f"   Overview: {phase3_output.market_research.overview[:150]}...")
        if phase3_output.market_research.market_size:
            print(f"   Market Size: {phase3_output.market_research.market_size.value}")
        print(f"   Key Trends: {len(phase3_output.market_research.key_trends)} identified")
        
        print("\n2️⃣ Porter's Five Forces:")
        print(f"   Supplier Power: {phase3_output.porters_analysis.supplier_power.strength.value}")
        print(f"   Buyer Power: {phase3_output.porters_analysis.buyer_power.strength.value}")
        print(f"   Competitive Rivalry: {phase3_output.porters_analysis.competitive_rivalry.strength.value}")
        print(f"   Threat of Substitution: {phase3_output.porters_analysis.threat_of_substitution.strength.value}")
        print(f"   Threat of New Entry: {phase3_output.porters_analysis.threat_of_new_entry.strength.value}")
        
        print("\n3️⃣ Competitor Analysis:")
        for comp in phase3_output.competitor_analysis.competitors[:3]:
            print(f"   📊 {comp.name}")
            print(f"      Business Model: {comp.business_model[:50]}...")
            print(f"      Strengths: {', '.join(comp.strengths[:2])}")
        
        print("\n4️⃣ Unique Selling Points:")
        print(f"   Primary USP: {phase3_output.usp_generation.primary_usp.usp}")
        print(f"   Positioning: {phase3_output.usp_generation.positioning_statement[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Phase 3 failed: {e}")
    
    print("\n✅ Phase 3 completed. Waiting before Phase 7...")
    time.sleep(5)
    
    # =========================================================================
    # PHASE 7: Documentation (All 4 Document Types)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 7: DOCUMENTATION (ALL 4 DOCUMENT TYPES)")
    print("=" * 80)
    
    phase7 = DocumentationPhase(llm_client=llm)
    
    # All 4 document type inputs
    all_doc_inputs = [
        ("SRS (Software Engineering)", PHASE7_SRS_INPUT),
        ("Academic Report", PHASE7_ACADEMIC_INPUT),
        ("Business Proposal", PHASE7_BUSINESS_INPUT),
        ("Startup Pitch", PHASE7_PITCH_INPUT),
    ]
    
    for doc_name, doc_input in all_doc_inputs:
        print("\n" + "-" * 60)
        print(f"📄 Generating: {doc_name}")
        print("-" * 60)
        
        print("\n📥 INPUT:")
        print(f"  Document Type: {doc_input.document_type.value}")
        print(f"  Project Title: {doc_input.project_title}")
        print(f"  Author: {doc_input.author_name}")
        
        # Populate with all previous phase outputs
        doc_input.phase1_output = phase_outputs.get("phase1")
        doc_input.phase2_output = phase_outputs.get("phase2")
        doc_input.phase3_output = phase_outputs.get("phase3")
        
        # Debug: Show what context is available
        print(f"\n  Context from Phase 1: {'Yes' if phase_outputs.get('phase1') else 'No'}")
        print(f"  Context from Phase 2: {'Yes' if phase_outputs.get('phase2') else 'No'}")
        print(f"  Context from Phase 3: {'Yes' if phase_outputs.get('phase3') else 'No'}")
        
        print(f"\n⏳ Processing {doc_name}...")
        
        try:
            phase7_output = phase7.run(doc_input)
            
            print("\n📤 OUTPUT:")
            print(f"\n📄 Document Type: {phase7_output.document_type.value}")
            
            if phase7_output.srs_document:
                srs = phase7_output.srs_document
                print(f"\n📋 SRS Document: {srs.document_title}")
                print(f"   Version: {srs.version}")
                print(f"   Authors: {', '.join(srs.authors)}")
                
                print("\n   Sections:")
                print(f"   1. Introduction: {srs.introduction.title}")
                print(f"   2. Overall Description: {srs.overall_description.title}")
                print(f"   3. Specific Requirements: {srs.specific_requirements.title}")
                print(f"   4. External Interfaces: {srs.external_interfaces.title}")
                
                print(f"\n   System Features: {len(srs.system_features)}")
                for sf in srs.system_features[:3]:
                    print(f"   - {sf.get('feature_id', 'N/A')}: {sf.get('name', 'Unknown')}")
                
                print(f"\n   Glossary Terms: {len(srs.glossary)}")
            
            if phase7_output.academic_report:
                report = phase7_output.academic_report
                print(f"\n📚 Academic Report: {report.title}")
                if report.authors:
                    print(f"   Authors: {', '.join(report.authors)}")
                print(f"   Abstract: {report.abstract[:100]}..." if report.abstract else "   Abstract: N/A")
            
            if phase7_output.business_proposal:
                proposal = phase7_output.business_proposal
                print(f"\n💼 Business Proposal: {proposal.title}")
                print(f"   Executive Summary: {proposal.executive_summary[:100]}..." if proposal.executive_summary else "   Executive Summary: N/A")
                if proposal.problem_statement:
                    print(f"   Problem: {proposal.problem_statement.content[:80]}..." if proposal.problem_statement.content else "")
            
            if phase7_output.startup_pitch:
                pitch = phase7_output.startup_pitch
                print(f"\n🚀 Startup Pitch: {pitch.pitch_title}")
                print(f"   Tagline: {pitch.tagline}")
                print(f"   Elevator Pitch: {pitch.elevator_pitch[:100]}..." if pitch.elevator_pitch else "   Elevator Pitch: N/A")
            
            if phase7_output.missing_information:
                print("\n⚠️ Missing Information:")
                for missing in phase7_output.missing_information:
                    print(f"   - {missing}")
            
            # Export the document to user's folder
            print(f"\n📁 Exporting {doc_name}...")
            exported_files = phase7.export_document(
                output=phase7_output,
                user_id=SIMULATED_USER_ID,
                base_path=str(Path(__file__).parent / "data"),
            )
            
            if exported_files:
                print("\n✅ Documents exported:")
                for file_type, file_path in exported_files.items():
                    print(f"   - {file_type.upper()}: {file_path}")
            else:
                print("\n⚠️ No documents were exported")
                    
        except Exception as e:
            print(f"\n❌ {doc_name} failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait a bit between document generations
        if doc_input != all_doc_inputs[-1][1]:
            print("\n⏳ Waiting 5 seconds before next document...")
            time.sleep(5)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print("\n✅ Phases Executed:")
    print("   1. Problem Definition - Quality scoring, personas, pain moments, root causes")
    print("   2A. Feature Analyzer - Requirements, MVP scope, warnings")
    print("   2B. User Journey - Step-by-step user flow")
    print("   3. Market Analysis - Porter's Five Forces, competitors, USPs")
    print("   7. Documentation - All 4 document types:")
    print("      - IEEE SRS (Software Engineering)")
    print("      - Academic Report (VinUni template)")
    print("      - Business Proposal")
    print("      - Startup Pitch")
    
    # Save outputs to file for inspection
    output_path = Path(__file__).parent / "data" / SIMULATED_USER_ID / "phase_outputs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(phase_outputs, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n📁 Full outputs saved to: {output_path}")
    except Exception as e:
        print(f"\n⚠️ Could not save outputs: {e}")


if __name__ == "__main__":
    simulate_phase_conversation()

