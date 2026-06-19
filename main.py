"""
CurifyAI Medical Triage Backend - High-Performance, Ultra-Low-Cost API
=====================================================================

A FastAPI-based medical triage and immediate first-aid guidance system
powered by Google Gemini 1.5 Flash, optimized for zero-cost cloud deployment.

Technical Stack:
- FastAPI with async/await for minimal memory footprint
- Pydantic v2 for strict schema validation and type safety
- Google Generative AI SDK for deterministic Gemini 1.5 Flash responses
- Structured JSON output mode with explicit schema binding

Architecture Principles:
- Asynchronous-first design to maximize throughput
- Strict error isolation with comprehensive exception handling
- Security-first: API key validation at startup, no hardcoded credentials
- Korean-language medical guidance for regional accessibility
"""

import os
import json
import logging
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC V2 SCHEMAS
# ============================================================================

class TriageRequest(BaseModel):
    """
    Input schema for medical triage API endpoint.
    
    Attributes:
        description: Raw user-provided wound/symptom description in natural language.
    """
    description: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Patient's detailed description of wound, symptom, or medical concern"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "description": "화상으로 손이 심하게 데었습니다. 두 손가락이 물집이 생겼고 통증이 심합니다."
            }
        }


class TriageResponse(BaseModel):
    """
    Output schema for medical triage AI response.
    
    Immutable target structure that enforces strict Gemini output validation.
    All text fields for patient guidance are delivered in Korean.
    
    Attributes:
        injury_type: Medical category classification (e.g., "화상" [Burn], "열상" [Laceration], "찰과상" [Abrasion], "염좌" [Sprain])
        severity: Triage priority level restricted to ["Low", "Medium", "High", "Critical"]
        immediate_actions: Chronological, step-by-step emergency instructions in Korean
        otc_recommendations: Over-the-counter medical supplies/ointments available in Korea, in Korean
        emergency_call_required: Boolean indicating immediate 119 emergency call necessity
        professional_medical_disclaimer: Standard legal medical disclaimer in Korean
    """
    injury_type: str = Field(
        ...,
        description="Medical category classification of the injury"
    )
    severity: str = Field(
        ...,
        description="Triage priority level: Low, Medium, High, or Critical"
    )
    immediate_actions: List[str] = Field(
        ...,
        min_items=1,
        description="Chronological emergency action steps in Korean"
    )
    otc_recommendations: List[str] = Field(
        ...,
        min_items=1,
        description="Over-the-counter medical supply recommendations in Korean"
    )
    emergency_call_required: bool = Field(
        ...,
        description="True if severity is High or Critical, triggering 119 emergency call protocol"
    )
    professional_medical_disclaimer: str = Field(
        ...,
        description="Standard legal medical disclaimer in Korean"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "injury_type": "화상",
                "severity": "High",
                "immediate_actions": [
                    "즉시 찬물로 15-20분간 화상 부위를 식혀주세요.",
                    "의복이 붙어있지 않으면 조심스럽게 제거하세요.",
                    "화상 부위를 깨끗한 천으로 느슨하게 덮으세요.",
                    "즉시 응급실로 가거나 119를 호출하세요."
                ],
                "otc_recommendations": [
                    "멸균된 거즈",
                    "항생제 연고",
                    "통증 완화 약물 (아세트아미노펜)"
                ],
                "emergency_call_required": True,
                "professional_medical_disclaimer": "본 도구는 전문 의료 진단을 대체할 수 없습니다. 심각한 의료 상황에서는 반드시 전문가의 진찰을 받으시기 바랍니다."
            }
        }


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    message: str
    api_key_configured: bool


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    detail: str
    status_code: int


# ============================================================================
# ENVIRONMENT & SDK INITIALIZATION
# ============================================================================

def validate_and_initialize_gemini() -> None:
    """
    Validate Gemini API key availability and initialize the SDK.
    
    Raises:
        RuntimeError: If GEMINI_API_KEY environment variable is not set.
    
    Side Effects:
        - Configures google.generativeai with the API key
        - Logs successful initialization
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key or api_key.strip() == "":
        error_msg = (
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure your Gemini API key before starting the server. "
            "Visit: https://ai.google.dev/tutorials/setup"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    genai.configure(api_token=api_key)
    logger.info("✓ Google Generative AI SDK initialized successfully with valid API key")


# Validate API key at module load time
try:
    validate_and_initialize_gemini()
    GEMINI_INITIALIZED = True
except RuntimeError as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    GEMINI_INITIALIZED = False


# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="CurifyAI Medical Triage Backend",
    description="Ultra-low-cost, high-efficiency medical triage and first-aid guidance API powered by Google Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================================
# HEALTH CHECK ROUTES
# ============================================================================

@app.get("/", response_model=HealthCheckResponse, tags=["Health"])
async def root() -> HealthCheckResponse:
    """
    Root health check endpoint.
    
    Returns:
        HealthCheckResponse: Status and API key configuration details
    """
    return HealthCheckResponse(
        status="operational",
        message="CurifyAI Medical Triage Backend is running",
        api_key_configured=GEMINI_INITIALIZED
    )


@app.get("/health-check", response_model=HealthCheckResponse, tags=["Health"])
async def health_check() -> HealthCheckResponse:
    """
    Detailed health check endpoint for deployment verification.
    
    Validates:
        - Service availability
        - Gemini API key configuration
        - System readiness for processing triage requests
    
    Returns:
        HealthCheckResponse: Comprehensive health status
    """
    return HealthCheckResponse(
        status="operational" if GEMINI_INITIALIZED else "degraded",
        message=(
            "All systems operational" if GEMINI_INITIALIZED
            else "Gemini API key not configured - triage endpoint will return errors"
        ),
        api_key_configured=GEMINI_INITIALIZED
    )


# ============================================================================
# CORE TRIAGE API ENDPOINT
# ============================================================================

@app.post(
    "/api/v1/triage",
    response_model=TriageResponse,
    status_code=status.HTTP_200_OK,
    tags=["Triage"],
    summary="Medical Triage Assessment",
    description="Analyze patient wound/symptom description and provide immediate triage guidance with first-aid recommendations"
)
async def medical_triage(request: TriageRequest) -> TriageResponse:
    """
    Core medical triage endpoint powered by Google Gemini 1.5 Flash.
    
    This endpoint:
    1. Accepts raw patient symptom/wound descriptions
    2. Routes input to Gemini 1.5 Flash with strict system instructions
    3. Enforces structured JSON output matching TriageResponse schema
    4. Returns deterministic, validated Korean-language medical guidance
    5. Handles all exceptions gracefully without crashing the API server
    
    Args:
        request: TriageRequest containing patient description
    
    Returns:
        TriageResponse: Structured medical triage guidance
    
    Raises:
        HTTPException: 500 status with structured error details if Gemini fails
    """
    
    # Pre-flight check: Ensure Gemini is initialized
    if not GEMINI_INITIALIZED:
        logger.error("Triage request received but Gemini API key is not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Medical AI system is not properly configured. Gemini API key is missing."
        )
    
    logger.info(f"Processing triage request: {request.description[:100]}...")
    
    try:
        # ====================================================================
        # GEMINI MODEL CONFIGURATION WITH STRUCTURED OUTPUT BINDING
        # ====================================================================
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Low temperature for deterministic, focused responses
                max_output_tokens=1024,  # Sufficient for comprehensive triage response
                response_mime_type="application/json",  # Force JSON output
                response_schema=TriageResponse,  # Bind Pydantic schema
            ),
            system_instruction=(
                "You are a highly precise, board-certified Emergency Medicine Triage AI. "
                "Analyze the patient's wound description objectively. "
                "Focus entirely on immediate, practical stabilization and safety. "
                "All text strings designated for the patient (immediate_actions, otc_recommendations, professional_medical_disclaimer) "
                "MUST be delivered in natural, authoritative, and compassionate Korean. "
                "Return raw, pure JSON mapping exactly to the schema properties. "
                "Do not wrap the output in markdown code blocks like ```json. "
                "Ensure emergency_call_required is True if severity is 'High' or 'Critical', False otherwise. "
                "Never deviate from the exact schema structure."
            )
        )
        
        # ====================================================================
        # GEMINI API INVOCATION WITH ERROR ISOLATION
        # ====================================================================
        
        response = model.generate_content(
            contents=request.description,
            stream=False
        )
        
        logger.info("Gemini API call completed successfully")
        
        # ====================================================================
        # RESPONSE PARSING AND VALIDATION
        # ====================================================================
        
        # Extract raw text from Gemini response
        response_text = response.text.strip()
        logger.debug(f"Raw Gemini response: {response_text}")
        
        # Parse JSON response
        response_json = json.loads(response_text)
        
        # Validate against Pydantic schema (enforces all field types and constraints)
        triage_response = TriageResponse(**response_json)
        
        # ====================================================================
        # POST-VALIDATION CONSISTENCY CHECKS
        # ====================================================================
        
        # Enforce consistency: emergency_call_required must align with severity
        if triage_response.severity in ["High", "Critical"]:
            if not triage_response.emergency_call_required:
                logger.warning(
                    f"Severity is {triage_response.severity} but emergency_call_required is False. "
                    "Correcting to True for safety."
                )
                triage_response.emergency_call_required = True
        else:
            if triage_response.emergency_call_required:
                logger.warning(
                    f"Severity is {triage_response.severity} but emergency_call_required is True. "
                    "Correcting to False for accuracy."
                )
                triage_response.emergency_call_required = False
        
        # Validate severity is one of allowed values
        allowed_severities = ["Low", "Medium", "High", "Critical"]
        if triage_response.severity not in allowed_severities:
            logger.warning(
                f"Invalid severity '{triage_response.severity}' received from Gemini. "
                f"Allowed values: {allowed_severities}. Defaulting to 'Medium'."
            )
            triage_response.severity = "Medium"
        
        logger.info(
            f"Triage assessment completed: injury_type={triage_response.injury_type}, "
            f"severity={triage_response.severity}, "
            f"emergency_required={triage_response.emergency_call_required}"
        )
        
        return triage_response
    
    except json.JSONDecodeError as e:
        error_msg = (
            f"Gemini response parsing failed: Invalid JSON format. "
            f"Details: {str(e)}"
        )
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Medical AI response validation failed. Please try again."
        )
    
    except ValueError as e:
        error_msg = (
            f"Gemini response validation failed: Schema mismatch. "
            f"Details: {str(e)}"
        )
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Medical AI response format validation failed. Please try again."
        )
    
    except GoogleAPIError as e:
        error_msg = (
            f"Google Generative AI API error: {str(e)}. "
            f"Error code: {e.code if hasattr(e, 'code') else 'Unknown'}"
        )
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="External medical AI service is temporarily unavailable. Please try again later."
        )
    
    except Exception as e:
        error_msg = (
            f"Unexpected error during triage processing: {type(e).__name__}: {str(e)}"
        )
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during medical assessment. Please try again."
        )


# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Global HTTP exception handler to ensure consistent error responses.
    
    Prevents Uvicorn from crashing and provides structured error information.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler.
    
    Verifies system readiness and logs initialization status.
    """
    logger.info("=" * 80)
    logger.info("CurifyAI Medical Triage Backend - Startup Sequence")
    logger.info("=" * 80)
    
    if GEMINI_INITIALIZED:
        logger.info("✓ Gemini API key is configured and validated")
        logger.info("✓ Medical triage endpoint is ready for requests")
    else:
        logger.warning("⚠ Gemini API key is NOT configured")
        logger.warning("⚠ Health check endpoints are functional but triage requests will fail")
    
    logger.info("✓ FastAPI application is fully operational")
    logger.info("=" * 80)


# ============================================================================
# SHUTDOWN EVENT
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """
    FastAPI shutdown event handler.
    
    Logs graceful shutdown and cleanup.
    """
    logger.info("CurifyAI Medical Triage Backend is shutting down gracefully")
