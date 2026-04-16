"""
Natural language vessel query parser.

Converts a free-text vessel description into a structured VesselProfile
using OpenAI structured output, fulfilling the task requirement:
  "a system that consumes a complex Port Tariff PDF and a natural language
   vessel query to automate the calculation of different port dues."

Example input:
  "Calculate port dues for SUDESTADA, a 51,300 GT bulk carrier at Durban,
   DWT 93274, NT 31192, LOA 229.2m, 3.39 days alongside, 2 operations"

Example output:
  (VesselProfile, "Durban")
"""

from __future__ import annotations

from typing import Any

from beartype import beartype
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from core.config import OPENAI_API_KEY, OPENAI_MODEL
from core.vessel_profile import VesselProfile, load_vessel_profile

# ---------------------------------------------------------------------------
# Intermediate schema for LLM extraction (flat, forgiving)
# ---------------------------------------------------------------------------


class ParsedVesselQuery(BaseModel):
    """Schema the LLM fills from a natural language vessel description.

    All fields except vessel_name and port are optional — the LLM extracts
    whatever it can find in the text, and defaults fill the rest.
    """

    vessel_name: str = Field(description="Name of the vessel")
    port: str = Field(description="Port name (e.g. Durban, Cape Town)")

    vessel_type: str = Field(default="General Cargo", description="Vessel type (e.g. Bulk Carrier, Container Ship, Tanker)")
    gross_tonnage: float = Field(default=0, description="Gross tonnage (GT)")
    net_tonnage: float = Field(default=0, description="Net tonnage (NT)")
    dwt: float = Field(default=0, description="Deadweight tonnage (DWT)")
    loa_meters: float = Field(default=0, description="Length overall in meters")
    beam_meters: float | None = Field(default=None, description="Beam in meters")
    draft_meters: float | None = Field(default=None, description="Draft in meters")

    days_alongside: float | None = Field(default=None, description="Days alongside (duration of port stay)")
    num_operations: int = Field(default=2, description="Number of operations (arrival + departure = 2)")
    activity: str | None = Field(default=None, description="Activity description (e.g. Loading Containers)")
    cargo_quantity_mt: float | None = Field(default=None, description="Cargo quantity in metric tonnes")


_QUERY_PROMPT = """\
You are a maritime data extraction assistant.

Parse the following natural language vessel query and extract all vessel details
and port information you can find.  If a value is not mentioned, leave it as the
default (0 for numbers, null for optional fields).

For gross_tonnage: look for "GT", "gross tonnage", or just a tonnage number.
For net_tonnage: look for "NT" or "net tonnage".
For loa_meters: look for "LOA", "length overall", or "length" in meters.
For days_alongside: look for "days alongside", "days in port", or duration info.
For num_operations: default is 2 (arrival + departure) unless otherwise stated.

VESSEL QUERY:
{query}
"""


@beartype
def parse_vessel_query(query: str) -> tuple[VesselProfile, str]:
    """
    Parse a natural language vessel description into a VesselProfile and port name.

    Uses OpenAI structured output to guarantee the response matches the
    ParsedVesselQuery schema, then converts to the standard VesselProfile.

    Raises ValueError if OPENAI_API_KEY is not configured.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is required for natural language query parsing. "
            "Set it in .env or as an environment variable."
        )

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    structured_llm = llm.with_structured_output(ParsedVesselQuery)

    prompt = _QUERY_PROMPT.format(query=query)
    parsed: ParsedVesselQuery = structured_llm.invoke(prompt)  # type: ignore[assignment]

    print(f"[query_parser] Parsed: {parsed.vessel_name} @ {parsed.port}")
    print(f"[query_parser]   GT={parsed.gross_tonnage}, NT={parsed.net_tonnage}, "
          f"DWT={parsed.dwt}, LOA={parsed.loa_meters}m")
    print(f"[query_parser]   Type={parsed.vessel_type}, Days={parsed.days_alongside}, "
          f"Ops={parsed.num_operations}")

    # Convert to the standard vessel data dict format
    vessel_data: dict[str, Any] = {
        "vessel_metadata": {
            "name": parsed.vessel_name,
            "built_year": None,
            "flag": None,
            "classification_society": None,
            "call_sign": None,
        },
        "technical_specs": {
            "type": parsed.vessel_type,
            "dwt": parsed.dwt,
            "gross_tonnage": parsed.gross_tonnage,
            "net_tonnage": parsed.net_tonnage,
            "loa_meters": parsed.loa_meters,
            "beam_meters": parsed.beam_meters,
            "draft_sw_s_w_t": [parsed.draft_meters or 0.0, 0.0, 0.0],
        },
        "operational_data": {
            "cargo_quantity_mt": parsed.cargo_quantity_mt,
            "days_alongside": parsed.days_alongside,
            "num_operations": parsed.num_operations,
            "activity": parsed.activity,
        },
    }

    vessel = load_vessel_profile(vessel_data, port=parsed.port)
    return vessel, parsed.port
