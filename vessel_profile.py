from __future__ import annotations

from datetime import datetime
from typing import Any

from beartype import beartype
from pydantic import BaseModel, Field, model_validator


class VesselMetadata(BaseModel):
    name: str
    built_year: int | None = None
    flag: str | None = None
    classification_society: str | None = None
    call_sign: str | None = None


class TechnicalSpecs(BaseModel):
    imo_number: str | None = None
    type: str
    dwt: float
    gross_tonnage: float
    net_tonnage: float
    loa_meters: float
    beam_meters: float | None = None
    moulded_depth_meters: float | None = None
    lbp_meters: float | None = None
    draft_sw_s_w_t: list[float] = Field(default_factory=list)
    suez_gt: float | None = None
    suez_nt: float | None = None


class OperationalData(BaseModel):
    cargo_quantity_mt: float | None = None
    days_alongside: float | None = None
    arrival_time: datetime | None = None
    departure_time: datetime | None = None
    activity: str | None = None
    num_operations: int = 2
    num_holds: int | None = None


class VesselProfile(BaseModel):
    vessel_metadata: VesselMetadata
    technical_specs: TechnicalSpecs
    operational_data: OperationalData
    port: str = ""

    @model_validator(mode="after")
    def derive_port_duration(self) -> VesselProfile:
        ops = self.operational_data
        if ops.arrival_time and ops.departure_time and ops.days_alongside is None:
            delta = ops.departure_time - ops.arrival_time
            ops.days_alongside = delta.total_seconds() / 86400
        return self

    # ------------------------------------------------------------------
    # Convenience properties used by calculators
    # ------------------------------------------------------------------

    @property
    def gt(self) -> float:
        return self.technical_specs.gross_tonnage

    @property
    def nt(self) -> float:
        return self.technical_specs.net_tonnage

    @property
    def loa(self) -> float:
        return self.technical_specs.loa_meters

    @property
    def days_in_port(self) -> float:
        """Actual time between arrival and departure in days."""
        ops = self.operational_data
        if ops.arrival_time and ops.departure_time:
            delta = ops.departure_time - ops.arrival_time
            return delta.total_seconds() / 86400
        return ops.days_alongside or 0.0

    @property
    def days_alongside_value(self) -> float:
        return self.operational_data.days_alongside or self.days_in_port

    @property
    def vessel_type(self) -> str:
        return self.technical_specs.type.lower()

    @property
    def num_operations(self) -> int:
        return self.operational_data.num_operations


@beartype
def load_vessel_profile(data: dict[str, Any], port: str = "") -> VesselProfile:
    """Parse a raw vessel profile dict into a typed VesselProfile."""
    return VesselProfile(**data, port=port)
