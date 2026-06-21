from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any

from .backtest import BacktestConfig


RISK_PROFILE_CATALOG_SCHEMA_VERSION = 1
DEFAULT_RISK_PROFILE_CATALOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "strategy_calibration_profiles.toml"
)


@dataclass(frozen=True)
class ResolvedRiskProfileCatalog:
    """Validated strategy-calibration risk profiles from versioned TOML."""

    schema_version: int
    catalog_version: str
    source_path: Path
    compact_order: tuple[str, ...]
    profiles: tuple[dict[str, Any], ...]


def _profile_identity(
    base_cfg: BacktestConfig,
    params: dict[str, Any],
) -> tuple[float | int | str, ...]:
    merged = {**asdict(base_cfg), **params}
    return (
        int(merged.get("leverage", base_cfg.leverage)),
        round(float(merged["max_position_fraction"]), 6),
        round(float(merged["stop_loss"]), 6),
        round(float(merged["take_profit"]), 6),
        float(bool(merged["use_atr_exits"])),
        round(float(merged["stop_loss_atr_multiplier"]), 6),
        round(float(merged["take_profit_atr_multiplier"]), 6),
        round(float(merged["min_exit_pct"]), 6),
        round(float(merged["max_exit_pct"]), 6),
        round(float(merged["risk_per_trade"]), 6),
        round(
            float(
                merged.get(
                    "min_position_scale",
                    base_cfg.min_position_scale,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "min_volatility_scale",
                    base_cfg.min_volatility_scale,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "max_volatility_scale",
                    base_cfg.max_volatility_scale,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "volatility_target",
                    base_cfg.volatility_target,
                )
            ),
            6,
        ),
        float(
            bool(
                merged.get(
                    "event_position_boost_enabled",
                    base_cfg.event_position_boost_enabled,
                )
            )
        ),
        round(
            float(
                merged.get(
                    "event_position_min_score",
                    base_cfg.event_position_min_score,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "event_position_boost_strength",
                    base_cfg.event_position_boost_strength,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "position_rebalance_band",
                    base_cfg.position_rebalance_band,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "max_position_fraction_step",
                    base_cfg.max_position_fraction_step,
                )
            ),
            6,
        ),
        round(float(merged["funding_rate_buffer"]), 6),
        round(float(merged["min_atr_cost_multiplier"]), 6),
        round(
            float(
                merged.get(
                    "min_confidence_gap",
                    base_cfg.min_confidence_gap,
                )
            ),
            6,
        ),
        float(
            bool(
                merged.get(
                    "market_structure_gate_enabled",
                    base_cfg.market_structure_gate_enabled,
                )
            )
        ),
        round(
            float(
                merged.get(
                    "min_market_structure_score",
                    base_cfg.min_market_structure_score,
                )
            ),
            6,
        ),
        float(
            bool(
                merged.get(
                    "crowding_filter_enabled",
                    base_cfg.crowding_filter_enabled,
                )
            )
        ),
        round(
            float(
                merged.get(
                    "max_crowding_risk",
                    base_cfg.max_crowding_risk,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "max_notional_exposure",
                    base_cfg.max_notional_exposure,
                )
            ),
            6,
        ),
        float(
            bool(
                merged.get(
                    "platform_event_gate_enabled",
                    base_cfg.platform_event_gate_enabled,
                )
            )
        ),
        round(
            float(
                merged.get(
                    "platform_event_min_score",
                    base_cfg.platform_event_min_score,
                )
            ),
            6,
        ),
        float(
            bool(
                merged.get(
                    "drawdown_cooldown_enabled",
                    base_cfg.drawdown_cooldown_enabled,
                )
            )
        ),
        round(
            float(
                merged.get(
                    "cooldown_drawdown",
                    base_cfg.cooldown_drawdown,
                )
            ),
            6,
        ),
        int(
            merged.get(
                "cooldown_loss_streak",
                base_cfg.cooldown_loss_streak,
            )
        ),
        int(merged.get("cooldown_bars", base_cfg.cooldown_bars)),
        round(
            float(
                merged.get(
                    "trend_gate_min_efficiency",
                    base_cfg.trend_gate_min_efficiency,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "range_gate_max_efficiency",
                    base_cfg.range_gate_max_efficiency,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "range_reversion_min_score",
                    base_cfg.range_reversion_min_score,
                )
            ),
            6,
        ),
        str(
            merged.get(
                "strategy_archetype_policy",
                base_cfg.strategy_archetype_policy,
            )
        ),
        str(
            merged.get(
                "volatility_regime_policy",
                base_cfg.volatility_regime_policy,
            )
        ),
        int(
            merged.get(
                "volatility_regime_lookback",
                base_cfg.volatility_regime_lookback,
            )
        ),
        round(
            float(
                merged.get(
                    "volatility_regime_low_quantile",
                    base_cfg.volatility_regime_low_quantile,
                )
            ),
            6,
        ),
        round(
            float(
                merged.get(
                    "volatility_regime_high_quantile",
                    base_cfg.volatility_regime_high_quantile,
                )
            ),
            6,
        ),
    )


def _resolve_profile_params(
    base_cfg: BacktestConfig,
    raw_profile: dict[str, Any],
) -> dict[str, Any]:
    fields = set(BacktestConfig.__dataclass_fields__)
    params: dict[str, Any] = {}
    base_fields = raw_profile.get("base", [])
    if not isinstance(base_fields, list):
        raise ValueError("risk profile base must be an array")
    for field_name in base_fields:
        name = str(field_name).strip()
        if name not in fields:
            raise ValueError(f"unknown BacktestConfig field: {name}")
        if name in params:
            raise ValueError(f"duplicate risk profile field: {name}")
        params[name] = getattr(base_cfg, name)

    for rule in ("values", "min", "max"):
        raw_values = raw_profile.get(rule, {})
        if not isinstance(raw_values, dict):
            raise ValueError(f"risk profile {rule} must be a table")
        for field_name, configured_value in raw_values.items():
            name = str(field_name).strip()
            if name not in fields:
                raise ValueError(f"unknown BacktestConfig field: {name}")
            if name in params:
                raise ValueError(f"duplicate risk profile field: {name}")
            if rule == "values":
                params[name] = configured_value
                continue
            base_value = getattr(base_cfg, name)
            if (
                isinstance(base_value, bool)
                or not isinstance(base_value, (int, float))
                or isinstance(configured_value, bool)
                or not isinstance(configured_value, (int, float))
            ):
                raise ValueError(
                    f"risk profile {rule} requires numeric field: {name}"
                )
            if rule == "min":
                params[name] = min(base_value, configured_value)
            else:
                params[name] = max(base_value, configured_value)

    BacktestConfig(**{**asdict(base_cfg), **params})
    return params


def resolve_risk_profile_catalog(
    base_cfg: BacktestConfig,
    *,
    compact: bool = False,
    path: str | Path | None = None,
) -> ResolvedRiskProfileCatalog:
    """Load, validate, resolve, and de-duplicate the risk-profile catalog."""

    source_path = Path(path or DEFAULT_RISK_PROFILE_CATALOG_PATH).resolve()
    with source_path.open("rb") as handle:
        payload = tomllib.load(handle)
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != RISK_PROFILE_CATALOG_SCHEMA_VERSION:
        raise ValueError(
            "unsupported risk profile catalog schema_version: "
            f"{schema_version}"
        )
    catalog_version = str(payload.get("catalog_version", "")).strip()
    if not catalog_version:
        raise ValueError("risk profile catalog_version is required")
    compact_order_raw = payload.get("compact_order", [])
    if not isinstance(compact_order_raw, list):
        raise ValueError("risk profile compact_order must be an array")
    compact_order = tuple(
        str(item).strip() for item in compact_order_raw if str(item).strip()
    )
    if len(compact_order) != len(set(compact_order)):
        raise ValueError("risk profile compact_order contains duplicates")

    raw_profiles = payload.get("profiles", [])
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("risk profile catalog requires profiles")
    resolved: list[dict[str, Any]] = []
    names: set[str] = set()
    allowed_profile_keys = {"name", "base", "values", "min", "max"}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise ValueError("risk profile entry must be a table")
        unknown_keys = set(raw_profile) - allowed_profile_keys
        if unknown_keys:
            raise ValueError(
                "unknown risk profile keys: "
                + ", ".join(sorted(unknown_keys))
            )
        name = str(raw_profile.get("name", "")).strip()
        if not name:
            raise ValueError("risk profile name is required")
        if name in names:
            raise ValueError(f"duplicate risk profile name: {name}")
        names.add(name)
        resolved.append(
            {
                "name": name,
                "params": _resolve_profile_params(base_cfg, raw_profile),
            }
        )
    missing_compact = set(compact_order) - names
    if missing_compact:
        raise ValueError(
            "compact risk profiles missing from catalog: "
            + ", ".join(sorted(missing_compact))
        )

    unique: dict[
        tuple[float | int | str, ...],
        dict[str, Any],
    ] = {}
    for profile in resolved:
        identity = _profile_identity(base_cfg, profile["params"])
        unique[identity] = profile
    profiles = list(unique.values())
    if compact:
        by_name = {str(item["name"]): item for item in profiles}
        profiles = [
            by_name[name] for name in compact_order if name in by_name
        ]
    return ResolvedRiskProfileCatalog(
        schema_version=schema_version,
        catalog_version=catalog_version,
        source_path=source_path,
        compact_order=compact_order,
        profiles=tuple(profiles),
    )
