from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from sav_annual_common import MAP_DIR, OUT_DIR, panel, run_scenarios

SCENARIOS = {"baseline": 1.0, "TP_plus25": 1.25, "TP_minus25": 0.75, "TP_minus50": 0.5}


def main() -> None:
    area_df, all_maps = run_scenarios("TP_pred", SCENARIOS)
    area_df.to_csv(OUT_DIR / "tp_scenario_area_summary.csv", index=False)

    base = area_df[area_df["Scenario"] == "baseline"][["Year", "Total_km2", "North_km2"]].rename(
        columns={"Total_km2": "base_total", "North_km2": "base_north"}
    )
    delta = []
    for scen in [s for s in SCENARIOS if s != "baseline"]:
        sub = area_df[area_df["Scenario"] == scen][["Year", "Total_km2", "North_km2"]].merge(base, on="Year")
        sub["Scenario"] = scen
        sub["Delta_total_km2"] = sub["Total_km2"] - sub["base_total"]
        sub["Delta_north_km2"] = sub["North_km2"] - sub["base_north"]
        delta.append(sub[["Scenario", "Year", "Delta_total_km2", "Delta_north_km2"]])
    pd.concat(delta, ignore_index=True).to_csv(OUT_DIR / "tp_scenario_delta_vs_baseline.csv", index=False)

    for name, maps in all_maps.items():
        panel(maps, MAP_DIR / f"SAV_ALL_YEARS_{name}.png", f"Annual SAV {name}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for scen in SCENARIOS:
        sub = area_df[area_df["Scenario"] == scen].sort_values("Year")
        axes[0].plot(sub["Year"], sub["Total_km2"], marker="o", label=scen)
        axes[1].plot(sub["Year"], sub["North_km2"], marker="o", label=scen)
    axes[0].set_ylabel("Total SAV area (km$^2$)")
    axes[1].set_ylabel("North SAV area (km$^2$)")
    axes[1].set_xlabel("Year")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tp_scenario_timeseries.png", dpi=300)
    plt.close(fig)
    print(area_df[area_df["Year"] == 2024].to_string(index=False))


if __name__ == "__main__":
    main()
