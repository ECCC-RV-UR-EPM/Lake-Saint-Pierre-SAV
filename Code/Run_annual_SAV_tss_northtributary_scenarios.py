from __future__ import annotations

from pathlib import Path

from north_tributary_scenarios_common import run_north_tributary_scenarios

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "Data"

SCENARIOS = {
    "TSS_northtrib_plus25": 1.25,
    "TSS_northtrib_minus25": 0.75,
    "TSS_northtrib_minus50": 0.50,
}


def main() -> None:
    run_north_tributary_scenarios(
        variable="TSS",
        pred_col="TSS_pred",
        filled_input_dir=DATA_ROOT / "02_tss" / "TSS_prediction_LSP" / "Prediction" / "TSS_prediction_filled_parquet",
        channel_model_path=DATA_ROOT / "02_tss" / "TSS_prediction_LSP" / "Final_Model_TSS_noTotal" / "model_channel.pkl",
        nonchannel_model_path=DATA_ROOT / "02_tss" / "TSS_prediction_LSP" / "Final_Model_TSS_noTotal" / "model_nonchannel.pkl",
        scaled_prefixes=["In_TSS_"],
        scenarios=SCENARIOS,
        output_prefix="tss_northtrib_scenario",
    )


if __name__ == "__main__":
    main()
