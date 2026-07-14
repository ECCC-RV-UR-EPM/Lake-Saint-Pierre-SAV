from __future__ import annotations

from pathlib import Path

from north_tributary_scenarios_common import run_north_tributary_scenarios

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "Data"

SCENARIOS = {
    "TP_northtrib_plus25": 1.25,
    "TP_northtrib_minus25": 0.75,
    "TP_northtrib_minus50": 0.50,
}


def main() -> None:
    run_north_tributary_scenarios(
        variable="TP",
        pred_col="TP_pred",
        filled_input_dir=DATA_ROOT / "03_tp_tn" / "TP_prediction_LSP" / "Prediction" / "TP_prediction_filled_parquet",
        channel_model_path=DATA_ROOT / "03_tp_tn" / "TP_prediction_LSP" / "Final_Model_TP_noTotal" / "model_channel.pkl",
        nonchannel_model_path=DATA_ROOT / "03_tp_tn" / "TP_prediction_LSP" / "Final_Model_TP_noTotal" / "model_nonchannel.pkl",
        scaled_prefixes=["In_TP_"],
        scenarios=SCENARIOS,
        output_prefix="tp_northtrib_scenario",
    )


if __name__ == "__main__":
    main()
