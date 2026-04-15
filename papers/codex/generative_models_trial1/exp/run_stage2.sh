#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

. .venv/bin/activate
export PYTHONPATH=.

mkdir -p \
  exp/generate_cache/logs \
  exp/single_sample/logs \
  exp/best_of_4_global_siglip/logs \
  exp/detector_only_structured/logs \
  exp/crop_structured_siglip/logs \
  exp/daam_no_counterfactual/logs \
  exp/assign_and_verify/logs \
  exp/ablations/logs \
  exp/analyze/logs \
  exp/assignment_quality_manual/logs

python exp/prepare_data/run.py > exp/generate_cache/logs/prepare_data.log 2>&1

python -m exp.shared.run_core --mode generate --split dev --k 4 > exp/generate_cache/logs/dev_generate.log 2>&1
python -m exp.shared.run_core --mode generate --split test --k 4 > exp/generate_cache/logs/test_generate.log 2>&1
python -m exp.shared.run_core --mode generate --split transfer --k 4 > exp/generate_cache/logs/transfer_generate.log 2>&1
python -m exp.shared.run_core --mode generate --split candidate_budget --k 8 > exp/generate_cache/logs/candidate_budget_generate.log 2>&1

python exp/single_sample/run.py --split dev > exp/single_sample/logs/dev.log 2>&1
python exp/single_sample/run.py --split test > exp/single_sample/logs/test.log 2>&1
python exp/single_sample/run.py --split transfer > exp/single_sample/logs/transfer.log 2>&1
python exp/single_sample/run.py --split candidate_budget > exp/single_sample/logs/candidate_budget.log 2>&1

python exp/best_of_4_global_siglip/run.py --split dev --k 4 > exp/best_of_4_global_siglip/logs/dev.log 2>&1
python exp/best_of_4_global_siglip/run.py --split test --k 4 > exp/best_of_4_global_siglip/logs/test.log 2>&1
python exp/best_of_4_global_siglip/run.py --split transfer --k 4 > exp/best_of_4_global_siglip/logs/transfer.log 2>&1
python exp/best_of_4_global_siglip/run.py --split candidate_budget --k 8 > exp/best_of_4_global_siglip/logs/candidate_budget.log 2>&1

python exp/detector_only_structured/run.py --split dev --k 4 > exp/detector_only_structured/logs/dev.log 2>&1
python exp/detector_only_structured/run.py --split test --k 4 > exp/detector_only_structured/logs/test.log 2>&1
python exp/detector_only_structured/run.py --split transfer --k 4 > exp/detector_only_structured/logs/transfer.log 2>&1
python exp/detector_only_structured/run.py --split candidate_budget --k 8 > exp/detector_only_structured/logs/candidate_budget.log 2>&1

python exp/crop_structured_siglip/run.py --split dev --k 4 > exp/crop_structured_siglip/logs/dev.log 2>&1
python exp/crop_structured_siglip/run.py --split test --k 4 > exp/crop_structured_siglip/logs/test.log 2>&1
python exp/crop_structured_siglip/run.py --split transfer --k 4 > exp/crop_structured_siglip/logs/transfer.log 2>&1
python exp/crop_structured_siglip/run.py --split candidate_budget --k 8 > exp/crop_structured_siglip/logs/candidate_budget.log 2>&1

python exp/daam_no_counterfactual/run.py --split dev --k 4 > exp/daam_no_counterfactual/logs/dev.log 2>&1
python exp/daam_no_counterfactual/run.py --split test --k 4 > exp/daam_no_counterfactual/logs/test.log 2>&1
python exp/daam_no_counterfactual/run.py --split transfer --k 4 > exp/daam_no_counterfactual/logs/transfer.log 2>&1
python exp/daam_no_counterfactual/run.py --split candidate_budget --k 8 > exp/daam_no_counterfactual/logs/candidate_budget.log 2>&1

python exp/assign_and_verify/run.py --split dev --k 4 > exp/assign_and_verify/logs/dev.log 2>&1
python exp/assign_and_verify/run.py --split test --k 4 > exp/assign_and_verify/logs/test.log 2>&1
python exp/assign_and_verify/run.py --split transfer --k 4 > exp/assign_and_verify/logs/transfer.log 2>&1
python exp/assign_and_verify/run.py --split candidate_budget --k 8 > exp/assign_and_verify/logs/candidate_budget.log 2>&1

python exp/ablations/run.py > exp/ablations/logs/test_and_candidate_budget.log 2>&1
python exp/assignment_quality_manual/run.py > exp/assignment_quality_manual/logs/manifest.log 2>&1
python exp/analyze/run.py > exp/analyze/logs/analyze.log 2>&1
