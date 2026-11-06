"""Run a staged joint-fit driver (R51f to R51i, R66 to R69) on the cluster gateway; the
generalization of orchestrate_gateway_r51f.py (which ran R51f), with the
run chosen by --run.  The replica count per class (8 through R51h, 32 for
R51i) sets the slot count per generation, and the block-parallel R51i
evaluation takes a whole worker (32 tasks) per branch.

Every driver step runs on the gateway, in the remote checkout, under
REMOTE_PYTHON: the pre-registration (1 task), generation (one job per slot,
1 task each, all open branches at once), evaluation (one job per floor
branch, EVAL_TASKS tasks, branches concurrently), the profile, gate, clean
validation and seal.  The run dir lives on the cluster, so every record's
run_dir identity and every digest is computed there; the sealed dir is
fetched back into the local artifact store at the end.

The orchestrator is resumable: it reads the remote run dir before every
phase and keeps a local ledger of submitted jobs, so a restart never
duplicates a job whose output exists or whose job is still alive.  A LOST
verdict (the shared-filesystem heartbeat stalls of R44/R45) triggers a
liveness probe of the job's wrapper process on its worker: a live wrapper
is waited for, and a job is resubmitted only when it FAILED, or when it is
LOST, its wrapper is gone, and its output is missing.  (The first R51f
evaluation round resubmitted a live LOST job after a fixed grace time; the
duplicate was killed before it wrote anything, 2026-09-30 13:57.)

Usage: python3 helpers/benchmarking/orchestrate_gateway_joint_fit.py --run r51g [--sync-only] [--no-sync]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# a submission with wait_for_slot blocks in the remote agent until a slot
# frees; the client's default 60 s ssh bound cut such a call once the R51i
# generation (512 slots) filled the cluster, after the remote submit had
# already gone through (2026-10-01 13:03)
os.environ.setdefault("NA49_GW_TIMEOUT", "14400")

from helpers.na49_gateway.cluster_client import ClusterGatewayClient, _default_agent_command  # noqa: E402
from helpers.na49_gateway.gateway_config import get_config_list, get_config_str, load_user_config  # noqa: E402

RUNS = {
    "r51f": {"run_id": "r51f_joint_fit_2026-09-30", "driver": "helpers/benchmarking/staged_r51f_joint_fit.py",
             "gate": "gate5_followup_r51f.json",
             "extra_files": ("dev/00runs/r51e_topology_fit_2026-09-29/report.json",
                             "dev/00runs/r51e_topology_fit_2026-09-29/seal_manifest.json")},
    "r51g": {"run_id": "r51g_joint_fit_2026-09-30", "driver": "helpers/benchmarking/staged_r51g_joint_fit.py",
             "gate": "gate5_followup_r51g.json",
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r51f_joint_fit_2026-09-30/report.json",
                             "dev/00runs/r51f_joint_fit_2026-09-30/seal_manifest.json")},
    "r51h": {"run_id": "r51h_joint_fit_2026-10-01", "driver": "helpers/benchmarking/staged_r51h_joint_fit.py",
             "gate": "gate5_followup_r51h.json",
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r51g_joint_fit_2026-09-30/report.json",
                             "dev/00runs/r51g_joint_fit_2026-09-30/seal_manifest.json")},
    "r51i": {"run_id": "r51i_joint_fit_2026-10-01", "driver": "helpers/benchmarking/staged_r51i_joint_fit.py",
             "gate": "gate5_followup_r51i.json", "n_floors": 4, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r51h_joint_fit_2026-10-01/report.json",
                             "dev/00runs/r51h_joint_fit_2026-10-01/seal_manifest.json")},
    "r66": {"run_id": "r66_joint_flavor_fit_2026-10-02", "driver": "helpers/benchmarking/staged_r66_joint_flavor_fit.py",
            "gate": "gate6_flavor_r66.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r51i_joint_fit_2026-10-01/report.json",
                            "dev/00runs/r51i_joint_fit_2026-10-01/seal_manifest.json")},
    "r67": {"run_id": "r67_central_component_test_2026-10-02", "driver": "helpers/benchmarking/staged_r67_central_component_test.py",
            "gate": "gate6_central_component_r67.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r51i_joint_fit_2026-10-01/report.json",
                            "dev/00runs/r51i_joint_fit_2026-10-01/seal_manifest.json",
                            "dev/00runs/r66_joint_flavor_fit_2026-10-02/report.json",
                            "dev/00runs/r66_joint_flavor_fit_2026-10-02/seal_manifest.json")},
    "r68": {"run_id": "r68_central_component_uncapped_2026-10-02", "driver": "helpers/benchmarking/staged_r68_central_component_uncapped.py",
            "gate": "gate6_central_component_r68.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r66_joint_flavor_fit_2026-10-02/report.json",
                            "dev/00runs/r66_joint_flavor_fit_2026-10-02/seal_manifest.json",
                            "dev/00runs/r67_central_component_test_2026-10-02/report.json",
                            "dev/00runs/r67_central_component_test_2026-10-02/seal_manifest.json")},
    "r69": {"run_id": "r69_flavor_central_uncapped_2026-10-02", "driver": "helpers/benchmarking/staged_r69_flavor_central_uncapped.py",
            "gate": "gate6_flavor_central_r69.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r66_joint_flavor_fit_2026-10-02/report.json",
                            "dev/00runs/r66_joint_flavor_fit_2026-10-02/seal_manifest.json")},
    "r70": {"run_id": "r70_flavor_central_convolved_2026-10-02", "driver": "helpers/benchmarking/staged_r70_flavor_central_convolved.py",
            "gate": "gate6_flavor_central_r70.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r69_flavor_central_uncapped_2026-10-02/branch_report_f0.json",
                            "dev/00runs/r69_flavor_central_uncapped_2026-10-02/seal_manifest.json")},
    "r71": {"run_id": "r71_central_component_fixed_kernel_2026-10-03", "driver": "helpers/benchmarking/staged_r71_central_component_fixed_kernel.py",
            "gate": "gate6_central_component_r71.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r70_flavor_central_convolved_2026-10-02/branch_report_f0.json",
                            "dev/00runs/r70_flavor_central_convolved_2026-10-02/seal_manifest.json")},
    "r73": {"run_id": "r73_central_component_density_step_2026-10-03", "driver": "helpers/benchmarking/staged_r73_central_component_density_step.py",
            "gate": "gate6_central_component_r73.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r71_central_component_fixed_kernel_2026-10-03/branch_report_f0.json",
                            "dev/00runs/r71_central_component_fixed_kernel_2026-10-03/seal_manifest.json")},
    "r77": {"run_id": "r77_single_fit_2026-10-06", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r77.json",
            "gate": "gate8_single_fit_r77.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r75_charge_sensitive_fit_2026-10-04/branch_report_f0.json",
                            "dev/00runs/r75_charge_sensitive_fit_2026-10-04/seal_manifest.json")},
    "r81": {"run_id": "r81_central_power_law_2026-10-07", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r81.json",
            "gate": "gate8_central_power_law_r81.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r80_central_on_calibrated_2026-10-07/branch_report_f0.json",
                            "dev/00runs/r80_central_on_calibrated_2026-10-07/seal_manifest.json")},
    "r108": {"run_id": "r108_systematics_2026-10-22", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r108.json",
             "gate": "gate8_systematics_r108.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r107_pair_s_wave_2026-10-22/branch_report_f0.json",
                             "dev/00runs/r107_pair_s_wave_2026-10-22/seal_manifest.json")},
    "r109": {"run_id": "r109_sd_window_2026-10-23", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r109.json",
             "gate": "gate8_sd_window_r109.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r108_systematics_2026-10-22/branch_report_f0.json",
                             "dev/00runs/r108_systematics_2026-10-22/seal_manifest.json")},
    "r110": {"run_id": "r110_sd_window_replica_2026-10-23", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r110.json",
             "gate": "gate8_sd_window_replica_r110.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r109_sd_window_2026-10-23/branch_report_f0.json",
                             "dev/00runs/r109_sd_window_2026-10-23/seal_manifest.json")},
    "r111": {"run_id": "r111_coherence_2026-10-23", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r111.json",
             "gate": "gate8_coherence_r111.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r110_sd_window_replica_2026-10-23/branch_report_f0.json",
                             "dev/00runs/r110_sd_window_replica_2026-10-23/seal_manifest.json")},
    "r112": {"run_id": "r112_sd_tail_2026-10-24", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r112.json",
             "gate": "gate8_sd_tail_r112.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r111_coherence_2026-10-23/branch_report_f0.json",
                             "dev/00runs/r111_coherence_2026-10-23/seal_manifest.json")},
    "r113": {"run_id": "r113_sd_tail_2026-10-24", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r113.json",
             "gate": "gate8_sd_tail_r113.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r111_coherence_2026-10-23/branch_report_f0.json",
                             "dev/00runs/r111_coherence_2026-10-23/seal_manifest.json")},
    "r114": {"run_id": "r114_sd_tail_replica_2026-10-24", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r114.json",
             "gate": "gate8_sd_tail_replica_r114.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r113_sd_tail_2026-10-24/branch_report_f0.json",
                             "dev/00runs/r113_sd_tail_2026-10-24/seal_manifest.json")},
    "r115": {"run_id": "r115_intact_t_break_2026-10-24", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r115.json",
             "gate": "gate8_intact_t_break_r115.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r114_sd_tail_replica_2026-10-24/branch_report_f0.json",
                             "dev/00runs/r114_sd_tail_replica_2026-10-24/seal_manifest.json")},
    "r141": {"run_id": "r141_aligned_lines_2026-11-05", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r141.json",
             "gate": "gate8_aligned_lines_r141.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r140_symmetric_remnants_2026-11-05/branch_report_f0.json",
                             "dev/00runs/r140_symmetric_remnants_2026-11-05/seal_manifest.json")},
    "r140": {"run_id": "r140_symmetric_remnants_2026-11-05", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r140.json",
             "gate": "gate8_symmetric_remnants_r140.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r139_symmetric_lines_2026-11-04/branch_report_f0.json",
                             "dev/00runs/r139_symmetric_lines_2026-11-04/seal_manifest.json")},
    "r139": {"run_id": "r139_symmetric_lines_2026-11-04", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r139.json",
             "gate": "gate8_symmetric_lines_r139.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r138_counted_pairs_2026-11-04/branch_report_f0.json",
                             "dev/00runs/r138_counted_pairs_2026-11-04/seal_manifest.json")},
    "r138": {"run_id": "r138_counted_pairs_2026-11-04", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r138.json",
             "gate": "gate8_counted_pairs_r138.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r137_fourteen_parameters_2026-11-03/branch_report_f0.json",
                             "dev/00runs/r137_fourteen_parameters_2026-11-03/seal_manifest.json")},
    "r137": {"run_id": "r137_fourteen_parameters_2026-11-03", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r137.json",
             "gate": "gate8_fourteen_parameters_r137.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r136_fixed_choices_2026-11-03/branch_report_f0.json",
                             "dev/00runs/r136_fixed_choices_2026-11-03/seal_manifest.json")},
    "r136": {"run_id": "r136_fixed_choices_2026-11-03", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r136.json",
             "gate": "gate8_fixed_choices_r136.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r135_sd_intact_protons_2026-11-02/branch_report_f0.json",
                             "dev/00runs/r135_sd_intact_protons_2026-11-02/seal_manifest.json")},
    "r135": {"run_id": "r135_sd_intact_protons_2026-11-02", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r135.json",
             "gate": "gate8_sd_intact_protons_r135.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r133_pomeron_shrinkage_2026-11-01/branch_report_f0.json",
                             "dev/00runs/r133_pomeron_shrinkage_2026-11-01/seal_manifest.json")},
    "r133": {"run_id": "r133_pomeron_shrinkage_2026-11-01", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r133.json",
             "gate": "gate8_pomeron_shrinkage_r133.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r132_epsilon_dl_2026-10-31/branch_report_f0.json",
                             "dev/00runs/r132_epsilon_dl_2026-10-31/seal_manifest.json")},
    "r134": {"run_id": "r134_pomeron_shrinkage_eps_free_2026-11-01", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r134.json",
             "gate": "gate8_pomeron_shrinkage_eps_free_r134.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r132_epsilon_dl_2026-10-31/branch_report_f0.json",
                             "dev/00runs/r132_epsilon_dl_2026-10-31/seal_manifest.json")},
    "r132": {"run_id": "r132_epsilon_dl_2026-10-31", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r132.json",
             "gate": "gate8_epsilon_dl_r132.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r131_epsilon_free_2026-10-31/branch_report_f0.json",
                             "dev/00runs/r131_epsilon_free_2026-10-31/seal_manifest.json")},
    "r131": {"run_id": "r131_epsilon_free_2026-10-31", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r131.json",
             "gate": "gate8_epsilon_free_r131.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r130_xi_tower_2026-10-30/branch_report_f0.json",
                             "dev/00runs/r130_xi_tower_2026-10-30/seal_manifest.json")},
    "r130": {"run_id": "r130_xi_tower_2026-10-30", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r130.json",
             "gate": "gate8_xi_tower_r130.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r129_pdg_elastic_profile_2026-10-30/branch_report_f0.json",
                             "dev/00runs/r129_pdg_elastic_profile_2026-10-30/seal_manifest.json")},
    "r129": {"run_id": "r129_pdg_elastic_profile_2026-10-30", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r129.json",
             "gate": "gate8_pdg_elastic_profile_r129.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r128_kaon_identification_knots_2026-10-28/branch_report_f0.json",
                             "dev/00runs/r128_kaon_identification_knots_2026-10-28/seal_manifest.json")},
    "r128": {"run_id": "r128_kaon_identification_knots_2026-10-28", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r128.json",
             "gate": "gate8_kaon_identification_knots_r128.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r127_excited_kaons_2026-10-28/branch_report_f0.json",
                             "dev/00runs/r127_excited_kaons_2026-10-28/seal_manifest.json")},
    "r127": {"run_id": "r127_excited_kaons_2026-10-28", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r127.json",
             "gate": "gate8_excited_kaons_r127.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r126_correlated_errors_replica_2026-10-28/branch_report_f0.json",
                             "dev/00runs/r126_correlated_errors_replica_2026-10-28/seal_manifest.json")},
    "r126": {"run_id": "r126_correlated_errors_replica_2026-10-28", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r126.json",
             "gate": "gate8_correlated_errors_replica_r126.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r125_correlated_errors_2026-10-28/branch_report_f0.json",
                             "dev/00runs/r125_correlated_errors_2026-10-28/seal_manifest.json")},
    "r125": {"run_id": "r125_correlated_errors_2026-10-28", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r125.json",
             "gate": "gate8_correlated_errors_r125.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r124_ope_absorption_2026-10-27/branch_report_f0.json",
                             "dev/00runs/r124_ope_absorption_2026-10-27/seal_manifest.json")},
    "r124": {"run_id": "r124_ope_absorption_2026-10-27", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r124.json",
             "gate": "gate8_ope_absorption_r124.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r123_ope_neutral_partner_2026-10-27/branch_report_f0.json",
                             "dev/00runs/r123_ope_neutral_partner_2026-10-27/seal_manifest.json")},
    "r123": {"run_id": "r123_ope_neutral_partner_2026-10-27", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r123.json",
             "gate": "gate8_ope_neutral_partner_r123.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r122_ope_proton_form_factor_2026-10-27/branch_report_f0.json",
                             "dev/00runs/r122_ope_proton_form_factor_2026-10-27/seal_manifest.json")},
    "r122": {"run_id": "r122_ope_proton_form_factor_2026-10-27", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r122.json",
             "gate": "gate8_ope_proton_form_factor_r122.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r121_one_pion_exchange_2026-10-26/branch_report_f0.json",
                             "dev/00runs/r121_one_pion_exchange_2026-10-26/seal_manifest.json")},
    "r121": {"run_id": "r121_one_pion_exchange_2026-10-26", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r121.json",
             "gate": "gate8_one_pion_exchange_r121.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r120_epsilon_fixed_2026-10-26/branch_report_f0.json",
                             "dev/00runs/r120_epsilon_fixed_2026-10-26/seal_manifest.json")},
    "r120": {"run_id": "r120_epsilon_fixed_2026-10-26", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r120.json",
             "gate": "gate8_epsilon_fixed_r120.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r119_tail_edge_replica_2026-10-26/branch_report_f0.json",
                             "dev/00runs/r119_tail_edge_replica_2026-10-26/seal_manifest.json")},
    "r119": {"run_id": "r119_tail_edge_replica_2026-10-26", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r119.json",
             "gate": "gate8_tail_edge_replica_r119.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r118_tail_edge_2026-10-25/branch_report_f0.json",
                             "dev/00runs/r118_tail_edge_2026-10-25/seal_manifest.json")},
    "r118": {"run_id": "r118_tail_edge_2026-10-25", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r118.json",
             "gate": "gate8_tail_edge_r118.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r117_bbar_scope_replica_2026-10-25/branch_report_f0.json",
                             "dev/00runs/r117_bbar_scope_replica_2026-10-25/seal_manifest.json")},
    "r117": {"run_id": "r117_bbar_scope_replica_2026-10-25", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r117.json",
             "gate": "gate8_bbar_scope_replica_r117.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r116_bbar_scope_2026-10-25/branch_report_f0.json",
                             "dev/00runs/r116_bbar_scope_2026-10-25/seal_manifest.json")},
    "r116": {"run_id": "r116_bbar_scope_2026-10-25", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r116.json",
             "gate": "gate8_bbar_scope_r116.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r115_intact_t_break_2026-10-24/branch_report_f0.json",
                             "dev/00runs/r115_intact_t_break_2026-10-24/seal_manifest.json")},
    "r107": {"run_id": "r107_pair_s_wave_2026-10-22", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r107.json",
             "gate": "gate8_pair_s_wave_r107.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r106_baryon_tables_2026-10-21/branch_report_f0.json",
                             "dev/00runs/r106_baryon_tables_2026-10-21/seal_manifest.json")},
    "r106": {"run_id": "r106_baryon_tables_2026-10-21", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r106.json",
             "gate": "gate8_baryon_tables_r106.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r105_completions_2026-10-21/branch_report_f0.json",
                             "dev/00runs/r105_completions_2026-10-21/seal_manifest.json")},
    "r105": {"run_id": "r105_completions_2026-10-21", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r105.json",
             "gate": "gate8_completions_r105.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r104_dalitz_2026-10-20/branch_report_f0.json",
                             "dev/00runs/r104_dalitz_2026-10-20/seal_manifest.json")},
    "r104": {"run_id": "r104_dalitz_2026-10-20", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r104.json",
             "gate": "gate8_dalitz_r104.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r103_redraw_shapes_2026-10-20/branch_report_f0.json",
                             "dev/00runs/r103_redraw_shapes_2026-10-20/seal_manifest.json")},
    "r103": {"run_id": "r103_redraw_shapes_2026-10-20", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r103.json",
             "gate": "gate8_redraw_shapes_r103.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r102_duality_shapes_2026-10-18/branch_report_f0.json",
                             "dev/00runs/r102_duality_shapes_2026-10-18/seal_manifest.json")},
    "r102": {"run_id": "r102_duality_shapes_2026-10-18", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r102.json",
             "gate": "gate8_duality_shapes_r102.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r100_pdg_tables_2026-10-17/branch_report_f0.json",
                             "dev/00runs/r100_pdg_tables_2026-10-17/seal_manifest.json")},
    "r101": {"run_id": "r101_factors_free_2026-10-18", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r101.json",
             "gate": "gate8_factors_free_r101.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r100_pdg_tables_2026-10-17/branch_report_f0.json",
                             "dev/00runs/r100_pdg_tables_2026-10-17/seal_manifest.json")},
    "r100": {"run_id": "r100_pdg_tables_2026-10-17", "driver": "helpers/benchmarking/run_engine.py",
             "spec": "dev/mixing2006/config/runs/r100.json",
             "gate": "gate8_pdg_tables_r100.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
             "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
             "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
             "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                             "dev/00runs/r99_xi_review_2026-10-16/branch_report_f0.json",
                             "dev/00runs/r99_xi_review_2026-10-16/seal_manifest.json")},
    "r99": {"run_id": "r99_xi_review_2026-10-16", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r99.json",
            "gate": "gate8_xi_review_r99.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r98_completion_2026-10-15/branch_report_f0.json",
                            "dev/00runs/r98_completion_2026-10-15/seal_manifest.json")},
    "r98": {"run_id": "r98_completion_2026-10-15", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r98.json",
            "gate": "gate8_completion_r98.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r97_intact_vertex_2026-10-15/branch_report_f0.json",
                            "dev/00runs/r97_intact_vertex_2026-10-15/seal_manifest.json")},
    "r97": {"run_id": "r97_intact_vertex_2026-10-15", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r97.json",
            "gate": "gate8_intact_vertex_r97.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r96_free_rv_2026-10-13/branch_report_f0.json",
                            "dev/00runs/r96_free_rv_2026-10-13/seal_manifest.json")},
    "r96": {"run_id": "r96_free_rv_2026-10-13", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r96.json",
            "gate": "gate8_free_rv_r96.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r95_no_low_mass_kind_2026-10-13/branch_report_f0.json",
                            "dev/00runs/r95_no_low_mass_kind_2026-10-13/seal_manifest.json")},
    "r95": {"run_id": "r95_no_low_mass_kind_2026-10-13", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r95.json",
            "gate": "gate8_no_low_mass_kind_r95.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r94_nd_duality_2026-10-12/branch_report_f0.json",
                            "dev/00runs/r94_nd_duality_2026-10-12/seal_manifest.json")},
    "r94": {"run_id": "r94_nd_duality_2026-10-12", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r94.json",
            "gate": "gate8_nd_duality_r94.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r93_nucleon_kstar_2026-10-12/branch_report_f0.json",
                            "dev/00runs/r93_nucleon_kstar_2026-10-12/seal_manifest.json")},
    "r93": {"run_id": "r93_nucleon_kstar_2026-10-12", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r93.json",
            "gate": "gate8_nucleon_kstar_r93.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r92_light_scalars_2026-10-12/branch_report_f0.json",
                            "dev/00runs/r92_light_scalars_2026-10-12/seal_manifest.json")},
    "r92": {"run_id": "r92_light_scalars_2026-10-12", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r92.json",
            "gate": "gate8_light_scalars_r92.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r91_meson_split_2026-10-11/branch_report_f0.json",
                            "dev/00runs/r91_meson_split_2026-10-11/seal_manifest.json")},
    "r91": {"run_id": "r91_meson_split_2026-10-11", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r91.json",
            "gate": "gate8_meson_split_r91.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r90_nd_isospin_mixture_2026-10-10/branch_report_f0.json",
                            "dev/00runs/r90_nd_isospin_mixture_2026-10-10/seal_manifest.json")},
    "r90": {"run_id": "r90_nd_isospin_mixture_2026-10-10", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r90.json",
            "gate": "gate8_nd_isospin_mixture_r90.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r89_pairs_step_minimum_2026-10-10/branch_report_f0.json",
                            "dev/00runs/r89_pairs_step_minimum_2026-10-10/seal_manifest.json")},
    "r89": {"run_id": "r89_pairs_step_minimum_2026-10-10", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r89.json",
            "gate": "gate8_pairs_step_minimum_r89.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r88_conjugate_towers_2026-10-09/branch_report_f0.json",
                            "dev/00runs/r88_conjugate_towers_2026-10-09/seal_manifest.json")},
    "r88": {"run_id": "r88_conjugate_towers_2026-10-09", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r88.json",
            "gate": "gate8_conjugate_towers_r88.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r87_sd_intact_2026-10-09/branch_report_f0.json",
                            "dev/00runs/r87_sd_intact_2026-10-09/seal_manifest.json")},
    "r87": {"run_id": "r87_sd_intact_2026-10-09", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r87.json",
            "gate": "gate8_sd_intact_r87.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r86_floor_pools_2026-10-09/branch_report_f0.json",
                            "dev/00runs/r86_floor_pools_2026-10-09/seal_manifest.json")},
    "r86": {"run_id": "r86_floor_pools_2026-10-09", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r86.json",
            "gate": "gate8_floor_pools_r86.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r85_floor2_2026-10-09/branch_report_f0.json",
                            "dev/00runs/r85_floor2_2026-10-09/seal_manifest.json")},
    "r85": {"run_id": "r85_floor2_2026-10-09", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r85.json",
            "gate": "gate8_floor2_r85.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r84_isoscalar_dipion_2026-10-08/branch_report_f0.json",
                            "dev/00runs/r84_isoscalar_dipion_2026-10-08/seal_manifest.json")},
    "r84": {"run_id": "r84_isoscalar_dipion_2026-10-08", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r84.json",
            "gate": "gate8_isoscalar_dipion_r84.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r83_vertex_slopes_2026-10-08/branch_report_f0.json",
                            "dev/00runs/r83_vertex_slopes_2026-10-08/seal_manifest.json")},
    "r83": {"run_id": "r83_vertex_slopes_2026-10-08", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r83.json",
            "gate": "gate8_vertex_slopes_r83.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r82_leading_split_2026-10-08/branch_report_f0.json",
                            "dev/00runs/r82_leading_split_2026-10-08/seal_manifest.json")},
    "r82": {"run_id": "r82_leading_split_2026-10-08", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r82.json",
            "gate": "gate8_leading_split_r82.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r78e_edge_control_2026-10-07/branch_report_f0.json",
                            "dev/00runs/r78e_edge_control_2026-10-07/seal_manifest.json")},
    "r80": {"run_id": "r80_central_on_calibrated_2026-10-07", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r80.json",
            "gate": "gate8_central_on_calibrated_r80.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r79e_calibrated_scale_edge_2026-10-07/branch_report_f0.json",
                            "dev/00runs/r79e_calibrated_scale_edge_2026-10-07/seal_manifest.json")},
    "r79": {"run_id": "r79_calibrated_scale_2026-10-07", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r79.json",
            "gate": "gate8_calibrated_scale_r79.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r78e_edge_control_2026-10-07/branch_report_f0.json",
                            "dev/00runs/r78e_edge_control_2026-10-07/seal_manifest.json")},
    "r79e": {"run_id": "r79e_calibrated_scale_edge_2026-10-07", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r79e.json",
            "gate": "gate8_calibrated_scale_edge_r79e.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r78e_edge_control_2026-10-07/branch_report_f0.json",
                            "dev/00runs/r78e_edge_control_2026-10-07/seal_manifest.json")},
    "r78": {"run_id": "r78_corrected_fit_2026-10-07", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r78.json",
            "gate": "gate8_corrected_fit_r78.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r77_single_fit_2026-10-06/branch_report_f0.json",
                            "dev/00runs/r77_single_fit_2026-10-06/seal_manifest.json")},
    "r78e": {"run_id": "r78e_edge_control_2026-10-07", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r78e.json",
            "gate": "gate8_edge_control_r78e.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"), "inactive_classes": ("nd",),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r77_single_fit_2026-10-06/branch_report_f0.json",
                            "dev/00runs/r77_single_fit_2026-10-06/seal_manifest.json")},
    "r76": {"run_id": "r76_isovector_slope_2026-10-04", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r76.json",
            "gate": "gate8_isovector_slope_r76.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r75_charge_sensitive_fit_2026-10-04/branch_report_f0.json",
                            "dev/00runs/r75_charge_sensitive_fit_2026-10-04/seal_manifest.json")},
    "r75": {"run_id": "r75_charge_sensitive_fit_2026-10-04", "driver": "helpers/benchmarking/run_engine.py",
            "spec": "dev/mixing2006/config/runs/r75.json",
            "gate": "gate8_charge_sensitive_r75.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r74_central_component_barrier_step_2026-10-03/branch_report_f0.json",
                            "dev/00runs/r74_central_component_barrier_step_2026-10-03/seal_manifest.json")},
    "r74replay": {"run_id": "r74_engine_replay_2026-10-04", "driver": "helpers/benchmarking/run_engine.py",
                  "spec": "dev/mixing2006/config/runs/r74_engine_replay.json", "stop_after": "profile",
                  "gate": "gate6_central_component_r74.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
                  "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
                  "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
                  "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                                  "dev/00runs/r73_central_component_density_step_2026-10-03/branch_report_f0.json",
                                  "dev/00runs/r73_central_component_density_step_2026-10-03/seal_manifest.json")},
    "r74": {"run_id": "r74_central_component_barrier_step_2026-10-03", "driver": "helpers/benchmarking/staged_r74_central_component_barrier_step.py",
            "gate": "gate6_central_component_r74.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r73_central_component_density_step_2026-10-03/branch_report_f0.json",
                            "dev/00runs/r73_central_component_density_step_2026-10-03/seal_manifest.json")},
    "r72": {"run_id": "r72_fixed_kernel_r70_shapes_2026-10-03", "driver": "helpers/benchmarking/staged_r72_fixed_kernel_r70_shapes.py",
            "gate": "gate6_central_component_r72.json", "n_floors": 1, "replicas": 32, "eval_tasks": 32, "profile_tasks": 32,
            "classes": ("sd_projectile", "sd_target", "dd", "nd", "nd0"),
            "fanout": True, "fanout_tasks": 32, "gen_tasks": 4,
            "extra_files": ("dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
                            "dev/00runs/r70_flavor_central_convolved_2026-10-02/branch_report_f0.json",
                            "dev/00runs/r70_flavor_central_convolved_2026-10-02/seal_manifest.json")},
}
RUN_ID = ""
DRIVER_REL = ""
#: the run spec of an engine run (helpers/benchmarking/run_engine.py) and the
#: argument every driver command carries for it ("" for a staged driver)
SPEC_REL = ""
DRIVER_ARGS = ""
GATE_NAME = ""
REMOTE_PYTHON = "/usr/bin/python3.12"
N_FLOORS = 3
REPLICAS = 8
CLASSES = ("sd_projectile", "sd_target", "dd", "nd")
N_SLOTS = len(CLASSES) * REPLICAS
#: the slots the run generates: every slot, or the slots of the classes the
#: run entry does not list as inactive (the engine's two-component mode keeps
#: the five-class slot numbering and generates no nd slot, 2026-10-06)
ACTIVE_SLOT_IDS: list = []
MAX_GENERATIONS = 4
#: Task slots per worker (2026-11-05, the owner: "you can go up to 64 per node, it has hyperthreading"): the workers
#: are 32-core EPYC 7543P nodes with two threads per core and 250 GB, and a 32-task fit job holds about 19 GB, so two
#: fit jobs share a node. At 32 the 72-job profile and clean fan-outs ran in six waves of twelve (R139: 155 and 146 min
#: of its 9.3 h); at 64 they run in three of twenty-four. The jobs' results do not depend on their placement.
MAX_TASKS_PER_WORKER = 64
EVAL_TASKS = 24
PROFILE_TASKS = 8
POLL_SECONDS = 120
MAX_RESUBMITS = 2
LEDGER = Path()
#: Fan-out (2026-10-02, the owner: "i dont see much running on the
#: gateway"): a run whose spec sets ``"fanout": True`` runs its evaluation,
#: profile and clean-validation steps in rounds; each round's step either
#: writes its output or a pending record naming fit jobs, which run as
#: separate gateway jobs of FANOUT_TASKS tasks (see fit_fanout.py).
FANOUT = False
FANOUT_TASKS = 32
MAX_FANOUT_ROUNDS = 12
#: Tasks per generation job: a driver with chunked generation splits the
#: slot's events over them (1 keeps one process per slot).
GEN_TASKS = 1

SYNC_DIRS = ("helpers/benchmarking", "dev/mixing2006/na49_mixing", "dev/mixing2006/helpers", "dev/mixing2006/config")
SYNC_FILES = (
    "dev/mixing2006/pyproject.toml",
    *(f"dev/mixing2006/spshadrons.web.cern.ch/data/{name}" for name in (
        "xsect_piminus.txt", "xsect_piplus.txt", "xsect_kaplus_pp.txt", "xsect_kaminus_pp.txt",
        "xsect_proton_pp.txt", "xsect_antiproton_pp.txt")),
)
RSYNC_EXCLUDES = ("__pycache__", "*.pyc", ".pytest_cache")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False).stdout.strip()


class Gateway:
    def __init__(self) -> None:
        config = load_user_config()
        self.host = get_config_str(config, "host")
        self.ssh_options = tuple(get_config_list(config, "ssh_options") or [])
        self.remote_root = "~/" + get_config_str(config, "remote_project_relpath")
        self.client = ClusterGatewayClient(host=self.host, agent_command=_default_agent_command(config),
                                           ssh_options=self.ssh_options, jobs_root=get_config_str(config, "jobs_root"),
                                           stale_heartbeat_seconds=3600, launch_grace_seconds=14400)
        self.run_dir = f"{self.remote_root}/dev/00runs/{RUN_ID}"
        # the remote agent has no config of its own: the worker pool comes from
        # the local per-user config with every submission
        self.workers = get_config_list(config, "workers") or []
        if not self.workers:
            raise RuntimeError("no workers in the per-user gateway config")

    # -- ssh helpers ---------------------------------------------------
    def ssh(self, command: str, *, input_text: str | None = None, check: bool = True) -> str:
        proc = subprocess.run(["ssh", *self.ssh_options, self.host, command], input=input_text, text=True,
                              capture_output=True, check=False, timeout=600)
        if check and proc.returncode != 0:
            raise RuntimeError(f"ssh {command!r} failed: {proc.stderr.strip()}")
        return proc.stdout

    def remote_python(self, script: str) -> str:
        return self.ssh("python3 -", input_text=script)

    def remote_files(self) -> set[str]:
        out = self.ssh(f"ls -1 {self.run_dir} 2>/dev/null || true", check=True)
        return {line.strip() for line in out.splitlines() if line.strip()}

    def job_alive(self, job_id: str, worker: str | None) -> bool | None:
        """True when the job's wrapper process runs on its worker, False when
        it is gone, None when the probe itself fails (never a resubmit)."""
        if not worker:
            return None
        probe = f"ssh -o BatchMode=yes {worker} 'ps -eo args | grep -c \"[/]{job_id}/run.sh\"'"
        proc = subprocess.run(["ssh", *self.ssh_options, self.host, probe], text=True, capture_output=True,
                              check=False, timeout=120)
        out = proc.stdout.strip().splitlines()
        if not out or not out[-1].isdigit():
            return None
        return int(out[-1]) > 0

    def remote_json(self, name: str) -> dict | None:
        """A remote run-dir record, or None when absent; a partial read (a
        record being written) is retried before it raises."""
        for attempt in range(3):
            out = self.ssh(f"cat {self.run_dir}/{name} 2>/dev/null || true")
            if not out.strip():
                return None
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                if attempt == 2:
                    raise
                time.sleep(5)
        return None

    # -- sync ----------------------------------------------------------
    def sync(self) -> dict:
        ssh_cmd = " ".join(shlex.quote(t) for t in ["ssh", *self.ssh_options])
        excludes = [f"--exclude={e}" for e in RSYNC_EXCLUDES]
        for rel in SYNC_DIRS:
            self.ssh(f"mkdir -p {self.remote_root}/{rel}")
            subprocess.run(["rsync", "-az", "--delete", *excludes, "-e", ssh_cmd, f"{REPO_ROOT / rel}/",
                            f"{self.host}:{self.remote_root}/{rel}/"], check=True)
            log(f"sync: {rel}/")
        for rel in SYNC_FILES + tuple(RUNS_EXTRA):
            parent = rel.rsplit("/", 1)[0]
            self.ssh(f"mkdir -p {self.remote_root}/{parent}")
            subprocess.run(["rsync", "-az", "-e", ssh_cmd, str((REPO_ROOT / rel).resolve()),
                            f"{self.host}:{self.remote_root}/{rel}"], check=True)
            log(f"sync: {rel}")
        # the digests the driver freezes, plus the driver and the data tables
        spec_rels = list(dict.fromkeys(list(_frozen_rels()) + [DRIVER_REL] + [r for r in SYNC_FILES + tuple(RUNS_EXTRA) if "/data/" in r]))
        local = {rel: sha256((REPO_ROOT / rel).resolve()) for rel in spec_rels}
        manifest = {
            "schema": "na49_mixing.gateway_sync/1", "run": RUN_ID, "synced_utc": now(),
            "source": {"outer": {"head": git(REPO_ROOT, "rev-parse", "HEAD"), "dirty_paths": git(REPO_ROOT, "status", "--porcelain").splitlines()},
                       "nested": {"head": git(REPO_ROOT / "dev" / "mixing2006", "rev-parse", "HEAD"),
                                  "dirty_paths": git(REPO_ROOT / "dev" / "mixing2006", "status", "--porcelain").splitlines()}},
            "targets": {"dirs": list(SYNC_DIRS), "files": list(SYNC_FILES) + list(RUNS_EXTRA)},
            "remote_python": REMOTE_PYTHON, "sha256": local,
        }
        self.ssh(f"cat > {self.remote_root}/SYNC_MANIFEST.json", input_text=json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        script = ("import hashlib, json, os\nroot = os.path.expanduser(%r)\nrels = %r\nout = {}\n"
                  "for r in rels:\n    p = os.path.join(root, r)\n"
                  "    out[r] = hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.isfile(p) else None\n"
                  "print(json.dumps(out))\n") % (self.remote_root, spec_rels)
        remote = json.loads(self.remote_python(script))
        bad = {r: (local[r][:12], (remote.get(r) or "missing")[:12]) for r in spec_rels if remote.get(r) != local[r]}
        if bad:
            raise RuntimeError(f"remote digests differ after sync: {bad}")
        log(f"sync verified: {len(spec_rels)} digests match (outer {manifest['source']['outer']['head'][:7]}, "
            f"nested {manifest['source']['nested']['head'][:7]})")
        return manifest


def _frozen_rels() -> tuple:
    return frozen_rels_of(REPO_ROOT / DRIVER_REL, SPEC_REL)


# ----------------------------------------------------------------------
# The local job ledger (2026-10-02, external reviews g1, m1, q1)
# ----------------------------------------------------------------------
#: The legacy ledger directory: a session-specific path of the agent that
#: wrote the first orchestrators.  Read once for migration, never written.
LEGACY_LEDGER_DIR = Path.home() / ".claude" / "jobs" / "a8313650" / "tmp"


def ledger_path(run: str) -> Path:
    """The persistent per-user ledger of a run's submitted jobs:
    ``$NA49_GW_LEDGER_DIR`` or ``~/.local/state/na49_gateway``.  A ledger
    that exists only at the legacy session path is copied over once, so a
    resumed run never re-submits a live job."""
    base = Path(os.environ.get("NA49_GW_LEDGER_DIR") or (Path.home() / ".local" / "state" / "na49_gateway"))
    path = base / f"{run}_gateway_jobs.json"
    legacy = LEGACY_LEDGER_DIR / f"{run}_gateway_jobs.json"
    if not path.is_file() and legacy.is_file():
        base.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, path)
        log(f"ledger migrated from {legacy} to {path}")
    return path


def read_ledger(path: Path) -> dict:
    """The ledger state, or {} for a new run; a ledger that does not parse
    is refused (starting empty would re-submit live jobs)."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"the job ledger {path} does not parse ({exc}); repair or move it before resuming") from exc


def write_ledger(path: Path, state: dict) -> None:
    """Write the ledger atomically (a temporary file, then a rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def frozen_rels_of(driver_path: Path, spec_rel: str = "") -> tuple:
    """The driver's ``FROZEN_RELS`` literal, read with ``ast`` (no driver
    import and no ``exec`` on the orchestrator side).  The run engine
    defines ``BASE_FROZEN_RELS`` and adds itself, the run spec and the
    spec's ``frozen_extra`` when it configures (2026-10-04)."""
    tree = ast.parse(Path(driver_path).read_text())
    literals = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in ("FROZEN_RELS", "BASE_FROZEN_RELS"):
                    literals[t.id] = tuple(ast.literal_eval(node.value))
    if "BASE_FROZEN_RELS" in literals:
        if not spec_rel:
            raise RuntimeError(f"{driver_path} is the run engine; the run entry needs a spec")
        spec = json.loads((REPO_ROOT / spec_rel).read_text())
        driver_rel = Path(driver_path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        return literals["BASE_FROZEN_RELS"] + (driver_rel, spec_rel) + tuple(spec.get("frozen_extra", ()))
    if "FROZEN_RELS" in literals:
        return literals["FROZEN_RELS"]
    raise RuntimeError(f"{driver_path} defines no FROZEN_RELS literal")


class Jobs:
    """The local ledger of submitted jobs, keyed by the expected output."""

    def __init__(self, gw: Gateway) -> None:
        self.gw = gw
        self.state = read_ledger(LEDGER)

    def save(self) -> None:
        write_ledger(LEDGER, self.state)

    def ensure(self, key: str, args: str, tasks: int, name: str) -> None:
        """Submit the driver step unless a live or finished job exists for it."""
        rec = self.state.get(key)
        if rec and rec.get("state") not in ("FAILED", "SUBMIT_FAILED", "RESUBMIT"):
            return
        attempts = (rec or {}).get("attempts", 0)
        if attempts > MAX_RESUBMITS:
            raise RuntimeError(f"{key}: {attempts} attempts failed; see the job logs")
        cmd = " ".join(part for part in (REMOTE_PYTHON, DRIVER_REL, DRIVER_ARGS, args) if part)
        job = self.gw.client.submit(cmd, workdir=self.gw.remote_root, name=name, tasks=tasks, workers=self.gw.workers,
                                    max_tasks_per_worker=MAX_TASKS_PER_WORKER, wait_for_slot=True, poll_seconds=10.0)
        self.state[key] = {"job_id": job["job_id"], "cmd": cmd, "tasks": tasks, "submitted_utc": now(),
                           "submitted_epoch": time.time(), "state": job.get("state", "SUBMITTED"), "attempts": attempts + 1}
        self.save()
        log(f"submitted {name} ({tasks} tasks): {job['job_id']}")

    def wait_for(self, expected: dict, label: str) -> None:
        """Wait until every key's expected output file exists in the remote run dir
        (an entry may be a tuple of alternative outputs; any one of them counts)."""
        while True:
            try:
                files = self.gw.remote_files()
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                # a shared-filesystem stall of the run-dir listing is retried
                log(f"{label}: run-dir listing failed ({exc}); retrying")
                time.sleep(POLL_SECONDS)
                continue
            missing = {k: f for k, f in expected.items() if not present(f, files)}
            if not missing:
                log(f"{label}: all {len(expected)} outputs present")
                return
            try:
                listing = {j["job_id"]: j for j in self.gw.client.list(limit=2000)}
            except Exception as exc:  # noqa: BLE001 - an ssh hiccup is retried at the next poll
                log(f"{label}: job listing failed ({exc}); retrying")
                time.sleep(POLL_SECONDS)
                continue
            for key in list(missing):
                rec = self.state.get(key)
                if not rec:
                    continue
                if rec["job_id"] not in listing:
                    # outside the listing window: ask for the job directly
                    try:
                        listing[rec["job_id"]] = self.gw.client.status(rec["job_id"])
                    except Exception as exc:  # noqa: BLE001 - retried at the next poll
                        log(f"{key}: status of {rec['job_id']} failed ({exc}); retrying")
                        continue
                st = listing[rec["job_id"]].get("state")
                rec["state"] = st
                if st == "FAILED" or st == "SUBMIT_FAILED" or st == "CANCELED":
                    tail = self.gw.client.result(rec["job_id"], tail_lines=20)
                    err = tail.get("stderr_tail") or []
                    err = err if isinstance(err, list) else str(err).splitlines()
                    log(f"{key}: {st} (exit {tail.get('exit_code')}); stderr tail: {' | '.join(err[-5:])}")
                    rec["state"] = "RESUBMIT"
                elif st == "COMPLETED":
                    # a shared-filesystem listing may lag the write; after ten
                    # minutes a completed job without its output is a logic
                    # error, never a resubmission
                    seen = rec.setdefault("completed_seen", time.time())
                    if time.time() - seen > 600:
                        self.save()
                        raise RuntimeError(f"{key}: COMPLETED without its output {missing[key]}")
                elif st == "LOST":
                    alive = self.gw.job_alive(rec["job_id"], listing[rec["job_id"]].get("worker"))
                    if alive is False:
                        log(f"{key}: LOST, its wrapper is gone and its output is missing; resubmitting")
                        rec["state"] = "RESUBMIT"
                    elif alive:
                        rec["state"] = "LOST_ALIVE"
                        if time.time() - rec.get("alive_logged", 0) > 3600:
                            log(f"{key}: reported LOST, but its wrapper runs on {listing[rec['job_id']].get('worker')}; waiting")
                            rec["alive_logged"] = time.time()
            self.save()
            resub = [k for k in missing if (self.state.get(k) or {}).get("state") == "RESUBMIT"]
            for key in resub:
                rec = self.state[key]
                tail = rec["cmd"].split(DRIVER_REL, 1)[1].strip()
                if DRIVER_ARGS and tail.startswith(DRIVER_ARGS):
                    tail = tail[len(DRIVER_ARGS):].strip()
                self.ensure(key, tail, rec["tasks"], key)
            log(f"{label}: {len(expected) - len(missing)}/{len(expected)} outputs present")
            time.sleep(POLL_SECONDS)


def present(expected, files) -> bool:
    """True when the expected output (a name, or a tuple of alternatives)
    is in the remote listing."""
    names = expected if isinstance(expected, tuple) else (expected,)
    return any(name in files for name in names)


def run_steps(gw, jobs: Jobs, steps: list, label: str) -> None:
    """Run driver steps ``(key, args, tasks, output)`` to their outputs.

    Without fan-out, one job per step.  With fan-out, in rounds: round r
    submits each unfinished step with ``--fanout gateway --fanout-key
    <key>_r<r>``; the step writes its output or the pending record
    ``fanout_pending_<key>_r<r>.json``; the pending fit jobs then run as
    one gateway job each, and the next round re-enters the step.  Every
    round's record keeps its own name, so a restarted orchestrator replays
    the finished rounds without resubmitting them."""
    if not FANOUT:
        files = gw.remote_files()
        expected = {}
        for key, args, tasks, out in steps:
            expected[key] = out
            if out not in files:
                jobs.ensure(key, args, tasks, key)
        jobs.wait_for(expected, label)
        return
    for rnd in range(MAX_FANOUT_ROUNDS):
        files = gw.remote_files()
        active = [s for s in steps if s[3] not in files]
        if not active:
            return
        expected = {}
        for key, args, tasks, out in active:
            rkey = f"{key}_r{rnd}"
            pending = f"fanout_pending_{rkey}.json"
            expected[rkey] = (out, pending)
            if pending not in files:
                jobs.ensure(rkey, f"{args} --fanout gateway --fanout-key {rkey}", tasks, rkey)
        jobs.wait_for(expected, f"{label} round {rnd}")
        files = gw.remote_files()
        fan = {}
        for key, args, tasks, out in active:
            if out in files:
                continue
            record = gw.remote_json(f"fanout_pending_{key}_r{rnd}.json")
            if not record or not record.get("batches"):
                raise RuntimeError(f"{key} round {rnd}: neither the output {out} nor a usable pending record")
            for batch in record["batches"]:
                ns, bid = batch["namespace"], batch["batch"]
                for i in batch["missing"]:
                    tkey = f"fan_{ns}_{bid}_{i}"
                    fan[tkey] = f"fanout_{ns}_{bid}_{i}.json"
                    if fan[tkey] not in files:
                        jobs.ensure(tkey, f"--fanout-task {ns} {bid} {i}", FANOUT_TASKS, tkey)
        if fan:
            jobs.wait_for(fan, f"{label} fan-out {rnd} ({len(fan)} jobs)")
    raise RuntimeError(f"{label}: still unfinished after {MAX_FANOUT_ROUNDS} fan-out rounds")


def slot_name(f: int, g: int, s: int) -> str:
    cls = CLASSES[s // REPLICAS]
    return f"slot_{cls}_{s % REPLICAS}_f{f}_g{g}.json"


def points(gw: Gateway, f: int) -> list[dict]:
    doc = gw.remote_json(f"generation_points_f{f}.json")
    return doc["points"] if doc else []


RUNS_EXTRA: list = []


def main() -> int:
    global RUN_ID, DRIVER_REL, SPEC_REL, DRIVER_ARGS, GATE_NAME, LEDGER, N_FLOORS, REPLICAS, CLASSES, N_SLOTS, EVAL_TASKS, PROFILE_TASKS, FANOUT, FANOUT_TASKS, GEN_TASKS, ACTIVE_SLOT_IDS
    parser = argparse.ArgumentParser(description="a staged joint-fit driver on the cluster gateway")
    parser.add_argument("--run", required=True, choices=sorted(RUNS))
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--no-sync", action="store_true", help="skip the code sync (resume after a local restart)")
    args = parser.parse_args()
    spec = RUNS[args.run]
    RUN_ID, DRIVER_REL, GATE_NAME = spec["run_id"], spec["driver"], spec["gate"]
    SPEC_REL = spec.get("spec", "")
    DRIVER_ARGS = f"--spec {SPEC_REL}" if SPEC_REL else ""
    N_FLOORS = int(spec.get("n_floors", 3))
    REPLICAS = int(spec.get("replicas", 8))
    CLASSES = tuple(spec.get("classes", CLASSES))
    N_SLOTS = len(CLASSES) * REPLICAS
    inactive = tuple(spec.get("inactive_classes", ()))
    ACTIVE_SLOT_IDS = [s for s in range(N_SLOTS) if CLASSES[s // REPLICAS] not in inactive]
    EVAL_TASKS = int(spec.get("eval_tasks", EVAL_TASKS))
    PROFILE_TASKS = int(spec.get("profile_tasks", PROFILE_TASKS))
    FANOUT = bool(spec.get("fanout", False))
    FANOUT_TASKS = int(spec.get("fanout_tasks", FANOUT_TASKS))
    GEN_TASKS = int(spec.get("gen_tasks", GEN_TASKS))
    RUNS_EXTRA[:] = list(spec["extra_files"]) + ([SPEC_REL] if SPEC_REL else [])
    LEDGER = ledger_path(args.run)
    gw = Gateway()
    jobs = Jobs(gw)
    files = gw.remote_files()
    if not args.no_sync and "pre_registration.json" not in files:
        gw.sync()
    elif not args.no_sync:
        log("the remote pre-registration exists; the code is frozen for this run, so no re-sync")
    if args.sync_only:
        return 0
    if "pre_registration.json" not in files:
        jobs.ensure("prereg", "--pre-registration", 1, f"{args.run}_prereg")
        jobs.wait_for({"prereg": "pre_registration.json"}, "pre-registration")
    for rnd in range(MAX_GENERATIONS):
        files = gw.remote_files()
        open_branches = [f for f in range(N_FLOORS) if f"branch_report_f{f}.json" not in files]
        if not open_branches:
            break
        # generation: the latest registered generation of every open branch
        expected = {}
        for f in open_branches:
            pts = points(gw, f)
            if not pts:
                raise RuntimeError(f"branch {f} has no registered generation point in the remote run dir")
            g = max(int(p["generation"]) for p in pts)
            for s in ACTIVE_SLOT_IDS:
                name = slot_name(f, g, s)
                key = f"gen_f{f}_g{g}_s{s}"
                expected[key] = name
                if name not in files:
                    jobs.ensure(key, f"--generate --floor-index {f} --generation {g} --slot {s} --workers {GEN_TASKS}", GEN_TASKS, key)
        jobs.wait_for(expected, f"generation round {rnd}")
        # evaluation: one job per open branch whose latest complete generation
        # has no fit report yet, concurrently (a generation registered by an
        # evaluation that finished during this round is generated next round)
        steps = []
        files = gw.remote_files()
        for f in open_branches:
            if f"branch_report_f{f}.json" in files:
                continue
            complete = [int(p["generation"]) for p in points(gw, f) if all(slot_name(f, int(p["generation"]), s) in files for s in ACTIVE_SLOT_IDS)]
            if not complete:
                continue
            g = max(complete)
            steps.append((f"eval_f{f}_g{g}", f"--evaluate --floor-index {f}", EVAL_TASKS, f"fit_report_f{f}_g{g}.json"))
        run_steps(gw, jobs, steps, f"evaluation round {rnd}")
        files = gw.remote_files()
        for f in open_branches:
            state = "final" if f"branch_report_f{f}.json" in files else "regeneration requested"
            log(f"branch {f}: {state} after round {rnd}")
    files = gw.remote_files()
    if any(f"branch_report_f{f}.json" not in files for f in range(N_FLOORS)):
        raise RuntimeError("a branch still requests regeneration after the last generation")
    for key, args_, tasks, out, fanned in (("profile", "--profile", PROFILE_TASKS, "report.json", True),
                                           ("gate", "--gate", 1, GATE_NAME, False),
                                           ("clean", "--clean-validate", EVAL_TASKS, "clean_environment_commands.json", True),
                                           ("seal", "--seal", 1, "seal_manifest.json", False)):
        if out in gw.remote_files():
            continue
        if fanned and FANOUT:
            run_steps(gw, jobs, [(f"{args.run}_{key}", args_, tasks, out)], key)
        else:
            jobs.ensure(key, args_, tasks, f"{args.run}_{key}")
            jobs.wait_for({key: out}, key)
        if spec.get("stop_after") == key:
            log(f"stopping after {key} (the run entry's stop_after; no check, clean validation or seal)")
            break
    # fetch the sealed run dir into the local artifact store
    local = REPO_ROOT / "dev" / "00runs" / RUN_ID
    local.mkdir(parents=True, exist_ok=True)
    ssh_cmd = " ".join(shlex.quote(t) for t in ["ssh", *gw.ssh_options])
    subprocess.run(["rsync", "-az", "-e", ssh_cmd, f"{gw.host}:{gw.run_dir}/", f"{local}/"], check=True)
    log(f"fetched the sealed run dir to {local}")
    print("ALL DONE", now(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
