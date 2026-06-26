#include "aco_config.h"

#include <stdlib.h>

double aco_parse_min_rel_improvement_percent(const char *s,
                                             double default_fraction) {
  if (!s || !*s) {
    return default_fraction;
  }

  double percent = atof(s);
  if (percent <= 0.0) {
    return default_fraction;
  }
  return percent / 100.0;
}

void aco_runtime_config_defaults(AcoRuntimeConfig *config) {
  if (!config) {
    return;
  }

  config->timeout_seconds = 0.0;
  config->stagnation_epochs = 0;
  config->min_rel_improvement = 1e-3;
  config->fixed_epochs = 0;
  config->candidate_k = 0;
  config->ants = 0;
  config->seed = 1234u;
  config->progress_interval_seconds = 10.0;
  config->log_level = 1;
  config->reproducibility_mode = 0;
}

void aco_runtime_config_load_env(AcoRuntimeConfig *config) {
  const char *s_timeout;
  const char *s_stagnation;
  const char *s_rel;
  const char *s_fixed;
  const char *s_progress;
  const char *s_candidate_k;
  const char *s_ants;
  const char *s_log_level;
  const char *s_reproducibility;

  if (!config) {
    return;
  }

  aco_runtime_config_defaults(config);

  s_timeout = getenv("ACO_SOLVER_TIMEOUT_SECONDS");
  s_stagnation = getenv("ACO_SOLVER_STAGNATION_EPOCHS");
  s_rel = getenv("ACO_SOLVER_MIN_REL_IMPROVEMENT");
  s_fixed = getenv("ACO_SOLVER_FIXED_EPOCHS");
  s_progress = getenv("ACO_SOLVER_PROGRESS_INTERVAL_SECONDS");
  s_candidate_k = getenv("ACO_SOLVER_CANDIDATE_K");
  s_ants = getenv("ACO_SOLVER_ANTS");
  s_log_level = getenv("ACO_SOLVER_LOG_LEVEL");
  s_reproducibility = getenv("ACO_SOLVER_REPRODUCIBILITY_MODE");

  config->timeout_seconds = (s_timeout && *s_timeout) ? atof(s_timeout) : 0.0;
  config->stagnation_epochs =
      (s_stagnation && *s_stagnation) ? atoi(s_stagnation) : 0;
  config->min_rel_improvement =
      aco_parse_min_rel_improvement_percent(s_rel, 1e-3);
  config->fixed_epochs = (s_fixed && *s_fixed) ? atoi(s_fixed) : 0;
  config->progress_interval_seconds =
      (s_progress && *s_progress) ? atof(s_progress) : 10.0;
  config->candidate_k = (s_candidate_k && *s_candidate_k)
                            ? atoi(s_candidate_k)
                            : config->candidate_k;
  config->ants = (s_ants && *s_ants) ? atoi(s_ants) : config->ants;
  config->log_level =
      (s_log_level && *s_log_level) ? atoi(s_log_level) : config->log_level;
  config->reproducibility_mode =
      (s_reproducibility && *s_reproducibility)
          ? atoi(s_reproducibility)
          : config->reproducibility_mode;

  if (config->stagnation_epochs < 0) {
    config->stagnation_epochs = 0;
  }
  if (config->min_rel_improvement <= 0.0) {
    config->min_rel_improvement = 1e-3;
  }
  if (config->fixed_epochs < 0) {
    config->fixed_epochs = 0;
  }
  if (config->progress_interval_seconds < 0.0) {
    config->progress_interval_seconds = 0.0;
  }
  if (config->candidate_k < 0) {
    config->candidate_k = 0;
  }
  if (config->ants < 0) {
    config->ants = 0;
  }
  if (config->log_level < 0) {
    config->log_level = 0;
  }
  if (config->reproducibility_mode < 0) {
    config->reproducibility_mode = 0;
  }
}
