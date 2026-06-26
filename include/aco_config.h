#ifndef ACO_CONFIG_H
#define ACO_CONFIG_H

typedef struct {
  double timeout_seconds;
  int stagnation_epochs;
  double min_rel_improvement;
  int fixed_epochs;
  int candidate_k;
  int ants;
  unsigned int seed;
  double progress_interval_seconds;
} AcoRuntimeConfig;

void aco_runtime_config_defaults(AcoRuntimeConfig *config);
void aco_runtime_config_load_env(AcoRuntimeConfig *config);
double aco_parse_min_rel_improvement_percent(const char *s,
                                             double default_fraction);

#endif
