#ifndef CONFIG_H
# define CONFIG_H

struct s_aco_config
{
	double					timeout_seconds;
	int						stagnation_epochs;
	double					min_rel_improvement;
	int						fixed_epochs;
	int						candidate_k;
	int						ants;
	unsigned int			seed;
	double					progress_interval_seconds;
	int						log_level;
	int						reproducibility_mode;
};
typedef struct s_aco_config	t_aco_config;

enum e_aco_log_levels
{
	ACO_LOG_SILENT = 0,
	ACO_LOG_PROGRESS = 1
};

enum e_aco_repro_modes
{
	ACO_REPRO_STATISTICAL = 0,
	ACO_REPRO_DETERMINISTIC_BENCHMARK = 1
};

void				aco_runtime_config_defaults(t_aco_config *config);
void				aco_runtime_config_load_env(t_aco_config *config);
double				aco_parse_min_rel_improvement_percent(const char *s,
						double default_fraction);

/* Compatibility Typedefs for Non-Refactored Types */
typedef t_aco_config		AcoRuntimeConfig;

#endif
