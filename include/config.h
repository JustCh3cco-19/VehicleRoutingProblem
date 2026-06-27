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
typedef struct s_aco_config	t_config;

enum e_aco_log_levels
{
	LOG_SILENT = 0,
	LOG_PROGRESS = 1
};

enum e_aco_repro_modes
{
	REPRO_STATISTICAL = 0,
	REPRO_DETERMINISTIC_BENCHMARK = 1
};

void				runtime_config_defaults(t_config *config);
void				runtime_config_load_env(t_config *config);
double				parse_min_rel_improvement_percent(const char *s,
						double default_fraction);

#endif
