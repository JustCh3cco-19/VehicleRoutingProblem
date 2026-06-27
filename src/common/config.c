#include "config.h"
#include <stdlib.h>

double	parse_min_rel_improvement_percent(const char *s,
		double default_fraction)
{
	double	percent;

	if (!s || !*s)
		return (default_fraction);
	percent = atof(s);
	if (percent <= 0.0)
		return (default_fraction);
	return (percent / 100.0);
}

void	runtime_config_defaults(t_config *config)
{
	if (!config)
		return ;
	config->timeout_seconds = 300.0;
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

static void	aco_runtime_config_load_env_mats(t_config *config,
		const char **vars)
{
	config->timeout_seconds = (vars[0] && *vars[0]) ? atof(vars[0]) : 300.0;
	config->stagnation_epochs = (vars[1] && *vars[1]) ? atoi(vars[1]) : 0;
	config->min_rel_improvement = parse_min_rel_improvement_percent(
			vars[2], 1e-3);
	config->fixed_epochs = (vars[3] && *vars[3]) ? atoi(vars[3]) : 0;
	config->progress_interval_seconds = (vars[4] && *vars[4]) ? atof(vars[4])
		: 10.0;
	config->candidate_k = (vars[5] && *vars[5]) ? atoi(vars[5])
		: config->candidate_k;
	config->ants = (vars[6] && *vars[6]) ? atoi(vars[6]) : config->ants;
	config->log_level = (vars[7] && *vars[7]) ? atoi(vars[7])
		: config->log_level;
	config->reproducibility_mode = (vars[8] && *vars[8]) ? atoi(vars[8])
		: config->reproducibility_mode;
}

static void	runtime_config_read_env(const char **vars)
{
	vars[0] = getenv("SOLVER_TIMEOUT_SECONDS");
	vars[1] = getenv("SOLVER_STAGNATION_EPOCHS");
	vars[2] = getenv("SOLVER_MIN_REL_IMPROVEMENT");
	vars[3] = getenv("SOLVER_FIXED_EPOCHS");
	vars[4] = getenv("SOLVER_PROGRESS_INTERVAL_SECONDS");
	vars[5] = getenv("SOLVER_CANDIDATE_K");
	vars[6] = getenv("SOLVER_ANTS");
	vars[7] = getenv("SOLVER_LOG_LEVEL");
	vars[8] = getenv("SOLVER_REPRODUCIBILITY_MODE");
}

static void	runtime_config_clamp(t_config *config)
{
	if (config->stagnation_epochs < 0)
		config->stagnation_epochs = 0;
	if (config->min_rel_improvement <= 0.0)
		config->min_rel_improvement = 1e-3;
	if (config->fixed_epochs < 0)
		config->fixed_epochs = 0;
	if (config->progress_interval_seconds < 0.0)
		config->progress_interval_seconds = 0.0;
	if (config->candidate_k < 0)
		config->candidate_k = 0;
	if (config->ants < 0)
		config->ants = 0;
	if (config->log_level < 0)
		config->log_level = 0;
	if (config->reproducibility_mode < 0)
		config->reproducibility_mode = 0;
}

void	runtime_config_load_env(t_config *config)
{
	const char	*vars[9];

	if (!config)
		return ;
	runtime_config_defaults(config);
	runtime_config_read_env(vars);
	aco_runtime_config_load_env_mats(config, vars);
	runtime_config_clamp(config);
}
