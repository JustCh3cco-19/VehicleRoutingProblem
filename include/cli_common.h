#ifndef CLI_COMMON_H
# define CLI_COMMON_H

# include "solution.h"

# ifdef __cplusplus
extern "C" {
# endif

enum e_aco_cli_mode
{
	CLI_MODE_DEMO = 0,
	CLI_MODE_INSTANCE = 1
};
typedef enum e_aco_cli_mode			t_cli_mode;

struct s_aco_cli_options
{
	t_cli_mode			mode;
	const char				*instance_path;
	int						n;
	int						k;
	int						m;
	unsigned int			seed;
	double					alpha;
	double					beta;
	double					rho;
	double					tau0;
	double					q;
	double					best_cost;
};
typedef struct s_aco_cli_options	t_cli_options;

int							cli_parse_int_arg(const char *s, int *out);
unsigned int				cli_parse_uint_arg(const char *s, int *ok);
void						cli_options_defaults(t_cli_options *options);
int							cli_parse_solver_options(int argc, char **argv,
								t_cli_options *options);
void						cli_print_usage(const char *program_name);
void						cli_print_solution_routes(const t_solution *best,
								int k);
void						cli_print_solution_cost(double best_cost);
int							cli_validate_solution_or_report(
								const t_solution *best, int n, int k,
								const int *demands, int vehicle_capacity,
								double best_cost);

# ifdef __cplusplus
}
# endif

#endif
