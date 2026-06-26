#ifndef CLI_COMMON_H
#define CLI_COMMON_H

#include "solution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ACO_CLI_MODE_DEMO = 0,
  ACO_CLI_MODE_INSTANCE = 1,
} AcoCliMode;

typedef struct {
  AcoCliMode mode;
  const char *instance_path;
  int n;
  int K;
  int m;
  unsigned int seed;
  double alpha;
  double beta;
  double rho;
  double tau0;
  double Q;
} AcoCliOptions;

int cli_parse_int_arg(const char *s, int *out);
unsigned int cli_parse_uint_arg(const char *s, int *ok);
void cli_options_defaults(AcoCliOptions *options);
int cli_parse_solver_options(int argc, char **argv, AcoCliOptions *options);
void cli_print_usage(const char *program_name);
void cli_print_solution_routes(const Solution *best, int K);
void cli_print_solution_cost(double best_cost);
int cli_validate_solution_or_report(const Solution *best, int n, int K,
                                    const int *demands, int vehicle_capacity,
                                    double best_cost);

#ifdef __cplusplus
}
#endif

#endif
