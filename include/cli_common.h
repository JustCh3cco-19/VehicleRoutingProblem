#ifndef CLI_COMMON_H
#define CLI_COMMON_H

#include "solution.h"

#ifdef __cplusplus
extern "C" {
#endif

int cli_parse_int_arg(const char *s, int *out);
unsigned int cli_parse_uint_arg(const char *s, int *ok);
void cli_print_solution_routes(const Solution *best, int K);
void cli_print_solution_cost(double best_cost);
int cli_validate_solution_or_report(const Solution *best, int n, int K,
                                    double best_cost);

#ifdef __cplusplus
}
#endif

#endif
