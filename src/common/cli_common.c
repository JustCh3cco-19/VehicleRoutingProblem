#include "cli_common.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cli_parse_double_arg(const char *s, double *out) {
  char *end = NULL;
  double v;

  if (!s || !out) {
    return 0;
  }

  errno = 0;
  v = strtod(s, &end);
  if (errno != 0 || end == s || *end != '\0' || !isfinite(v)) {
    return 0;
  }
  *out = v;
  return 1;
}

int cli_parse_int_arg(const char *s, int *out) {
  char *end = NULL;
  long v;

  if (!s || !out) {
    return 0;
  }

  errno = 0;
  v = strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') {
    return 0;
  }
  if (v < 0 || v > 100000000L) {
    return 0;
  }
  *out = (int)v;
  return 1;
}

unsigned int cli_parse_uint_arg(const char *s, int *ok) {
  char *end = NULL;
  unsigned long v;

  if (!s || !ok) {
    if (ok) {
      *ok = 0;
    }
    return 0u;
  }

  errno = 0;
  v = strtoul(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0' || v > 0xFFFFFFFFUL) {
    *ok = 0;
    return 0u;
  }
  *ok = 1;
  return (unsigned int)v;
}

void cli_options_defaults(t_cli_options *options) {
  if (!options) {
    return;
  }

  options->mode = CLI_MODE_DEMO;
  options->instance_path = NULL;
  options->n = 5;
  options->k = 2;
  options->m = 10;
  options->seed = 1234u;
  options->alpha = 1.0;
  options->beta = 2.0;
  options->rho = 0.5;
  options->tau0 = 1.0;
  options->q = 1.0;
}

static int cli_parse_numeric_options(int argc, char **argv, int start,
                                     t_cli_options *options) {
  for (int i = start; i < argc; i += 2) {
    if (i + 1 >= argc) {
      return 0;
    }

    double value = 0.0;
    if (!cli_parse_double_arg(argv[i + 1], &value)) {
      return 0;
    }

    if (strcmp(argv[i], "--alpha") == 0) {
      options->alpha = value;
    } else if (strcmp(argv[i], "--beta") == 0) {
      options->beta = value;
    } else if (strcmp(argv[i], "--rho") == 0) {
      options->rho = value;
    } else if (strcmp(argv[i], "--tau0") == 0) {
      options->tau0 = value;
    } else if (strcmp(argv[i], "--q") == 0) {
      options->q = value;
    } else {
      return 0;
    }
  }

  return options->alpha > 0.0 && options->beta > 0.0 && options->rho > 0.0 &&
         options->rho < 1.0 && options->tau0 > 0.0 && options->q > 0.0;
}

int cli_parse_solver_options(int argc, char **argv, t_cli_options *options) {
  int ok = 1;
  int first_value = 1;

  if (!options) {
    return 0;
  }

  cli_options_defaults(options);
  if (argc == 1) {
    return 1;
  }

  if (cli_parse_int_arg(argv[1], &options->n)) {
    options->mode = CLI_MODE_DEMO;
    if (argc < 5) {
      return 0;
    }
    ok = ok && cli_parse_int_arg(argv[2], &options->k);
    ok = ok && cli_parse_int_arg(argv[3], &options->m);
    options->seed = cli_parse_uint_arg(argv[4], &ok);
    first_value = 5;
    if (!ok || options->n <= 0 || options->k <= 0 || options->m < 0) {
      return 0;
    }
  } else {
    options->mode = CLI_MODE_INSTANCE;
    if (argc < 4) {
      return 0;
    }
    options->instance_path = argv[1];
    ok = ok && cli_parse_int_arg(argv[2], &options->k);
    ok = ok && cli_parse_int_arg(argv[3], &options->m);
    first_value = 4;
    if (argc > 4 && argv[4][0] != '-') {
      options->seed = cli_parse_uint_arg(argv[4], &ok);
      first_value = 5;
    }
    if (!ok || options->k <= 0 || options->m < 0) {
      return 0;
    }
  }

  return cli_parse_numeric_options(argc, argv, first_value, options);
}

void cli_print_usage(const char *program_name) {
  fprintf(stderr, "usage: %s [n k m seed] [--alpha v --beta v --rho v "
                  "--tau0 v --q v]\n",
          program_name);
  fprintf(stderr, "usage: %s <instance.vrp> <k> <m> [seed] "
                  "[--alpha v --beta v --rho v --tau0 v --q v]\n",
          program_name);
}

void cli_print_solution_routes(const t_solution *best, int k) {
  for (int i = 0; i < k; ++i) {
    const t_route *r = &best->routes[i];
    int printed = 0;
    printf("Route %d:", i + 1);
    for (int t = 0; t < r->len; ++t) {
      int node = r->nodes[t];
      if (node != 0) {
        printf("%s%d", printed ? " " : " ", node);
        printed = 1;
      }
    }
    printf("\n");
  }
}

void cli_print_solution_cost(double best_cost) {
  printf("Cost: %.3f\n", best_cost);
  printf("best cost: %.3f\n", best_cost);
}

int cli_validate_solution_or_report(const t_solution *best, int n, int k,
                                    const int *demands, int vehicle_capacity,
                                    double best_cost) {
  char err[160];

  if (!isfinite(best_cost) || best_cost <= 0.0) {
    fprintf(stderr, "invalid solution cost: %.12g\n", best_cost);
    return 0;
  }

  int valid = demands ? solution_validate_cvrp(best, n, k, demands,
                                              vehicle_capacity, err,
                                              sizeof(err))
                      : solution_validate(best, n, k, err, sizeof(err));
  if (!valid) {
    fprintf(stderr, "invalid solution: %s\n", err);
    return 0;
  }

  return 1;
}
