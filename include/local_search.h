#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "aco.h"

#ifdef __cplusplus
extern "C" {
#endif

int local_search_shared_init(SeqShared *shared, int n, double **c, double beta);
void local_search_shared_free(SeqShared *shared);
void local_search_refine_solution_common(Solution *sol, int K,
                                         int vehicle_capacity_customers,
                                         double **c,
                                         const SeqShared *shared);

#ifdef __cplusplus
}
#endif

#endif
