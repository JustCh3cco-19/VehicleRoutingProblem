# include "solver.h"
#include "seq/internal.h"

#if defined(__AVX2__)
# include <immintrin.h>

static void	update_avx_alpha1(t_score_row_params *params, int *t)
{
	__m128i	idx;
	__m256d	tau_v;
	__m128	eta_f;
	__m256d	eta_v;
	__m256d	prod;

	for (; *t + 4 <= params->k; *t += 4)
	{
		idx = _mm_loadu_si128((const __m128i *)(params->cand_row + *t));
		tau_v = _mm256_i32gather_pd(params->tau_row, idx, (int)sizeof(double));
		eta_f = _mm_loadu_ps(params->eta_row + *t);
		eta_v = _mm256_cvtps_pd(eta_f);
		prod = _mm256_mul_pd(tau_v, eta_v);
		_mm_storeu_ps(params->score_row + *t, _mm256_cvtpd_ps(prod));
	}
}
#endif

void	update_score_row_alpha1(t_score_row_params *params)
{
	int	t;
	int	node;

	t = 0;
#if defined(__AVX2__)
	update_avx_alpha1(params, &t);
#endif
	while (t < params->k)
	{
		node = params->cand_row[t];
		params->score_row[t] = (node > 0) ? (float)(params->tau_row[node]
				* (double)params->eta_row[t]) : 0.0f;
		t++;
	}
}

#if defined(__AVX2__)
static void	update_avx_alpha2(t_score_row_params *params, int *t)
{
	__m128i	idx;
	__m256d	tau_v;
	__m256d	tau2_v;
	__m128	eta_f;
	__m256d	eta_v;

	for (; *t + 4 <= params->k; *t += 4)
	{
		idx = _mm_loadu_si128((const __m128i *)(params->cand_row + *t));
		tau_v = _mm256_i32gather_pd(params->tau_row, idx, (int)sizeof(double));
		tau2_v = _mm256_mul_pd(tau_v, tau_v);
		eta_f = _mm_loadu_ps(params->eta_row + *t);
		eta_v = _mm256_cvtps_pd(eta_f);
		tau2_v = _mm256_mul_pd(tau2_v, eta_v);
		_mm_storeu_ps(params->score_row + *t, _mm256_cvtpd_ps(tau2_v));
	}
}
#endif

void	update_score_row_alpha2(t_score_row_params *params)
{
	int		t;
	int		node;
	double	tau_val;

	t = 0;
#if defined(__AVX2__)
	update_avx_alpha2(params, &t);
#endif
	while (t < params->k)
	{
		node = params->cand_row[t];
		if (node > 0)
		{
			tau_val = params->tau_row[node];
			params->score_row[t] = (float)(tau_val * tau_val
					* (double)params->eta_row[t]);
		}
		else
			params->score_row[t] = 0.0f;
		t++;
	}
}
