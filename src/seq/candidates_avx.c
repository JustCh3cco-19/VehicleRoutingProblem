#include "aco.h"
#include "seq/internal.h"

#if defined(__AVX2__)
# include <immintrin.h>
#endif

void	update_score_row_alpha1(const int *restrict cand_row,
		const float *restrict eta_row, float *restrict score_row,
		const double *restrict tau_row, int k)
{
	int	t;

	t = 0;
#if defined(__AVX2__)
	for (; t + 4 <= k; t += 4)
	{
		__m128i idx = _mm_loadu_si128((const __m128i *)(cand_row + t));
		__m256d tau_v = _mm256_i32gather_pd(tau_row, idx, (int)sizeof(double));
		__m128 eta_f = _mm_loadu_ps(eta_row + t);
		__m256d eta_v = _mm256_cvtps_pd(eta_f);
		__m256d prod = _mm256_mul_pd(tau_v, eta_v);
		_mm_storeu_ps(score_row + t, _mm256_cvtpd_ps(prod));
	}
#endif
	while (t < k)
	{
		int node = cand_row[t];
		score_row[t] = (node > 0) ? (float)(tau_row[node]
				* (double)eta_row[t]) : 0.0f;
		t++;
	}
}

void	update_score_row_alpha2(const int *restrict cand_row,
		const float *restrict eta_row, float *restrict score_row,
		const double *restrict tau_row, int k)
{
	int	t;

	t = 0;
#if defined(__AVX2__)
	for (; t + 4 <= k; t += 4)
	{
		__m128i idx = _mm_loadu_si128((const __m128i *)(cand_row + t));
		__m256d tau_v = _mm256_i32gather_pd(tau_row, idx, (int)sizeof(double));
		__m256d tau2_v = _mm256_mul_pd(tau_v, tau_v);
		__m128 eta_f = _mm_loadu_ps(eta_row + t);
		__m256d eta_v = _mm256_cvtps_pd(eta_f);
		__m256d prod = _mm256_mul_pd(tau2_v, eta_v);
		_mm_storeu_ps(score_row + t, _mm256_cvtpd_ps(prod));
	}
#endif
	while (t < k)
	{
		int node = cand_row[t];
		if (node > 0)
		{
			double tau_val = tau_row[node];
			score_row[t] = (float)(tau_val * tau_val * (double)eta_row[t]);
		}
		else
		{
			score_row[t] = 0.0f;
		}
		t++;
	}
}
