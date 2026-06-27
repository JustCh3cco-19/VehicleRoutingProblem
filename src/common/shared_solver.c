#include "internal.h"
#include "solution.h"

int	shared_select_next(int current, const int *unvisited_nodes,
		int unvisited_count, double **tau, double **eta,
		double alpha, double beta, double roulette_r,
		double *candidate_scores, int *selected_index,
		t_score_cache *score_cache)
{
	const double	*score_row;
	double			*tau_row;
	double			*eta_row;
	double			denom;
	int				idx;
	int				node;
	double			score;
	double			threshold;
	double			cumulative;
	int				chosen_idx;

	if (unvisited_count <= 0)
	{
		if (selected_index)
			*selected_index = -1;
		return (0);
	}
	score_row = score_cache_get_row(score_cache, current, tau, eta, alpha, beta);
	tau_row = tau[current];
	eta_row = eta[current];
	denom = 0.0;
	idx = 0;
	while (idx < unvisited_count)
	{
		node = unvisited_nodes[idx];
		score = 0.0;
		if (score_row)
			score = score_row[node];
		else
			score = fast_pow_nonneg(tau_row[node], alpha)
				* fast_pow_nonneg(eta_row[node], beta);
		candidate_scores[idx] = score;
		denom += score;
		idx++;
	}
	threshold = roulette_r * denom;
	cumulative = 0.0;
	chosen_idx = unvisited_count - 1;
	idx = 0;
	while (idx < unvisited_count)
	{
		cumulative += candidate_scores[idx];
		if (cumulative >= threshold)
		{
			chosen_idx = idx;
			break ;
		}
		idx++;
	}
	if (selected_index)
		*selected_index = chosen_idx;
	return (unvisited_nodes[chosen_idx]);
}

bool	shared_build_ant_solution(
		t_solution *sol, int n, int k, double **tau, double **eta, double alpha,
		double beta, int vehicle_capacity_customers, t_score_cache *score_cache,
		unsigned int *rng_state, int *unvisited_nodes, double *candidate_scores,
		double *random_draws)
{
	int		route_customer_cap;
	int		i;
	int		unvisited_count;
	int		draw_index;
	int		vehicle;
	t_route	*r;
	int		current;
	int		assigned_customers;
	int		remaining_vehicles;
	int		future_capacity;
	double	roulette_r;
	int		selected_idx;
	int		next;
	t_route	*last;
	int		last_customers;

	solution_reset(sol);
	route_customer_cap = vehicle_capacity_customers;
	if (route_customer_cap <= 0)
		route_customer_cap = n;
	i = 0;
	while (i < n)
	{
		unvisited_nodes[i] = i + 1;
		random_draws[i] = rand01_state(rng_state);
		i++;
	}
	unvisited_count = n;
	draw_index = 0;
	vehicle = 1;
	while (vehicle <= k)
	{
		r = &sol->routes[vehicle - 1];
		if (!route_append(r, 0))
			return (false);
		current = 0;
		assigned_customers = 0;
		remaining_vehicles = k - vehicle;
		future_capacity = remaining_vehicles * route_customer_cap;
		while (unvisited_count > 0 && unvisited_count > future_capacity &&
assigned_customers < route_customer_cap)
		{
			roulette_r = random_draws[draw_index++];
			selected_idx = -1;
			next = shared_select_next(current, unvisited_nodes, unvisited_count,
					tau, eta, alpha, beta, roulette_r,
					candidate_scores, &selected_idx, score_cache);
			if (next <= 0)
				break ;
			if (!route_append(r, next))
				return (false);
			assigned_customers++;
			unvisited_count--;
			unvisited_nodes[selected_idx] = unvisited_nodes[unvisited_count];
			current = next;
		}
		if (!route_append(r, 0))
			return (false);
		vehicle++;
	}
	if (unvisited_count > 0)
	{
		last = &sol->routes[k - 1];
		if (last->len > 0 && last->nodes[last->len - 1] == 0)
			last->len--;
		last_customers = last->len > 0 ? (last->len - 1) : 0;
		while (unvisited_count > 0 && last_customers < route_customer_cap)
		{
			unvisited_count--;
			if (!route_append(last, unvisited_nodes[unvisited_count]))
				return (false);
			last_customers++;
		}
		if (!route_append(last, 0))
			return (false);
	}
	return (true);
}
