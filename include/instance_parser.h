#ifndef INSTANCE_PARSER_H
# define INSTANCE_PARSER_H

#include <stdio.h>

struct s_vrp_instance
{
	int						n;
	int						vehicles;
	int						capacity;
	double					*coords_x;
	double					*coords_y;
	int						*demands;
};
typedef struct s_vrp_instance	t_vrp_instance;

struct s_vrp_parse_state
{
	FILE					*f;
	const char				*path;
	t_vrp_instance			*instance;
	int						n;
	int						vehicles;
	int						capacity;
	int						have_coords;
	int						have_demands;
	int						have_capacity;
	int						have_vehicles;
};
typedef struct s_vrp_parse_state	t_vrp_parse_state;

void						vrp_instance_init(t_vrp_instance *instance);
void						vrp_instance_free(t_vrp_instance *instance);
int							vrp_load_tsplib_instance(const char *path,
								t_vrp_instance *instance);
int							vrp_instance_create_euc2d_matrix(
								const t_vrp_instance *instance,
								double ***c_out);
int							vrp_instance_create_float_coords(
								const t_vrp_instance *instance,
								float **coords_x_out,
								float **coords_y_out);

#endif
