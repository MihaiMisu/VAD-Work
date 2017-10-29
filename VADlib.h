#ifndef VADLIB_H_   /* Include guard */
#define VADLIB_H_

struct element;

struct Vector;

struct Vector *short_term_power(int *sgn, unsigned int sgn_len, unsigned int samp_frec, double frame_time, double stop_time, double alpha);
struct Vector *zero_crossing_rate(int *sgn, unsigned int sgn_len, unsigned int samp_frec, double frame_time, double stop_time, double alpha);
void zrmse(int *sgn, unsigned int sgn_len, unsigned int samp_frec, double frame_time, double stop_time);

#endif // FOO_H_